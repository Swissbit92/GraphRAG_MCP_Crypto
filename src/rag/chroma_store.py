# src/rag/chroma_store.py
import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional, Tuple, Union

import requests
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log = logging.getLogger("chroma_rag")
if not log.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("RAG_MCP_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] chroma_rag: %(message)s",
    )


# -----------------------------------------------------------------------------
# Embedding providers
# -----------------------------------------------------------------------------
class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Chroma-compatible embedding wrapper for Ollama.

    Compatible with Chroma's expectations:
      - embed_documents(self, input: List[str]) -> List[List[float]]
      - embed_query(self, input: List[str]) -> List[List[float]]
      - __call__(self, input: Union[str, List[str]]) -> List[List[float]]  (safety)

    Env:
      OLLAMA_BASE=http://127.0.0.1:11434
      OLLAMA_EMBED_MODEL=nomic-embed-text
      OLLAMA_EMBED_TIMEOUT=60           # optional (seconds)
    """

    def __init__(self, base: Optional[str] = None, model: Optional[str] = None, timeout: Optional[float] = None):
        self.base = (base or os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.timeout = float(timeout or os.getenv("OLLAMA_EMBED_TIMEOUT", 60))
        self.session = requests.Session()
        log.info("Using Ollama embeddings model=%s base=%s", self.model, self.base)

    def _embed_one(self, text: str) -> List[float]:
        url = f"{self.base}/api/embeddings"
        try:
            r = self.session.post(url, json={"model": self.model, "prompt": text}, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"Ollama embedding request failed: {e}") from e

        vec = data.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise RuntimeError(f"Ollama returned no embedding (len={len(text)}) payload={str(data)[:200]}")
        return vec

    # --- Chroma expects these names -------------------------------------------------
    def embed_documents(self, input: List[str]) -> List[List[float]]:  # type: ignore[override]
        if not isinstance(input, list):
            raise TypeError("embed_documents expected List[str]")
        return [self._embed_one(t) for t in input]

    def embed_query(self, input: List[str]) -> List[List[float]]:  # type: ignore[override]
        if not isinstance(input, list):
            raise TypeError("embed_query expected List[str]")
        return [self._embed_one(t) for t in input]

    # --- Safety: some Chroma versions may call the function directly ---------------
    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(input, str):
            return [self._embed_one(input)]
        if isinstance(input, list):
            return [self._embed_one(x) for x in input]
        raise TypeError("__call__ expected str or List[str]")


def build_embedding_function():
    """
    Priority: OLLAMA -> sentence-transformers -> None (caller supplies embeddings).

    - USE_OLLAMA_EMBED=true/1/yes enables Ollama embeddings (default true)
    - If SENTENCE_TRANSFORMER_MODEL is set, use SentenceTransformerEmbeddingFunction
    - Otherwise return None (explicit embeddings must be provided)
    """
    use_ollama = os.getenv("USE_OLLAMA_EMBED", "true").lower() in ("1", "true", "yes")
    if use_ollama:
        try:
            return OllamaEmbeddingFunction()
        except Exception as e:
            # Be explicit; this avoids silent None → AssertionError deeper in Chroma
            raise RuntimeError(f"Failed to initialize OllamaEmbeddingFunction: {e}") from e

    model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL")
    if model_name:
        log.info("Using SentenceTransformer embeddings: %s", model_name)
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    log.warning("No embedding function configured (USE_OLLAMA_EMBED=false and no SENTENCE_TRANSFORMER_MODEL).")
    return None


# -----------------------------------------------------------------------------
# RAG store
# -----------------------------------------------------------------------------
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class ChromaRAG:
    """
    Thin wrapper around Chroma for GraphRAG.
    - Persists to CHROMA_DIR (default .chroma)
    - Creates/gets a collection (default 'whitepapers')
    - Upserts chunks; if expand_per_entity=True, writes one row per (chunk x entity)
      with a scalar 'entity_id' for robust filtering.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection: str = "whitepapers",
        embedding_fn=None,
        expand_per_entity: Optional[bool] = None,
    ):
        persist_dir = persist_dir or os.getenv("CHROMA_DIR", ".chroma")
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))

        # If embedding_fn is None, decide based on env (Ollama → ST → None)
        self.embedding_fn = embedding_fn if embedding_fn is not None else build_embedding_function()

        # default True to enable scalar filter on `entity_id`
        self.expand_per_entity = (
            expand_per_entity
            if expand_per_entity is not None
            else os.getenv("RAG_EXPAND_PER_ENTITY", "true").lower() in ("1", "true", "yes")
        )

        # Important: attach the embedding function to the collection
        # so Chroma performs embedding on upsert/query.
        self.col = self.client.get_or_create_collection(
            name=collection,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        log.info(
            "ChromaRAG ready: dir=%s collection=%s expand_per_entity=%s has_embed_fn=%s",
            persist_dir, collection, self.expand_per_entity, bool(self.embedding_fn),
        )

    # -- API --------------------------------------------------------------------
    def upsert_chunks(
        self,
        items: Iterable[Dict[str, Any]],
        id_prefix: str = "chunk:",
        require_embeddings: bool = False,
    ):
        """
        items: iterable of dicts with keys:
            - id (optional)
            - text (required)
            - metadata: {
                 doc_id, chunk_id, entity_ids: [full IRIs],
                 section_type, page, sha1, embed_model
              }

        If expand_per_entity=True:
          We write one row per (chunk x entity), with metadata.entity_id=<IRI> (scalar)
          and keep entity_ids as JSON string in metadata.entity_ids_json (no lists in metadata).
          Also write entity_ids_count (int).

        If no entities present, we write a single row with entity_id=None.
        """
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []

        for it in items:
            text = (it.get("text") or "").strip()
            if not text:
                continue
            md_in = dict(it.get("metadata", {}) or {})
            doc_id = md_in.get("doc_id") or ""
            chunk_id = md_in.get("chunk_id") or ""
            sha1 = md_in.get("sha1") or ""
            entity_ids: List[str] = md_in.get("entity_ids") or []

            # sanitize base metadata: ONLY scalars
            base_meta: Dict[str, Any] = {
                "doc_id": doc_id or None,
                "chunk_id": chunk_id or None,
                "section_type": md_in.get("section_type"),
                "page": md_in.get("page"),
                "sha1": sha1 or None,
                "embed_model": md_in.get("embed_model"),
                # store the array as a JSON string + a count for easy inspection
                "entity_ids_json": json.dumps(entity_ids) if entity_ids else "[]",
                "entity_ids_count": int(len(entity_ids)),
            }

            if self.expand_per_entity:
                expanded = entity_ids if entity_ids else [None]
                for e in expanded:
                    entity_id_scalar = e
                    row_id = it.get("id") or _sha1(f"{doc_id}/{chunk_id}/{sha1}/{entity_id_scalar or 'none'}")
                    ids.append(f"{id_prefix}{row_id}")
                    docs.append(text)
                    row_meta = dict(base_meta)
                    row_meta["entity_id"] = entity_id_scalar  # <-- scalar, filterable
                    metas.append(row_meta)
            else:
                # single row; no scalar entity_id available
                row_id = it.get("id") or _sha1(f"{doc_id}/{chunk_id}/{sha1}")
                ids.append(f"{id_prefix}{row_id}")
                docs.append(text)
                metas.append(base_meta)

        if not ids:
            log.info("upsert_chunks: nothing to upsert")
            return

        if self.embedding_fn is None and require_embeddings:
            raise RuntimeError(
                "No embedding function configured (embedding_fn=None) and require_embeddings=True. "
                "Set USE_OLLAMA_EMBED=true or SENTENCE_TRANSFORMER_MODEL, "
                "or pass an explicit embedding_fn to ChromaRAG."
            )

        t0 = time.perf_counter()
        self.col.upsert(ids=ids, documents=docs, metadatas=metas)
        took_ms = int((time.perf_counter() - t0) * 1000)
        log.info("Upserted %s rows into '%s' in %sms", len(ids), self.col.name, took_ms)

    def query(
        self,
        text: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        k: int = 8,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Hybrid query:
          - If text is provided and we have an embedding_fn -> semantic query
          - Otherwise use .get() (metadata filter only)
          - entity_ids -> builds scalar filter over 'entity_id'
          - where_document -> supports {'$contains':'term'} etc.

        VALID include keys: 'documents', 'embeddings', 'metadatas', 'distances', 'uris', 'data'
        """
        where = dict(where or {})
        include = include or ["metadatas", "documents", "distances"]

        # Build entity filter on scalar 'entity_id'
        if entity_ids:
            if len(entity_ids) == 1:
                where.update({"entity_id": {"$eq": entity_ids[0]}})
            else:
                where.update({"$or": [{"entity_id": {"$eq": e}} for e in entity_ids]})

        # No embeddings path -> use .get()
        if not text or self.embedding_fn is None:
            res = self.col.get(
                where=where if where else None,
                where_document=where_document if where_document else None,
                limit=k,
                include=include,   # avoid "ids" to keep payload compact
            )
            return res

        # Embedding-backed semantic query
        res = self.col.query(
            query_texts=[text],
            n_results=k,
            where=where if where else None,
            where_document=where_document if where_document else None,
            include=include,     # avoid "ids" to keep payload compact
        )
        return res


# -----------------------------------------------------------------------------
# Convenience loader for your pipeline outputs
# -----------------------------------------------------------------------------
def iter_pipeline_records(
    labels_dir: Path,
    chunks_dir: Path,
) -> Iterable[Dict[str, Any]]:
    """
    Joins labels (JSONL) with chunk texts by (doc_id, chunk_id).

    Expects:
      labels_dir/*.labels.jsonl
      chunks_dir/*.chunks.jsonl   -> records with keys: doc_id, chunk_id, text, page?, sha1?
    """
    chunk_index: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for chunk_file in sorted(chunks_dir.glob("*.chunks.jsonl")):
        with chunk_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = (str(rec.get("doc_id", "")), str(rec.get("chunk_id", "")))
                if key[0] and key[1]:
                    chunk_index[key] = rec

    # Local import to avoid circulars on module import
    from src.kg.namespaces import iri_entity

    for lab_file in sorted(labels_dir.glob("*.labels.jsonl")):
        with lab_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    lab = json.loads(line)
                except Exception:
                    continue

                key = (str(lab.get("doc_id", "")), str(lab.get("chunk_id", "")))
                chunk = chunk_index.get(key) or {}
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue

                # Build entity_ids array of full IRIs
                entity_ids: List[str] = []
                ents = lab.get("entities", {}) or {}
                for kind, vals in ents.items():
                    for v in vals or []:
                        v_clean = (v or "").strip()
                        if not v_clean:
                            continue
                        entity_ids.append(iri_entity(kind, v_clean)[1:-1])  # strip < >

                # Pick a single section_type if list
                section_type = lab.get("section_type")
                if isinstance(section_type, list) and section_type:
                    section_type = section_type[0]

                yield {
                    "id": None,  # computed deterministically in upsert
                    "text": text,
                    "metadata": {
                        "doc_id": lab.get("doc_id"),
                        "chunk_id": lab.get("chunk_id"),
                        "entity_ids": entity_ids,  # will be serialized in upsert
                        "section_type": section_type,
                        "page": chunk.get("page"),
                        "sha1": chunk.get("sha1"),
                        "embed_model": os.getenv(
                            "EMBED_MODEL_NAME",
                            os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                        ),
                    },
                }


# -----------------------------------------------------------------------------
# Helper for pipeline hook
# -----------------------------------------------------------------------------
def build_rag_index(outputs_dir: str, collection: str = "whitepapers"):
    """
    Index the latest pipeline outputs into Chroma.
    """
    out = Path(outputs_dir)
    labels_dir = out / "labels"
    chunks_dir = out / "docs"

    rag = ChromaRAG(
        persist_dir=os.getenv("CHROMA_DIR", ".chroma"),
        collection=collection,
        # If RAG_DISABLE_EMBED=true, force None and let require_embeddings be handled by caller
        embedding_fn=None if os.getenv("RAG_DISABLE_EMBED", "false").lower() in ("1", "true", "yes") else None,
        # If embedding_fn is None here, build_embedding_function() decides (Ollama or ST)
    )
    rag.upsert_chunks(iter_pipeline_records(labels_dir, chunks_dir))
