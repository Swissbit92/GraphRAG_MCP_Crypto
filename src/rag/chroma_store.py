# src/rag/chroma_store.py
# RAG store using ChromaDB (https://www.trychroma.com/)
# Supports Ollama local embeddings or SentenceTransformers models.

import os
import json
import hashlib
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import requests


# -------- Embedding providers -------------------------------------------------

class OllamaEmbeddingFunction:
    """
    Minimal local embedding via Ollama REST API.

    Env:
      USE_OLLAMA_EMBED=true|false          (switch, default true)
      OLLAMA_BASE=http://127.0.0.1:11434
      OLLAMA_EMBED_MODEL=nomic-embed-text  (or another local embed model)
    """
    def __init__(self, base: Optional[str] = None, model: Optional[str] = None):
        self.base = (base or os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.session = requests.Session()

    def name(self) -> str:
        return f"ollama:{self.model}"

    # Chroma ≥ 0.4.16 expects __call__(self, input)
    def __call__(self, input: List[str]) -> List[List[float]]:
        url = f"{self.base}/api/embeddings"
        out: List[List[float]] = []
        for t in input:
            resp = self.session.post(url, json={"model": self.model, "prompt": t})
            resp.raise_for_status()
            data = resp.json()
            vec = data.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError(f"Ollama embedding failed: {data}")
            out.append(vec)
        return out


def build_embedding_function():
    """
    Priority: OLLAMA -> sentence-transformers -> None (caller must pass embeddings).
    """
    use_ollama = os.getenv("USE_OLLAMA_EMBED", "true").lower() in ("1", "true", "yes")
    if use_ollama:
        return OllamaEmbeddingFunction()

    model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL")
    if model_name:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    return None


# -------- RAG store ----------------------------------------------------------

class ChromaRAG:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection: str = "whitepapers",
        embedding_fn=None,
    ):
        persist_dir = persist_dir or os.getenv("CHROMA_DIR", ".chroma")
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
        self.embedding_fn = embedding_fn if embedding_fn is not None else build_embedding_function()

        # Optional: clean recreate if embedding config changed
        reset = os.getenv("CHROMA_RESET", "false").lower() in ("1", "true", "yes")
        if reset:
            try:
                self.client.delete_collection(name=collection)
            except Exception:
                pass  # fine if doesn't exist

        self.col = self.client.get_or_create_collection(
            name=collection,
            embedding_function=None if self.embedding_fn is None else self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _normalize_id(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    @staticmethod
    def _expand_item_for_entities(
        item: Dict[str, Any],
        base_id: str,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Turn a single item (with metadata.entity_ids list) into N records,
        each with a scalar 'entity_id' in metadata. When no entities, create one with '__none__'.
        Returns: list of (id, text, metadata)
        """
        text = (item.get("text") or "").strip()
        if not text:
            return []

        md = dict(item.get("metadata", {}) or {})
        # Pull list of entity_ids if present, else treat as no-entity
        entity_ids_list: List[str] = md.pop("entity_ids", []) or []
        # Always keep original chunk/doc ids
        doc_id = md.get("doc_id")
        chunk_id = md.get("chunk_id")

        out: List[Tuple[str, str, Dict[str, Any]]] = []
        if not entity_ids_list:
            md1 = dict(md)
            md1["entity_id"] = "__none__"  # scalar placeholder
            out.append((base_id, text, md1))
            return out

        for i, eid in enumerate(entity_ids_list):
            md_i = dict(md)
            md_i["entity_id"] = str(eid)
            # keep chunk/doc context for grouping later
            md_i["doc_id"] = doc_id
            md_i["chunk_id"] = chunk_id
            out.append((f"{base_id}#e{i}", text, md_i))
        return out

    # -- API --------------------------------------------------------------------
    def upsert_chunks(
        self,
        items: Iterable[Dict[str, Any]],
        id_prefix: str = "chunk:",
        require_embeddings: bool = False,
    ):
        """
        items: iterable of dicts with keys:
            - id (optional) | we generate one from (doc_id, chunk_id, sha1) if absent
            - text (required)
            - metadata: { doc_id, chunk_id, entity_ids: [full IRIs], section_type, page, sha1, embed_model }
        """
        ids, docs, metadatas = [], [], []
        for it in items:
            md  = it.get("metadata", {}) or {}
            base = it.get("id")
            if not base:
                base = self._normalize_id(f"{md.get('doc_id','')}/{md.get('chunk_id','')}/{md.get('sha1','')}")
            base = f"{id_prefix}{base}"

            expanded = self._expand_item_for_entities(it, base)
            for rid, text, rmd in expanded:
                # Chroma requires scalar metadata values
                for k, v in list(rmd.items()):
                    if isinstance(v, (list, dict)):
                        rmd[k] = json.dumps(v, ensure_ascii=False)
                ids.append(rid)
                docs.append(text)
                metadatas.append(rmd)

        if not ids:
            return

        if self.embedding_fn is None and require_embeddings:
            raise RuntimeError("No embedding function configured, and require_embeddings=True.")

        self.col.upsert(ids=ids, documents=docs, metadatas=metadatas)

    def query(
        self,
        text: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        k: int = 8,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Basic retrieval. If entity_ids provided, filter by scalar 'entity_id'.
        """
        where = dict(where or {})
        if entity_ids:
            vals = [str(e) for e in entity_ids if e]
            if len(vals) == 1:
                where.update({"entity_id": {"$in": vals}})
            elif vals:
                where["$or"] = [{"entity_id": {"$in": [v]}} for v in vals]

        res = self.col.query(
            query_texts=[text] if text else None,
            n_results=k,
            where=where if where else None,
        )
        return res


# -------- convenience loader for your pipeline outputs -----------------------

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
    # Build a small in-memory index of chunk texts for the current file batch
    chunk_index: Dict[tuple, Dict[str, Any]] = {}

    for chunk_file in sorted(chunks_dir.glob("*.chunks.jsonl")):
        with chunk_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = (rec.get("doc_id"), rec.get("chunk_id"))
                if key[0] and key[1]:
                    chunk_index[key] = rec

    for lab_file in sorted(labels_dir.glob("*.labels.jsonl")):
        with lab_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    lab = json.loads(line)
                except Exception:
                    continue
                key = (lab.get("doc_id"), lab.get("chunk_id"))
                chunk = chunk_index.get(key) or {}
                text = chunk.get("text", "")
                if not text:
                    continue

                # Build entity_ids as full IRIs from (kind, value)
                entity_ids: List[str] = []
                ents = lab.get("entities", {}) or {}
                for kind, vals in ents.items():
                    for v in vals or []:
                        v_clean = (v or "").strip()
                        if not v_clean:
                            continue
                        from ..kg.namespaces import iri_entity
                        entity_ids.append(iri_entity(kind, v_clean)[1:-1])  # strip < >

                yield {
                    "id": None,  # auto from hash
                    "text": text,
                    "metadata": {
                        "doc_id": lab.get("doc_id"),
                        "chunk_id": lab.get("chunk_id"),
                        "entity_ids": entity_ids,  # list is OK here; we expand to scalar in upsert
                        "section_type": (lab.get("section_type") or [None])[0]
                            if isinstance(lab.get("section_type"), list)
                            else lab.get("section_type"),
                        "page": chunk.get("page"),
                        "sha1": chunk.get("sha1"),
                        "embed_model": os.getenv("EMBED_MODEL_NAME", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")),
                    },
                }
