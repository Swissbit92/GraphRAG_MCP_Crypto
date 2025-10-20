# src/mcp/rag_server.py
from __future__ import annotations

import logging
import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple
import requests

from pydantic import BaseModel, Field, model_validator
from fastmcp import FastMCP

from src.rag.chroma_store import ChromaRAG, iter_pipeline_records, build_rag_index

# Optional: prefixes for KG SPARQL enrichment
try:
    from src.kg.namespaces import sparql_prefix_block
except Exception:  # pragma: no cover
    sparql_prefix_block = lambda: ""  # best-effort if KG module is absent

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("RAG_MCP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] rag_mcp: %(message)s",
)
log = logging.getLogger("rag_mcp")

# -----------------------------------------------------------------------------
# Env / defaults
# -----------------------------------------------------------------------------
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "whitepapers")
DEFAULT_OUTPUTS_DIR = os.getenv("RAG_OUTPUTS_DIR", "outputs/run_simple")

# KG enrichment toggles
QA_KG_ENRICH = os.getenv("QA_KG_ENRICH", "true").strip().lower() in ("1", "true", "yes")
GRAPHDB_URL = os.getenv("GRAPHDB_URL", "http://localhost:7200").rstrip("/")
GRAPHDB_REPOSITORY = os.getenv("GRAPHDB_REPOSITORY", "mcp_kg")
GRAPHDB_USERNAME = os.getenv("GRAPHDB_USERNAME")
GRAPHDB_PASSWORD = os.getenv("GRAPHDB_PASSWORD")

# LLM
QA_LLM_MODEL = os.getenv("QA_LLM_MODEL") or os.getenv("OLLAMA_MODEL") or "llama3.1:latest"
QA_LLM_MODE = os.getenv("QA_LLM_MODE", "").strip().lower()  # "mock" to force offline
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")

# -----------------------------------------------------------------------------
# Pydantic schemas for existing tools (unchanged)
# -----------------------------------------------------------------------------
class SearchInput(BaseModel):
    text: Optional[str] = Field(
        None,
        description="Semantic query text. If omitted, retrieval can rely on metadata filters only.",
    )
    entity_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter by one or more KG entity IRIs (matches metadata.entity_id).",
    )
    k: int = Field(default=8, ge=1, le=100, description="Max results to return.")
    where: Optional[Dict[str, Any]] = Field(
        default=None, description="Chroma metadata filter (passed through)."
    )
    where_document: Optional[Dict[str, Any]] = Field(
        default=None, description="Document text filter, e.g. {'$contains':'consensus'}."
    )
    include: Optional[List[str]] = Field(
        default=None,
        description="Chroma include fields; defaults to ['metadatas','documents','distances']",
    )
    collection: Optional[str] = Field(
        default=None, description=f"Override collection (default '{DEFAULT_COLLECTION}')."
    )


class ReindexInput(BaseModel):
    outputs_dir: str = Field(
        default=DEFAULT_OUTPUTS_DIR,
        description="Pipeline outputs directory containing /labels and /docs.",
    )
    collection: Optional[str] = Field(
        default=None, description="Target Chroma collection (default from env)."
    )


class OneRecord(BaseModel):
    id: Optional[str] = Field(
        default=None,
        description="Stable id; if omitted the server/ChromaRAG can derive a deterministic id.",
    )
    text: str = Field(description="Raw chunk text to embed and index.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scalar metadata (doc_id, chunk_id, section_type, page, sha1, entity_ids[], embed_model, etc.).",
    )


class EmbedAndIndexInput(BaseModel):
    records: Optional[List[OneRecord]] = Field(
        default=None, description="Direct records to index (text + metadata)."
    )
    labels_dir: Optional[str] = Field(
        default=None, description="If provided with chunks_dir, server will iterate pipeline records."
    )
    chunks_dir: Optional[str] = Field(
        default=None, description="Directory with *.chunks.jsonl (paired with labels_dir)."
    )
    collection: Optional[str] = Field(default=None, description="Target collection.")
    require_embeddings: bool = Field(
        default=False,
        description="If True and no embedding function is configured, fail instead of upserting.",
    )

    @model_validator(mode="after")
    def _ensure_mode(self) -> "EmbedAndIndexInput":
        if not self.records and not (self.labels_dir and self.chunks_dir):
            raise ValueError("Provide either `records` OR both `labels_dir` and `chunks_dir`.")
        return self


class DeleteInput(BaseModel):
    ids: Optional[List[str]] = Field(
        default=None, description="Exact ids to delete (server auto-adds 'chunk:' if missing)."
    )
    where: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata filter for deletion (e.g., {'doc_id': {'$eq': 'paper-123'}})."
    )
    collection: Optional[str] = Field(default=None, description="Target collection.")

    @model_validator(mode="after")
    def _ids_or_where(self) -> "DeleteInput":
        if not self.ids and not self.where:
            raise ValueError("Provide at least `ids` or `where` to delete.")
        return self


class HealthOutput(BaseModel):
    ok: bool
    collection: str
    persist_dir: str
    count: int
    has_embeddings: bool
    expand_per_entity: bool

# -----------------------------------------------------------------------------
# NEW: QA schemas
# -----------------------------------------------------------------------------
class CitationItem(BaseModel):
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    entity_ids: List[str] = Field(default_factory=list)
    text: str


class QAInput(BaseModel):
    question: str = Field(description="User question to answer from the corpus.")
    entity_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of KG entity IRIs to focus retrieval ('entity_id' scalar filter).",
    )
    k: int = Field(default=8, ge=1, le=32, description="How many chunks to retrieve.")
    where: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional Chroma metadata filter."
    )
    where_document: Optional[Dict[str, Any]] = Field(
        default=None, description="Document text filter for Chroma."
    )
    collection: Optional[str] = Field(
        default=None, description=f"Chroma collection (default '{DEFAULT_COLLECTION}')."
    )
    llm_model: Optional[str] = Field(
        default=None, description="Override model for synthesis (default QA_LLM_MODEL/OLLAMA_MODEL)."
    )
    use_mock_llm: Optional[bool] = Field(
        default=None,
        description="Force a deterministic offline answer (no network). Overrides model.",
    )
    kg_enrich: Optional[bool] = Field(
        default=None, description="If true, query KG for labels/aliases of referenced entities."
    )


class QAOutput(BaseModel):
    answer: str
    citations: List[CitationItem]
    took_ms: int
    model_used: str
    debug: Optional[Dict[str, Any]] = None

# -----------------------------------------------------------------------------
# Server bootstrap
# -----------------------------------------------------------------------------
mcp = FastMCP("rag")

# Simple per-collection instance cache so all tools share the same ChromaRAG
_RAG_CACHE: Dict[str, ChromaRAG] = {}

def _get_rag(collection: Optional[str] = None) -> ChromaRAG:
    """Construct (or reuse) a ChromaRAG instance honoring env defaults."""
    col = collection or DEFAULT_COLLECTION
    inst = _RAG_CACHE.get(col)
    if inst is not None:
        return inst
    log.debug(f"Init ChromaRAG persist_dir={DEFAULT_CHROMA_DIR} collection={col}")
    inst = ChromaRAG(
        persist_dir=DEFAULT_CHROMA_DIR,
        collection=col,
        embedding_fn=None,          # let ChromaRAG decide based on env (Ollama, etc.)
        expand_per_entity=None,     # default from env (e.g., RAG_EXPAND_PER_ENTITY)
    )
    _RAG_CACHE[col] = inst
    return inst

# -----------------------------------------------------------------------------
# Helpers: normalize Chroma results → citations
# -----------------------------------------------------------------------------
def _norm_citations(chroma_res: Dict[str, Any]) -> List[CitationItem]:
    """
    Chroma .query includes arrays (possibly with one batch). We normalize
    into a flat list of CitationItem.
    """
    docs = chroma_res.get("documents") or []
    metas = chroma_res.get("metadatas") or []

    # Chroma returns [ [..] ] if query_texts provided; support both shapes.
    if docs and isinstance(docs[0], list):
        docs = docs[0]
    if metas and isinstance(metas[0], list):
        metas = metas[0]

    out: List[CitationItem] = []
    for d, m in zip(docs, metas):
        if not isinstance(d, str) or not isinstance(m, dict):
            continue
        out.append(
            CitationItem(
                doc_id=(m.get("doc_id") or None),
                chunk_id=(m.get("chunk_id") or None),
                entity_ids=json.loads(m.get("entity_ids_json", "[]")) if m.get("entity_ids_json") else (
                    [m["entity_id"]] if m.get("entity_id") else []
                ),
                text=d.strip(),
            )
        )
    return out

# -----------------------------------------------------------------------------
# Helpers: KG enrichment (labels/aliases)
# -----------------------------------------------------------------------------
def _kg_auth():
    if GRAPHDB_USERNAME and GRAPHDB_PASSWORD:
        return requests.auth.HTTPBasicAuth(GRAPHDB_USERNAME, GRAPHDB_PASSWORD)
    return None

def _kg_enrich_aliases(entity_iris: List[str]) -> Dict[str, List[str]]:
    """
    Return map: IRI -> [labels/altLabels]. Best-effort; failures yield {}.
    """
    if not entity_iris:
        return {}
    try:
        pref = sparql_prefix_block()
    except Exception:
        pref = ""

    # Build VALUES clause
    iris_clause = " ".join(f"<{iri}>" for iri in entity_iris[:128])  # clamp
    q = f"""{pref}
SELECT ?s ?label
WHERE {{
  VALUES ?s {{ {iris_clause} }}
  OPTIONAL {{ ?s rdfs:label ?l . BIND(STR(?l) AS ?label) }}
  OPTIONAL {{ ?s skos:altLabel ?al . BIND(STR(?al) AS ?label) }}
  FILTER(BOUND(?label))
}}
"""
    url = f"{GRAPHDB_URL}/repositories/{GRAPHDB_REPOSITORY}"
    try:
        r = requests.get(url, params={"query": q}, headers={"Accept": "application/sparql-results+json"},
                         auth=_kg_auth(), timeout=20)
        r.raise_for_status()
        data = r.json()
        out: Dict[str, List[str]] = {}
        for b in data.get("results", {}).get("bindings", []):
            s = b.get("s", {}).get("value")
            lbl = b.get("label", {}).get("value")
            if not s or not lbl:
                continue
            out.setdefault(s, [])
            if lbl not in out[s]:
                out[s].append(lbl)
        return out
    except Exception as e:  # pragma: no cover
        log.warning("KG enrich failed: %s", e)
        return {}

# -----------------------------------------------------------------------------
# Helpers: LLM synthesis (mock or Ollama)
# -----------------------------------------------------------------------------
def _mock_answer(question: str, citations: List[CitationItem], kg_notes: Dict[str, List[str]]) -> str:
    """
    Deterministic offline answer: use the top snippet(s) and any KG labels.
    """
    snippets = []
    for i, c in enumerate(citations[:2], 1):
        snippets.append(f"[{i}] {c.text[:300].strip()}")
    ent_labels = []
    for iri, labels in sorted(kg_notes.items()):
        if labels:
            ent_labels.append(f"- {iri} aka {', '.join(labels[:3])}")
    parts = []
    parts.append(f"Q: {question}")
    if ent_labels:
        parts.append("Known entities:\n" + "\n".join(ent_labels))
    if snippets:
        parts.append("From the corpus:\n" + "\n\n".join(snippets))
    parts.append("Answer: Based on the retrieved context above.")
    return "\n\n".join(parts)

def _ollama_generate(model: str, prompt: str, base: str = OLLAMA_BASE, temperature: float = 0.2) -> str:
    """
    Minimal Ollama /api/generate call. Returns the final 'response' string.
    """
    url = f"{base}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    resp = data.get("response") or ""
    return resp.strip()

def _build_prompt(question: str, citations: List[CitationItem], kg_notes: Dict[str, List[str]]) -> str:
    """
    Compose a concise prompt with numbered snippets and (optional) KG aliases.
    """
    lines = []
    lines.append("You are a precise crypto research assistant. Answer concisely and cite snippets as [1], [2], ...")
    if kg_notes:
        lines.append("\nEntity background (from KG):")
        for iri, labels in kg_notes.items():
            if labels:
                lines.append(f"- {iri}: aliases = {', '.join(labels[:5])}")
    lines.append("\nContext snippets:")
    for i, c in enumerate(citations, 1):
        # Keep the prompt compact
        snippet = c.text.replace("\n", " ").strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + " ..."
        lines.append(f"[{i}] {snippet}")
    lines.append(f"\nQuestion: {question}")
    lines.append("Answer (with bracket citations):")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Plain, test-callable implementations
# -----------------------------------------------------------------------------
def rag_search_impl(inp: SearchInput) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rag = _get_rag(inp.collection)
    res = rag.query(
        text=inp.text,
        entity_ids=inp.entity_ids,
        k=inp.k,
        where=inp.where,
        where_document=inp.where_document,
        include=inp.include or ["metadatas", "documents", "distances"],
    )
    took_ms = int((time.perf_counter() - t0) * 1000)

    docs = res.get("documents")
    if isinstance(docs, list) and docs and isinstance(docs[0], list):
        n = len(docs[0])
    elif isinstance(docs, list):
        n = len(docs)
    else:
        n = 0

    log.info(
        "rag.search k=%s text=%s entities=%s -> %s in %sms",
        inp.k, bool(inp.text), len(inp.entity_ids or []), n, took_ms
    )
    return {"results": res, "took_ms": took_ms}


def rag_reindex_impl(inp: ReindexInput) -> Dict[str, Any]:
    collection = inp.collection or DEFAULT_COLLECTION
    log.info("rag.reindex outputs_dir=%s collection=%s", inp.outputs_dir, collection)
    build_rag_index(outputs_dir=inp.outputs_dir, collection=collection)
    return {"ok": True, "collection": collection, "outputs_dir": inp.outputs_dir}


def rag_embed_and_index_impl(inp: EmbedAndIndexInput) -> Dict[str, Any]:
    rag = _get_rag(inp.collection)

    if inp.records:
        items = [{"id": r.id, "text": r.text, "metadata": r.metadata} for r in inp.records]
        rag.upsert_chunks(items, require_embeddings=inp.require_embeddings)
        log.info("rag.embed_and_index upserted records=%s", len(items))
        return {"ok": True, "mode": "records", "count": len(items)}

    from pathlib import Path
    labels_dir = Path(inp.labels_dir)  # type: ignore[arg-type]
    chunks_dir = Path(inp.chunks_dir)  # type: ignore[arg-type]
    items_iter = iter_pipeline_records(labels_dir, chunks_dir)
    rag.upsert_chunks(items_iter, require_embeddings=inp.require_embeddings)
    log.info("rag.embed_and_index upserted from labels_dir=%s chunks_dir=%s", labels_dir, chunks_dir)
    return {"ok": True, "mode": "pipeline_dirs", "labels_dir": str(labels_dir), "chunks_dir": str(chunks_dir)}


def rag_delete_impl(inp: DeleteInput) -> Dict[str, Any]:
    rag = _get_rag(inp.collection)
    ids = inp.ids
    if ids:
        ids = [i if i.startswith("chunk:") else f"chunk:{i}" for i in ids]
    deleted = rag.col.delete(ids=ids, where=inp.where)
    log.info("rag.delete ids=%s where=%s", len(inp.ids or []), bool(inp.where))
    return {"ok": True, "note": "delete issued", "ids": ids, "where": inp.where, "result": deleted}


def rag_health_impl() -> HealthOutput:
    rag = _get_rag()
    try:
        cnt = rag.col.count()
    except Exception:
        cnt = -1
    return HealthOutput(
        ok=True,
        collection=rag.col.name,
        persist_dir=DEFAULT_CHROMA_DIR,
        count=cnt,
        has_embeddings=rag.embedding_fn is not None,
        expand_per_entity=bool(getattr(rag, "expand_per_entity", False)),
    )

# ------------------------- NEW: rag.qa implementation -------------------------
def rag_qa_impl(inp: QAInput) -> QAOutput:
    t0 = time.perf_counter()

    # 1) Retrieve
    rag = _get_rag(inp.collection)
    res = rag.query(
        text=inp.question,
        entity_ids=inp.entity_ids,
        k=inp.k,
        where=inp.where,
        where_document=inp.where_document,
        include=["metadatas", "documents", "distances"],
    )
    citations = _norm_citations(res)

    # 2) Optional KG enrich
    do_enrich = QA_KG_ENRICH if inp.kg_enrich is None else bool(inp.kg_enrich)
    kg_notes: Dict[str, List[str]] = {}
    if do_enrich:
        # collect unique entity IRIs from the citations
        iris: List[str] = []
        for c in citations:
            for e in c.entity_ids:
                if e and e not in iris:
                    iris.append(e)
        if iris:
            kg_notes = _kg_enrich_aliases(iris)

    # 3) LLM synthesis (mock or Ollama)
    model = inp.llm_model or QA_LLM_MODEL
    use_mock = (inp.use_mock_llm is True) or (QA_LLM_MODE == "mock")
    if use_mock:
        answer = _mock_answer(inp.question, citations, kg_notes)
        model_used = "mock-llm"
    else:
        prompt = _build_prompt(inp.question, citations, kg_notes)
        try:
            answer = _ollama_generate(model=model, prompt=prompt)
            model_used = model
        except Exception as e:  # pragma: no cover (network issues)
            log.warning("LLM call failed, falling back to mock: %s", e)
            answer = _mock_answer(inp.question, citations, kg_notes)
            model_used = "mock-llm(fallback)"

    took_ms = int((time.perf_counter() - t0) * 1000)
    log.info("rag.qa qlen=%s k=%s entities=%s -> %s cites in %sms",
             len(inp.question), inp.k, len(inp.entity_ids or []), len(citations), took_ms)

    return QAOutput(
        answer=answer.strip(),
        citations=citations,
        took_ms=took_ms,
        model_used=model_used,
        debug={"retrieval_count": len(citations)}
    )

# -----------------------------------------------------------------------------
# FastMCP tool wrappers (now with descriptions)
# -----------------------------------------------------------------------------
@mcp.tool(
    name="rag.search",
    description="Semantic / filtered retrieval from Chroma (optionally filter by entity_ids and metadata).",
)
def tool_rag_search(inp: SearchInput) -> Dict[str, Any]:
    return rag_search_impl(inp)

@mcp.tool(
    name="rag.reindex",
    description="Rebuild the RAG collection from pipeline outputs (labels + chunks) in a given outputs_dir.",
)
def tool_rag_reindex(inp: ReindexInput) -> Dict[str, Any]:
    return rag_reindex_impl(inp)

@mcp.tool(
    name="rag.embed_and_index",
    description="Add new records: either direct {text, metadata, id?} or by joining labels_dir + chunks_dir.",
)
def tool_rag_embed_and_index(inp: EmbedAndIndexInput) -> Dict[str, Any]:
    return rag_embed_and_index_impl(inp)

@mcp.tool(
    name="rag.delete",
    description="Delete items by ids or metadata filter (where). Normalizes ids to the 'chunk:' prefix.",
)
def tool_rag_delete(inp: DeleteInput) -> Dict[str, Any]:
    return rag_delete_impl(inp)

@mcp.tool(
    name="rag.health",
    description="Basic store diagnostics: collection name, persist dir, count, embedding status, expand-per-entity.",
)
def tool_rag_health() -> HealthOutput:
    return rag_health_impl()

# NEW tool
@mcp.tool(
    name="rag.qa",
    description="Question answering over the RAG index with optional KG enrichment and LLM synthesis.",
)
def tool_rag_qa(inp: QAInput) -> QAOutput:
    return rag_qa_impl(inp)

# -----------------------------------------------------------------------------
# Test-friendly aliases (so tests can call rag_* directly)
# -----------------------------------------------------------------------------
rag_search = rag_search_impl
rag_reindex = rag_reindex_impl
rag_embed_and_index = rag_embed_and_index_impl
rag_delete = rag_delete_impl
rag_health = rag_health_impl
rag_qa = rag_qa_impl

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
