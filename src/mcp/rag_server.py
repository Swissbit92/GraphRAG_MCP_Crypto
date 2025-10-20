# src/mcp/rag_server.py
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator
from fastmcp import FastMCP

from src.rag.chroma_store import ChromaRAG, iter_pipeline_records, build_rag_index

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

# -----------------------------------------------------------------------------
# Pydantic schemas for tools
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

# -----------------------------------------------------------------------------
# Test-friendly aliases (so tests can call rag_* directly)
# -----------------------------------------------------------------------------
rag_search = rag_search_impl
rag_reindex = rag_reindex_impl
rag_embed_and_index = rag_embed_and_index_impl
rag_delete = rag_delete_impl
rag_health = rag_health_impl

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
