# src/mcp/kg_server.py
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field, model_validator
from fastmcp import FastMCP

# Optional: rdflib for local TTL validation when available
try:
    import rdflib  # type: ignore
except Exception:  # pragma: no cover
    rdflib = None

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("KG_MCP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] kg_mcp: %(message)s",
)
log = logging.getLogger("kg_mcp")

# -----------------------------------------------------------------------------
# Env / defaults
# -----------------------------------------------------------------------------
GRAPHDB_URL = os.getenv("GRAPHDB_URL", "http://localhost:7200").rstrip("/")
GRAPHDB_REPOSITORY = os.getenv("GRAPHDB_REPOSITORY", "mcp_kg")
GRAPHDB_USERNAME = os.getenv("GRAPHDB_USERNAME")  # optional
GRAPHDB_PASSWORD = os.getenv("GRAPHDB_PASSWORD")  # optional

KG_ONTOLOGY_PATH = os.getenv("KG_ONTOLOGY_PATH")
KG_SHAPES_PATH = os.getenv("KG_SHAPES_PATH")

DEFAULT_LABELS_DIR = os.getenv("KG_LABELS_DIR", "outputs/run_simple/labels")
DEFAULT_DOCS_DIR = os.getenv("KG_DOCS_DIR", "outputs/run_simple/docs")

# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------
def _auth() -> Optional[requests.auth.AuthBase]:
    if GRAPHDB_USERNAME and GRAPHDB_PASSWORD:
        return requests.auth.HTTPBasicAuth(GRAPHDB_USERNAME, GRAPHDB_PASSWORD)
    return None

def _repo_query_url() -> str:
    return f"{GRAPHDB_URL}/repositories/{GRAPHDB_REPOSITORY}"

def _repo_statements_url() -> str:
    return f"{GRAPHDB_URL}/repositories/{GRAPHDB_REPOSITORY}/statements"

def _repo_size_url() -> str:
    return f"{GRAPHDB_URL}/repositories/{GRAPHDB_REPOSITORY}/size"

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class ValidateLabelsInput(BaseModel):
    labels_dir: str = Field(default=DEFAULT_LABELS_DIR, description="Directory containing .ttl files to validate.")
    ontology_path: Optional[str] = Field(default=KG_ONTOLOGY_PATH, description="Path to core ontology .ttl (optional).")
    shapes_path: Optional[str] = Field(default=KG_SHAPES_PATH, description="Path to SHACL shapes .ttl (optional).")

class ValidateLabelsOutput(BaseModel):
    ok: bool
    files_checked: int
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class PushLabelsInput(BaseModel):
    labels_dir: str = Field(default=DEFAULT_LABELS_DIR, description="Directory with *.ttl to POST to GraphDB.")
    context: Optional[str] = Field(default=None, description="Named graph IRI to insert into (optional).")
    chunk_size: int = Field(default=1_000_000, ge=1000, description="Unused now; reserved for future streaming logic.")

class PushLabelsOutput(BaseModel):
    ok: bool
    inserted_files: int
    contexts: List[str] = Field(default_factory=list)

class SparqlQueryInput(BaseModel):
    query: str = Field(description="SPARQL SELECT/CONSTRUCT/DESCRIBE/ASK query string.")
    accept: str = Field(default="application/sparql-results+json", description="Accept header for result format.")

class SparqlQueryOutput(BaseModel):
    ok: bool
    result: Any

class SparqlUpdateInput(BaseModel):
    update: str = Field(description="SPARQL UPDATE string.")

class SparqlUpdateOutput(BaseModel):
    ok: bool

class ListDocumentsInput(BaseModel):
    docs_dir: str = Field(default=DEFAULT_DOCS_DIR, description="Directory with *.chunks.jsonl files.")
    limit: int = Field(default=100, ge=1, le=10_000)

class ListDocumentsOutput(BaseModel):
    ok: bool
    documents: List[Dict[str, Any]]

class GetChunkInput(BaseModel):
    docs_dir: str = Field(default=DEFAULT_DOCS_DIR, description="Directory with *.chunks.jsonl files.")
    doc_id: str = Field(description="Document identifier to search for.")
    chunk_id: Optional[str] = Field(default=None, description="Chunk id to match exactly (optional).")

class GetChunkOutput(BaseModel):
    ok: bool
    record: Optional[Dict[str, Any]] = None

class KgHealthOutput(BaseModel):
    ok: bool
    base_url: str
    repository: str
    triple_count: int
    auth_configured: bool

# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
mcp = FastMCP("kg")

# -----------------------------------------------------------------------------
# Plain implementations (callable from tests)
# -----------------------------------------------------------------------------
def validate_labels_impl(inp: ValidateLabelsInput) -> ValidateLabelsOutput:
    errors: List[str] = []
    warnings: List[str] = []
    files_checked = 0

    def _try_parse(path: Path):
        nonlocal files_checked
        files_checked += 1
        if rdflib is None:
            warnings.append(f"rdflib not installed; syntax not validated for {path.name}")
            return
        g = rdflib.Graph()
        try:
            g.parse(path.as_posix(), format="turtle")
        except Exception as e:  # pragma: no cover
            errors.append(f"{path.name}: {e}")

    # Optional ontology / shapes
    for opt in [inp.ontology_path, inp.shapes_path]:
        if opt:
            p = Path(opt)
            if p.exists():
                _try_parse(p)
            else:
                warnings.append(f"File not found: {p}")

    for p in sorted(Path(inp.labels_dir).glob("**/*.ttl")):
        _try_parse(p)

    ok = len(errors) == 0
    log.info("validate_labels dir=%s files=%s ok=%s", inp.labels_dir, files_checked, ok)
    return ValidateLabelsOutput(ok=ok, files_checked=files_checked, errors=errors, warnings=warnings)


def push_labels_impl(inp: PushLabelsInput) -> PushLabelsOutput:
    ttl_paths = sorted(Path(inp.labels_dir).glob("**/*.ttl"))
    inserted = 0
    contexts: List[str] = []
    headers = {"Content-Type": "text/turtle;charset=utf-8"}
    params: Dict[str, str] = {}
    if inp.context:
        params["context"] = f"<{inp.context}>"
        contexts.append(inp.context)

    for p in ttl_paths:
        data = p.read_text(encoding="utf-8")
        resp = requests.post(
            _repo_statements_url(),
            data=data.encode("utf-8"),
            headers=headers,
            params=params,
            auth=_auth(),
            timeout=60,
        )
        if resp.status_code not in (200, 204):
            log.error("push_labels failed for %s: %s %s", p.name, resp.status_code, getattr(resp, "text", "")[:300])
            raise RuntimeError(f"Failed to insert {p.name}: {resp.status_code} {getattr(resp, 'text', '')[:200]}")
        inserted += 1

    log.info("push_labels dir=%s files=%s ctx=%s", inp.labels_dir, inserted, contexts or ["<default>"])
    return PushLabelsOutput(ok=True, inserted_files=inserted, contexts=contexts)


def sparql_query_impl(inp: SparqlQueryInput) -> SparqlQueryOutput:
    headers = {"Accept": inp.accept}
    resp = requests.get(
        _repo_query_url(),
        params={"query": inp.query},
        headers=headers,
        auth=_auth(),
        timeout=60,
    )
    if resp.status_code != 200:
        log.error("sparql_query error %s: %s", resp.status_code, getattr(resp, "text", "")[:300])
        raise RuntimeError(f"SPARQL query error: {resp.status_code} {getattr(resp, 'text', '')[:200]}")

    # JSON if possible, else text
    result: Any
    ctype = resp.headers.get("Content-Type", "")
    if "application/sparql-results+json" in ctype or "application/json" in ctype:
        result = resp.json()
    else:
        result = resp.text

    log.info("sparql_query ok len=%s", (len(result) if hasattr(result, "__len__") else "n/a"))
    return SparqlQueryOutput(ok=True, result=result)


def sparql_update_impl(inp: SparqlUpdateInput) -> SparqlUpdateOutput:
    headers = {"Content-Type": "application/sparql-update"}
    resp = requests.post(
        _repo_statements_url(),
        data=inp.update.encode("utf-8"),
        headers=headers,
        auth=_auth(),
        timeout=60,
    )
    if resp.status_code not in (200, 204):
        log.error("sparql_update error %s: %s", resp.status_code, getattr(resp, "text", "")[:300])
        raise RuntimeError(f"SPARQL update error: {resp.status_code} {getattr(resp, 'text', '')[:200]}")

    log.info("sparql_update ok")
    return SparqlUpdateOutput(ok=True)


def list_documents_impl(inp: ListDocumentsInput) -> ListDocumentsOutput:
    docs_dir = Path(inp.docs_dir)
    if not docs_dir.exists():
        return ListDocumentsOutput(ok=True, documents=[])

    counts: Dict[str, int] = {}
    for path in sorted(docs_dir.glob("**/*.chunks.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > inp.limit:
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                doc_id = rec.get("metadata", {}).get("doc_id") or rec.get("doc_id")
                if not doc_id:
                    continue
                counts[doc_id] = counts.get(doc_id, 0) + 1

    documents = [{"doc_id": k, "chunks": v} for k, v in sorted(counts.items(), key=lambda kv: kv[0])]
    log.info("list_documents dir=%s n=%s", inp.docs_dir, len(documents))
    return ListDocumentsOutput(ok=True, documents=documents[: inp.limit])


def get_chunk_impl(inp: GetChunkInput) -> GetChunkOutput:
    docs_dir = Path(inp.docs_dir)
    if not docs_dir.exists():
        return GetChunkOutput(ok=False, record=None)

    for path in sorted(docs_dir.glob("**/*.chunks.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                md = rec.get("metadata", {})
                if (md.get("doc_id") or rec.get("doc_id")) != inp.doc_id:
                    continue
                if inp.chunk_id and (md.get("chunk_id") or rec.get("chunk_id")) != inp.chunk_id:
                    continue
                log.info("get_chunk doc_id=%s ok from %s", inp.doc_id, path.name)
                return GetChunkOutput(ok=True, record=rec)

    log.info("get_chunk doc_id=%s not found", inp.doc_id)
    return GetChunkOutput(ok=False, record=None)


def kg_health_impl() -> KgHealthOutput:
    """
    Probe GraphDB repository and return a small health summary.
    - Tries GET /repositories/{repo}/size (preferred); falls back to a COUNT(*) SPARQL if needed.
    """
    triple_count = -1

    # Preferred: native size endpoint
    try:
        resp = requests.get(_repo_size_url(), auth=_auth(), timeout=20)
        if resp.status_code == 200:
            try:
                triple_count = int(resp.text.strip())
            except Exception:
                triple_count = -1
    except Exception:
        triple_count = -1

    # Fallback via SPARQL COUNT
    if triple_count < 0:
        try:
            q = "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"
            sresp = requests.get(
                _repo_query_url(),
                params={"query": q},
                headers={"Accept": "application/sparql-results+json"},
                auth=_auth(),
                timeout=30,
            )
            if sresp.status_code == 200:
                data = sresp.json()
                val = data["results"]["bindings"][0]["n"]["value"]
                triple_count = int(val)
        except Exception:
            triple_count = -1

    ok = triple_count >= 0
    return KgHealthOutput(
        ok=ok,
        base_url=GRAPHDB_URL,
        repository=GRAPHDB_REPOSITORY,
        triple_count=triple_count,
        auth_configured=bool(GRAPHDB_USERNAME and GRAPHDB_PASSWORD),
    )

# -----------------------------------------------------------------------------
# FastMCP wrappers (with descriptions)
# -----------------------------------------------------------------------------
@mcp.tool(
    name="validate_labels",
    description="Validate Turtle files in a directory (syntax-level). If rdflib is installed, parses each file.",
)
def tool_validate_labels(inp: ValidateLabelsInput) -> ValidateLabelsOutput:
    return validate_labels_impl(inp)

@mcp.tool(
    name="push_labels",
    description="POST .ttl files into GraphDB (optionally into a named graph via 'context').",
)
def tool_push_labels(inp: PushLabelsInput) -> PushLabelsOutput:
    return push_labels_impl(inp)

@mcp.tool(
    name="sparql_query",
    description="Run a SPARQL query against GraphDB; returns JSON if available, otherwise raw text.",
)
def tool_sparql_query(inp: SparqlQueryInput) -> SparqlQueryOutput:
    return sparql_query_impl(inp)

@mcp.tool(
    name="sparql_update",
    description="Run a SPARQL UPDATE against GraphDB (e.g., INSERT DATA, DELETE WHERE, CLEAR GRAPH).",
)
def tool_sparql_update(inp: SparqlUpdateInput) -> SparqlUpdateOutput:
    return sparql_update_impl(inp)

@mcp.tool(
    name="list_documents",
    description="Enumerate docs from pipeline JSONL outputs, aggregating doc_id → chunk count.",
)
def tool_list_documents(inp: ListDocumentsInput) -> ListDocumentsOutput:
    return list_documents_impl(inp)

@mcp.tool(
    name="get_chunk",
    description="Fetch one chunk record by doc_id (and optional chunk_id) from pipeline JSONL outputs.",
)
def tool_get_chunk(inp: GetChunkInput) -> GetChunkOutput:
    return get_chunk_impl(inp)

@mcp.tool(
    name="kg.health",
    description="Basic GraphDB diagnostics for the configured repository (triple count, auth flag).",
)
def tool_kg_health() -> KgHealthOutput:
    return kg_health_impl()

# -----------------------------------------------------------------------------
# Test-friendly aliases (so tests can call directly)
# -----------------------------------------------------------------------------
validate_labels = validate_labels_impl
push_labels = push_labels_impl
sparql_query = sparql_query_impl
sparql_update = sparql_update_impl
list_documents = list_documents_impl
get_chunk = get_chunk_impl
kg_health = kg_health_impl

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
