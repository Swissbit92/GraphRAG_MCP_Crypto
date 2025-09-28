"""
MCP tools registry.

Tools exposed:
- validate_labels: validate label JSONL(s) against v0.1 schema
    input: {"path": "<glob or file>", "limit_bad": 20}
    output: {"ok": bool, "invalid_count": int, "samples": [ ... ]}

- push_labels: push triples to GraphDB from an outputs/<run>/labels dir
    input: {"out_dir": "outputs/run_simple", "batch_size": 100}
    output: {"ok": bool}

- sparql_query: run a SPARQL SELECT on GraphDB
    input: {"query": "SELECT ..."}
    output: {"ok": bool, "results": {...}}

- sparql_update: run a SPARQL UPDATE/INSERT/DELETE on GraphDB
    input: {"update": "INSERT DATA { ... }"}
    output: {"ok": bool}

- list_documents: list docs in outputs/<run>/docs
    input: {"out_dir": "outputs/run_simple"}
    output: {"ok": bool, "docs": [{"doc_id": ..., "title": ..., "pages": ...}]}

- get_chunk: fetch a chunk text by chunk_id
    input: {"out_dir": "outputs/run_simple", "chunk_id": "<doc:page:idx>"}
    output: {"ok": bool, "chunk": {...}}
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

from src.schema.contracts import validate_label_record_stream
from src.kg.graphdb_sink import push_labels_dir, _endpoints, _auth  # reuse existing code


def _env() -> Tuple[str, str]:
    load_dotenv()
    base = os.getenv("GRAPHDB_URL", "http://localhost:7200")
    repo = os.getenv("GRAPHDB_REPOSITORY", "mcp_kg")
    return base, repo


def _jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


# -------------------- Tool impls --------------------

def tool_validate_labels(inp: Dict[str, Any]) -> Dict[str, Any]:
    pat = inp.get("path")
    if not pat:
        return {"ok": False, "error": "path is required"}
    limit_bad = int(inp.get("limit_bad", 20))
    paths = [Path(p) for p in glob.glob(pat)]
    invalid = 0
    samples: List[Dict[str, Any]] = []
    for p in paths:
        for res in validate_label_record_stream(p):
            if not res["ok"]:
                invalid += 1
                if len(samples) < limit_bad:
                    samples.append({"file": str(p), **res})
    return {"ok": invalid == 0, "invalid_count": invalid, "samples": samples}


def tool_push_labels(inp: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = inp.get("out_dir") or "outputs/run_simple"
    batch_size = int(inp.get("batch_size", 100))

    base, repo = _env()
    try:
        push_labels_dir(Path(out_dir), base, repo, batch_size=batch_size)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_sparql_query(inp: Dict[str, Any]) -> Dict[str, Any]:
    query = inp.get("query")
    if not query:
        return {"ok": False, "error": "query is required"}

    base, repo = _env()
    query_ep, _ = _endpoints(base, repo)
    auth = _auth()
    headers = {"Accept": "application/sparql-results+json"}
    try:
        resp = requests.post(query_ep, data={"query": query}, headers=headers, auth=auth)
        resp.raise_for_status()
        return {"ok": True, "results": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_sparql_update(inp: Dict[str, Any]) -> Dict[str, Any]:
    update = inp.get("update")
    if not update:
        return {"ok": False, "error": "update is required"}
    base, repo = _env()
    _, update_ep = _endpoints(base, repo)
    auth = _auth()
    headers = {"Content-Type": "application/sparql-update"}
    try:
        resp = requests.post(update_ep, data=update.encode("utf-8"), headers=headers, auth=auth)
        resp.raise_for_status()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_list_documents(inp: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(inp.get("out_dir") or "outputs/run_simple")
    docs_dir = out_dir / "docs"
    docs: List[Dict[str, Any]] = []
    for meta in docs_dir.glob("*.meta.json"):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            docs.append({"doc_id": m.get("doc_id"), "title": m.get("title"), "pages": m.get("pages")})
        except Exception:
            pass
    return {"ok": True, "docs": docs}


def tool_get_chunk(inp: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(inp.get("out_dir") or "outputs/run_simple")
    chunk_id = inp.get("chunk_id")
    if not chunk_id:
        return {"ok": False, "error": "chunk_id is required (e.g., '<doc_id>:<page>:<idx>')"}

    docs_dir = out_dir / "docs"
    # scan all chunk files; for scale you'd index by doc_id, but this is fine for now.
    for chunks_file in docs_dir.glob("*.chunks.jsonl"):
        for rec in _jsonl_iter(chunks_file):
            if rec.get("chunk_id") == chunk_id:
                return {"ok": True, "chunk": rec}
    return {"ok": False, "error": f"chunk_id not found: {chunk_id}"}


# -------------------- Registry & public API --------------------

_REGISTRY = {
    "validate_labels": {
        "description": "Validate label JSONL(s) against v0.1 schema",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or glob to *.labels.jsonl"},
                "limit_bad": {"type": "integer", "default": 20}
            },
            "required": ["path"]
        },
        "fn": tool_validate_labels,
    },
    "push_labels": {
        "description": "Push triples to GraphDB from outputs/<run>/labels",
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string", "default": "outputs/run_simple"},
                "batch_size": {"type": "integer", "default": 100}
            }
        },
        "fn": tool_push_labels,
    },
    "sparql_query": {
        "description": "Run a SPARQL SELECT on GraphDB",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        },
        "fn": tool_sparql_query,
    },
    "sparql_update": {
        "description": "Run a SPARQL UPDATE/INSERT/DELETE on GraphDB",
        "input_schema": {
            "type": "object",
            "properties": {
                "update": {"type": "string"}
            },
            "required": ["update"]
        },
        "fn": tool_sparql_update,
    },
    "list_documents": {
        "description": "List documents discovered in outputs/<run>/docs",
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string", "default": "outputs/run_simple"}
            }
        },
        "fn": tool_list_documents,
    },
    "get_chunk": {
        "description": "Fetch a chunk by chunk_id from outputs/<run>/docs",
        "input_schema": {
            "type": "object",
            "properties": {
                "out_dir": {"type": "string", "default": "outputs/run_simple"},
                "chunk_id": {"type": "string"}
            },
            "required": ["chunk_id"]
        },
        "fn": tool_get_chunk,
    },
}


def list_tools() -> Dict[str, Any]:
    """Return a lightweight spec for host discovery or local inspection."""
    return {
        "ok": True,
        "tools": [
            {
                "name": name,
                "description": spec["description"],
                "input_schema": spec["input_schema"],
            }
            for name, spec in _REGISTRY.items()
        ],
    }


def run_tool(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    spec = _REGISTRY.get(name)
    if not spec:
        return {"ok": False, "error": f"unknown tool: {name}"}
    try:
        return spec["fn"](payload or {})
    except Exception as e:
        return {"ok": False, "error": str(e)}
