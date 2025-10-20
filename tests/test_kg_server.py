# tests/test_kg_server.py
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _fake_response_json(payload, status=200, ctype="application/sparql-results+json"):
    class _Resp:
        def __init__(self):
            self.status_code = status
            self._payload = payload
            self.headers = {"Content-Type": ctype}
            self.text = json.dumps(payload)
        def json(self):
            return self._payload
    return _Resp()


@pytest.fixture
def fake_requests(monkeypatch):
    """
    Patch requests.get / requests.post inside src.mcp.server so we don't hit a real GraphDB.
    """
    import src.mcp.kg_server as kg_server

    calls = {"get": [], "post": []}

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        calls["get"].append({"url": url, "params": params})
        # Return a minimal SPARQL JSON result (like your 334 example, but any number is fine)
        data = {
            "head": {"vars": ["n"]},
            "results": {"bindings": [{"n": {"type": "literal", "datatype": "http://www.w3.org/2001/XMLSchema#integer", "value": "123"}}]},
        }
        return _fake_response_json(data)

    def fake_post(url, data=None, headers=None, params=None, auth=None, timeout=None):
        calls["post"].append({"url": url, "params": params, "data_len": 0 if data is None else len(data)})
        # Simulate GraphDB statement/update success
        class _Resp:
            status_code = 204
            text = ""
            headers = {}
        return _Resp()

    # Patch only in the module under test
    monkeypatch.setattr(kg_server, "requests", SimpleNamespace(get=fake_get, post=fake_post))

    return calls


def test_sparql_query_ok(fake_requests):
    import src.mcp.kg_server as kg_server

    out = kg_server.sparql_query(kg_server.SparqlQueryInput(query="SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"))
    assert out.ok is True
    assert "head" in out.result and "results" in out.result
    n = out.result["results"]["bindings"][0]["n"]["value"]
    assert n == "123"  # from our fake response


def test_sparql_update_ok(fake_requests):
    import src.mcp.kg_server as kg_server

    out = kg_server.sparql_update(kg_server.SparqlUpdateInput(update="INSERT DATA { <urn:a> <urn:p> \"1\" }"))
    assert out.ok is True


def test_list_documents_and_get_chunk(tmp_path: Path, monkeypatch):
    import src.mcp.kg_server as kg_server

    # Create a temp docs dir with one chunks.jsonl file
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    f = docs_dir / "sample.chunks.jsonl"

    rec1 = {"text": "Hello world", "metadata": {"doc_id": "docA", "chunk_id": "A:1"}}
    rec2 = {"text": "Bye world", "metadata": {"doc_id": "docB", "chunk_id": "B:1"}}
    f.write_text(json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n", encoding="utf-8")

    # list_documents should see docA and docB
    out_list = kg_server.list_documents(kg_server.ListDocumentsInput(docs_dir=str(docs_dir), limit=10))
    assert out_list.ok is True
    doc_ids = {d["doc_id"] for d in out_list.documents}
    assert {"docA", "docB"} <= doc_ids

    # get_chunk for docA
    out_chunk = kg_server.get_chunk(kg_server.GetChunkInput(docs_dir=str(docs_dir), doc_id="docA", chunk_id="A:1"))
    assert out_chunk.ok is True
    assert out_chunk.record["metadata"]["doc_id"] == "docA"
    assert out_chunk.record["metadata"]["chunk_id"] == "A:1"


def test_validate_labels_without_rdflib(tmp_path: Path, monkeypatch):
    """
    Works even if rdflib is not installed; it should warn but still count files.
    """
    import src.mcp.kg_server as kg_server

    # Force rdflib absence at runtime
    monkeypatch.setattr(kg_server, "rdflib", None, raising=False)

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    ttl = labels_dir / "minimal.ttl"
    ttl.write_text('@prefix ex: <http://example.org/> . ex:a ex:p "x" .', encoding="utf-8")

    out = kg_server.validate_labels(kg_server.ValidateLabelsInput(labels_dir=str(labels_dir)))
    assert out.ok is True  # syntax not validated, but no errors collected
    assert out.files_checked == 1
    assert len(out.warnings) >= 1  # warns that rdflib isn't installed
