# tests/test_rag_qa.py
import types
import pytest

from src.mcp import rag_server
from src.mcp.rag_server import QAInput, CitationItem

class _FakeRAG:
    def __init__(self, results):
        self.results = results
        self.col = types.SimpleNamespace(name="whitepapers")  # for health if needed
        self.embedding_fn = None
        self.expand_per_entity = True

    def query(self, **kwargs):
        # Return a Chroma-like shape
        return self.results

def _make_results():
    # simulate Chroma .query() output (one query => nested lists)
    docs = [[
        "Bitcoin is a peer-to-peer electronic cash system enabling payments without trusted intermediaries.",
        "Proof-of-work secures the network by requiring computational effort to create new blocks."
    ]]
    metas = [[
        {"doc_id": "bitcoin-whitepaper", "chunk_id": "bitcoin-whitepaper:001:000",
         "entity_id": "https://kg.mcp.ai/id/token/bitcoin",
         "entity_ids_json": '["https://kg.mcp.ai/id/token/bitcoin"]'},
        {"doc_id": "bitcoin-whitepaper", "chunk_id": "bitcoin-whitepaper:002:000",
         "entity_id": None, "entity_ids_json": "[]"},
    ]]
    dists = [[0.12, 0.23]]
    return {"documents": docs, "metadatas": metas, "distances": dists}

def test_rag_qa_mock_llm(monkeypatch):
    fake = _FakeRAG(_make_results())

    # Patch the internal getter so QA uses our fake store
    monkeypatch.setattr(rag_server, "_get_rag", lambda collection=None: fake)

    # Force KG enrichment off so we don't hit SPARQL
    inp = QAInput(
        question="What is Bitcoin?",
        k=2,
        kg_enrich=False,
        use_mock_llm=True
    )
    out = rag_server.rag_qa(inp)

    assert out.answer  # non-empty
    assert out.model_used.startswith("mock-llm")
    assert len(out.citations) == 2
    # check stable extraction
    c0 = out.citations[0]
    assert isinstance(c0, CitationItem)
    assert c0.doc_id == "bitcoin-whitepaper"
    assert "peer-to-peer electronic cash" in c0.text

def test_rag_qa_entity_filter(monkeypatch):
    # same fake results; ensure path works with entity filter present
    fake = _FakeRAG(_make_results())
    monkeypatch.setattr(rag_server, "_get_rag", lambda collection=None: fake)

    inp = QAInput(
        question="How is the network secured?",
        entity_ids=["https://kg.mcp.ai/id/token/bitcoin"],
        k=2,
        kg_enrich=False,
        use_mock_llm=True
    )
    out = rag_server.rag_qa(inp)

    assert out.citations  # retrieved
    # The first citation should be the Bitcoin one with entity id
    assert "bitcoin-whitepaper" in (out.citations[0].doc_id or "")
    assert out.took_ms >= 0
