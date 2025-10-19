# tests/test_rag_server.py
from __future__ import annotations

import types
import builtins
import importlib
from typing import Any, Dict, List
import pytest


class _FakeCollection:
    def __init__(self, name: str = "whitepapers"):
        self.name = name
        self._store: Dict[str, Dict[str, Any]] = {}

    def count(self) -> int:
        return len(self._store)

    def delete(self, ids=None, where=None):
        if ids:
            for _id in ids:
                self._store.pop(_id, None)
        elif where:
            # very naive where: {'doc_id': {'$eq': 'x'}}
            # delete any row whose metadata matches the equality filter
            if "doc_id" in where and "$eq" in where["doc_id"]:
                doc_id = where["doc_id"]["$eq"]
                to_del = [k for k, v in self._store.items() if v.get("metadata", {}).get("doc_id") == doc_id]
                for k in to_del:
                    self._store.pop(k, None)
        return {"ok": True}


class _FakeRAG:
    def __init__(self, persist_dir, collection, embedding_fn=None, expand_per_entity=None):
        self.persist_dir = persist_dir
        self.embedding_fn = object()  # pretend we have embeddings configured
        self.expand_per_entity = True if expand_per_entity is None else bool(expand_per_entity)
        self.col = _FakeCollection(name=collection)

    def upsert_chunks(self, items, require_embeddings=False):
        if require_embeddings and self.embedding_fn is None:
            raise RuntimeError("Embeddings required")
        # items can be list or generator
        for it in items:
            _id = it.get("id") or f"chunk:auto:{len(self.col._store)+1}"
            if not _id.startswith("chunk:"):
                _id = f"chunk:{_id}"
            self.col._store[_id] = {"text": it["text"], "metadata": it.get("metadata", {})}

    def query(self, text=None, entity_ids=None, k=8, where=None, where_document=None, include=None):
        # return a Chroma-like shape
        docs = []
        metas = []
        for _id, row in list(self.col._store.items())[:k]:
            # filter by entity_ids if provided
            if entity_ids:
                row_eids = row.get("metadata", {}).get("entity_ids") or []
                # any overlap?
                if not any(e in row_eids for e in entity_ids):
                    continue
            docs.append(row["text"])
            metas.append(row["metadata"])
        return {"ids": [["fake:"+str(i) for i in range(len(docs))]], "documents": [docs], "metadatas": [metas]}


@pytest.fixture(autouse=True)
def patch_chromarag(monkeypatch):
    """
    Monkeypatch src.mcp.rag_server to use fake ChromaRAG & fake build_rag_index/iter_pipeline_records.
    """
    import src.mcp.rag_server as rag_server

    # Patch class & helpers
    monkeypatch.setattr(rag_server, "ChromaRAG", _FakeRAG, raising=True)

    def _fake_build(outputs_dir: str, collection: str):
        # Simulate building 2 records
        rag = rag_server._get_rag(collection)
        rag.upsert_chunks([
            {"id": "chunk:reindex:1", "text": "Bitcoin proof-of-work adjusts difficulty.", "metadata": {"doc_id": "btcwp", "entity_ids": ["kg:bitcoin"]}},
            {"id": "chunk:reindex:2", "text": "Lightning Network supports fast payments.", "metadata": {"doc_id": "lightning", "entity_ids": ["kg:lightning"]}},
        ])

    def _fake_iter(labels_dir, chunks_dir):
        # Simulate a generator yielding pipeline-joined items
        yield {"id": "chunk:pipe:1", "text": "Merkle trees summarize transactions.", "metadata": {"doc_id": "btcwp", "entity_ids": ["kg:bitcoin"]}}
        yield {"id": "chunk:pipe:2", "text": "Gossip propagation reduces latency.", "metadata": {"doc_id": "ethwp", "entity_ids": ["kg:ethereum"]}}

    monkeypatch.setattr(rag_server, "build_rag_index", _fake_build, raising=True)
    monkeypatch.setattr(rag_server, "iter_pipeline_records", _fake_iter, raising=True)
    # re-import not necessary; patched in place
    return rag_server


def test_health_ok():
    import src.mcp.rag_server as rag_server
    out = rag_server.rag_health()
    assert out.ok is True
    assert out.collection == "whitepapers"
    assert out.persist_dir == ".chroma"
    assert out.count in (-1, 0)  # fake collection starts empty


def test_embed_and_search_records():
    import src.mcp.rag_server as rag_server
    # Upsert 1 record
    resp = rag_server.rag_embed_and_index(
        rag_server.EmbedAndIndexInput(
            records=[
                rag_server.OneRecord(
                    id="my1",
                    text="Nakamoto consensus provides probabilistic finality.",
                    metadata={"doc_id": "btcwp", "entity_ids": ["kg:bitcoin"]},
                )
            ],
            require_embeddings=True,
        )
    )
    assert resp["ok"] and resp["count"] == 1

    # Search with entity filter
    sres = rag_server.rag_search(
        rag_server.SearchInput(text="finality", entity_ids=["kg:bitcoin"], k=5)
    )
    assert "results" in sres and sres["results"]["documents"]
    assert "Nakamoto consensus" in sres["results"]["documents"][0][0]


def test_reindex_builds_collection():
    import src.mcp.rag_server as rag_server
    resp = rag_server.rag_reindex(rag_server.ReindexInput(outputs_dir="outputs/run_simple"))
    assert resp["ok"] is True

    # After reindex, search should hit at least one of the seeded docs
    sres = rag_server.rag_search(
        rag_server.SearchInput(text="difficulty", entity_ids=["kg:bitcoin"], k=5)
    )
    assert sres["results"]["documents"][0]  # not empty


def test_pipeline_dirs_join_and_delete_by_where():
    import src.mcp.rag_server as rag_server

    # Index via (labels_dir, chunks_dir)
    resp = rag_server.rag_embed_and_index(
        rag_server.EmbedAndIndexInput(
            labels_dir="outputs/run_simple/labels",
            chunks_dir="outputs/run_simple/docs",
            collection="whitepapers",
        )
    )
    assert resp["ok"] and resp["mode"] == "pipeline_dirs"

    # Delete by where (doc_id == 'ethwp')
    dres = rag_server.rag_delete(
        rag_server.DeleteInput(where={"doc_id": {"$eq": "ethwp"}}, collection="whitepapers")
    )
    assert dres["ok"] is True
