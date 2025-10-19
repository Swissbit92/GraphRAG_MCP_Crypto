import os
import json
import argparse
import textwrap
from typing import List, Dict, Any, Optional
import requests

from pathlib import Path
from .chroma_store import ChromaRAG
from ..kg.namespaces import iri_entity

def ollama_generate(prompt: str, model: Optional[str] = None, base: Optional[str] = None) -> str:
    base = (base or os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")).rstrip("/")
    model = model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    url = f"{base}/api/generate"
    resp = requests.post(url, json={"model": model, "prompt": prompt, "stream": False})
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()

SYSTEM_INSTRUCTIONS = """You are a careful research assistant.
Answer the user's question using ONLY the provided context.
If the context does not contain the answer, say: "I don't know from the provided context."
Cite your sources at the end as [doc_id:page] for each supporting snippet.
Keep the answer concise and precise.
"""

def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        doc_id = ctx.get("doc_id")
        page = ctx.get("page")
        text = (ctx.get("text") or "").replace("\n", " ").strip()
        text = text[:1200]
        blocks.append(f"[{i}] ({doc_id}:{page}) {text}")

    context_str = "\n".join(blocks) if blocks else "(no context)"
    user_q = question.strip()

    prompt = textwrap.dedent(f"""
    {SYSTEM_INSTRUCTIONS}

    Context:
    {context_str}

    Question:
    {user_q}

    Answer (with [doc_id:page] citations):
    """).strip()

    return prompt

def to_entity_iris(entities_csv: Optional[str]) -> Optional[List[str]]:
    if not entities_csv:
        return None
    out = []
    for name in [x.strip() for x in entities_csv.split(",") if x.strip()]:
        iri = iri_entity("token", name)
        out.append(iri[1:-1])
    return out

def collect_contexts(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = res.get("documents", [[]])[0] if res.get("documents") else []
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []

    ids = []
    if "ids" in res and res["ids"]:
        ids = res["ids"][0]
    else:
        for m, d in zip(metas, docs):
            doc_id = (m or {}).get("doc_id")
            chunk_id = (m or {}).get("chunk_id")
            page = (m or {}).get("page")
            ids.append(f"{doc_id}:{chunk_id}:{page}")

    contexts: List[Dict[str, Any]] = []
    for d, m, i in zip(docs, metas, ids):
        contexts.append({
            "text": d,
            "doc_id": (m or {}).get("doc_id"),
            "page": (m or {}).get("page"),
            "chunk_id": (m or {}).get("chunk_id"),
            "entity_id": (m or {}).get("entity_id"),
            "id": i,
        })
    return contexts

def make_citation_list(contexts: List[Dict[str, Any]]) -> str:
    seen = []
    for c in contexts:
        key = f"{c.get('doc_id')}:{c.get('page')}"
        if key not in seen:
            seen.append(key)
    return ", ".join(f"[{s}]" for s in seen if s and ":" in s)

def main():
    ap = argparse.ArgumentParser(description="Quick LLM Q&A over Chroma RAG store.")
    ap.add_argument("--question", "-q", required=True, help="User question.")
    ap.add_argument("--k", type=int, default=6, help="Top-k chunks to retrieve.")
    ap.add_argument("--entities", "-e", default=None, help="Comma-separated entity names to filter (e.g., 'bitcoin,chainlink').")
    ap.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "whitepapers"))
    ap.add_argument("--persist_dir", default=os.getenv("CHROMA_DIR", ".chroma"))
    args = ap.parse_args()

    entity_ids = to_entity_iris(args.entities)

    rag = ChromaRAG(persist_dir=args.persist_dir, collection=args.collection)
    res = rag.query(text=args.question, entity_ids=entity_ids, k=args.k)

    contexts = collect_contexts(res)
    prompt = build_prompt(args.question, contexts)

    answer = ollama_generate(prompt)

    print("\n" + "=" * 80)
    print("QUESTION:")
    print(args.question)
    print("=" * 80)
    print("ANSWER:")
    print(answer)
    print("\nCitations:", make_citation_list(contexts))
    print("=" * 80)

    debug = os.getenv("QA_DEBUG", "false").lower() in ("1","true","yes")
    if debug:
        print("\n[Debug] Top contexts:")
        for i, c in enumerate(contexts, 1):
            print(f"--- [{i}] {c.get('doc_id')} page {c.get('page')} id={c.get('id')}")
            print((c.get("text") or "")[:280].replace("\n"," ") + " ...")

if __name__ == "__main__":
    main()
