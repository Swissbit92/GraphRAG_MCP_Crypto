import json
import re
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
from tqdm import tqdm

# Ollama via LangChain
from langchain_ollama import OllamaLLM

from .prompts import get_crypto_ontology_generation_prompt


# ---------- Utilities ----------

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _merge_yaml_schemas(schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {"nodes": [], "relationships": []}
    node_index: Dict[str, Dict[str, Any]] = {}
    rel_index: Dict[str, Dict[str, Any]] = {}

    for sch in schemas:
        for n in sch.get("nodes", []) or []:
            if not isinstance(n, dict) or len(n) != 1:
                continue
            name, spec = next(iter(n.items()))
            key = _normalize(name)
            if key not in node_index:
                node_index[key] = {"name": name, **(spec or {})}
            else:
                dst = node_index[key]
                if not dst.get("description") and spec.get("description"):
                    dst["description"] = spec.get("description")
                # merge attributes (list of {attr: spec})
                a_dst = {k: v for item in (dst.get("attributes") or []) for k, v in item.items()}
                for item in (spec.get("attributes") or []):
                    if not isinstance(item, dict) or len(item) != 1:
                        continue
                    an, av = next(iter(item.items()))
                    if an not in a_dst:
                        a_dst[an] = av
                dst["attributes"] = [{k: v} for k, v in a_dst.items()]

        for r in sch.get("relationships", []) or []:
            if not isinstance(r, dict) or len(r) != 1:
                continue
            name, spec = next(iter(r.items()))
            key = _normalize(name)
            if key not in rel_index:
                rel_index[key] = {"name": name, **(spec or {})}

    merged["nodes"] = [{v["name"]: {k: v[k] for k in v if k != "name"}} for v in node_index.values()]
    merged["relationships"] = [{v["name"]: {k: v[k] for k in v if k != "name"}} for v in rel_index.values()]
    return merged


# ---------- Evidence loader ----------

@dataclass
class DocEvidence:
    doc_meta: Dict[str, Any]
    chunk_texts: List[str]
    grounded_terms: List[str]


def load_run_corpus(run_dir: Path) -> Dict[str, DocEvidence]:
    docs_dir = run_dir / "docs"
    labels_dir = run_dir / "labels"

    by_doc_chunks: Dict[str, List[str]] = {}
    by_doc_meta: Dict[str, Dict[str, Any]] = {}
    by_doc_terms: Dict[str, List[str]] = {}

    # load texts + meta
    for fp in docs_dir.glob("*.jsonl"):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                doc_id = row.get("doc_id")
                if not doc_id:
                    continue
                by_doc_chunks.setdefault(doc_id, []).append(row.get("text", "") or "")
                by_doc_meta.setdefault(doc_id, {
                    "doc_id": doc_id,
                    "title": row.get("title"),
                    "pages": row.get("pages"),
                    "sha1": row.get("sha1"),
                    "filename": row.get("filename"),
                })

    # load grounded terms
    def _flatten_entities(d: Dict[str, Any]) -> List[str]:
        out = []
        for v in (d or {}).values():
            if isinstance(v, list):
                out.extend([str(x) for x in v if isinstance(x, str)])
        return out

    for fp in labels_dir.glob("*.jsonl"):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                doc_id = row.get("doc_id")
                if not doc_id:
                    continue
                terms = _flatten_entities(row.get("entities")) + (row.get("keyphrases") or [])
                by_doc_terms.setdefault(doc_id, []).extend([t for t in terms if isinstance(t, str)])

    out: Dict[str, DocEvidence] = {}
    for doc_id in by_doc_chunks:
        out[doc_id] = DocEvidence(
            doc_meta=by_doc_meta.get(doc_id) or {"doc_id": doc_id},
            chunk_texts=by_doc_chunks[doc_id],
            grounded_terms=sorted(set(by_doc_terms.get(doc_id, []))),
        )
    return out


# ---------- Proposer ----------

class OntologyProposer:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_chars_per_call: int = 8000,
        max_segments_per_doc: Optional[int] = None,
        sample_first_k_chars: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        ollama_options: Optional[dict] = None,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct")
        self.base_url = base_url or os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
        self.max_chars = max_chars_per_call

        # env overrides
        if max_segments_per_doc is None and os.getenv("ONTOLOGY_MAX_SEGS"):
            try:
                max_segments_per_doc = int(os.getenv("ONTOLOGY_MAX_SEGS"))
            except Exception:
                pass
        self.max_segments_per_doc = max_segments_per_doc

        if sample_first_k_chars is None and os.getenv("ONTOLOGY_SAMPLE_CHARS"):
            try:
                sample_first_k_chars = int(os.getenv("ONTOLOGY_SAMPLE_CHARS"))
            except Exception:
                pass
        self.sample_first_k_chars = sample_first_k_chars

        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        opts = {
            "num_predict": 768,       # a bit more room for YAML
            "temperature": 0.2,       # deterministic
            "top_p": 5.0,
            "stop": ["```", "\n\n\n"], # discourage extra prose / fences
        }
        if ollama_options:
            opts.update(ollama_options)

        self.llm = OllamaLLM(model=self.model, base_url=self.base_url, options=opts)
        self.system_prompt = get_crypto_ontology_generation_prompt()

    # ---- batching ----
    def _batched_segments(self, texts: List[str]) -> List[str]:
        if self.sample_first_k_chars:
            joined = "\n\n".join(texts)
            texts = [joined[: self.sample_first_k_chars]]

        segs, buf, size = [], [], 0
        for t in texts:
            t = t or ""
            if size + len(t) > self.max_chars and buf:
                segs.append("\n\n".join(buf))
                buf, size = [t], len(t)
            else:
                buf.append(t)
                size += len(t)
        if buf:
            segs.append("\n\n".join(buf))
        if self.max_segments_per_doc is not None:
            segs = segs[: self.max_segments_per_doc]
        return segs

    # ---- YAML extraction ----
    def _extract_yaml(self, s: str) -> str:
        """
        Make a best-effort extraction:
        1) If there's a fenced ```yaml block, take its contents.
        2) Else, find the largest text slice starting with a line 'nodes:' until EOF.
        3) Else, return original (may still be YAML).
        """
        if not s:
            return ""
        s = s.strip()

        # fenced
        m = re.search(r"```yaml(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # largest 'nodes:' to end
        lines = s.splitlines()
        try:
            start = next(i for i, ln in enumerate(lines) if ln.strip().lower().startswith("nodes:"))
            return "\n".join(lines[start:]).strip()
        except StopIteration:
            pass

        return s

    def _compose_user_prompt(self, doc_meta: Dict[str, Any], grounded_terms: List[str], segment_text: str) -> str:
        grounded_preview = ", ".join(sorted(set(grounded_terms))[:50])
        title = doc_meta.get("title") or doc_meta.get("doc_id") or "whitepaper"
        sha1 = doc_meta.get("sha1") or ""
        pages = doc_meta.get("pages") or ""
        filename = doc_meta.get("filename") or ""
        header = (
            f"Title: {title}\n"
            f"SHA1: {sha1}\n"
            f"Pages: {pages}\n"
            f"Filename: {filename}\n"
            f"Grounded terms: {grounded_preview}\n\n"
            f"TEXT SEGMENT:\n"
        )
        return header + segment_text

    def _cache_key(self, doc_id: str, segment_text: str) -> str:
        h = hashlib.sha1(segment_text.encode("utf-8")).hexdigest()
        return f"{_normalize(doc_id)}__{h}"

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.yaml"
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return None
        return None

    def _write_cache_yaml(self, key: str, data: Dict[str, Any]) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / f"{key}.yaml"
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def _write_cache_raw(self, key: str, text: str) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / f"{key}.raw.txt"
        with p.open("w", encoding="utf-8") as f:
            f.write(text or "")

    def propose_for_document(self, doc_id: str, evidence: DocEvidence) -> Dict[str, Any]:
        schemas: List[Dict[str, Any]] = []
        segments = self._batched_segments(evidence.chunk_texts)

        for seg in segments:
            key = self._cache_key(doc_id, seg)
            cached = self._read_cache(key)
            if cached is not None and (cached.get("nodes") or cached.get("relationships")):
                schemas.append(cached)
                continue

            user_prompt = self._compose_user_prompt(evidence.doc_meta, evidence.grounded_terms, seg)
            # concatenate system + user; OllamaLLM doesn't have separate roles
            raw = self.llm.invoke(f"{self.system_prompt}\n\n{user_prompt}")
            raw = str(raw or "").strip()
            self._write_cache_raw(key, raw)

            yaml_text = self._extract_yaml(raw)
            try:
                data = yaml.safe_load(yaml_text) or {}
            except Exception:
                data = {}

            # ensure shape
            if not isinstance(data, dict):
                data = {}
            data.setdefault("nodes", [])
            data.setdefault("relationships", [])

            # if still empty, at least return the required baseline so consolidation has something
            if not data["nodes"] and not data["relationships"]:
                data = {
                    "nodes": [
                        {"Whitepaper": {"description": "A specific whitepaper document.",
                                        "attributes": [
                                            {"title": {"type": "str", "description": "Document title"}},
                                            {"sha1": {"type": "str", "description": "Document SHA1"}},
                                            {"pages": {"type": "int", "description": "Total pages"}},
                                            {"filename": {"type": "str", "description": "Filename if known"}},
                                        ]}},
                        {"WhitepaperSection": {"description": "A section/segment of a whitepaper.",
                                               "attributes": [
                                                   {"section_id": {"type": "str", "description": "Section identifier or synthetic id"}},
                                                   {"title": {"type": "str", "description": "Optional title, if identifiable"}},
                                                   {"page_start": {"type": "int", "description": "Start page number if known"}},
                                                   {"page_end": {"type": "int", "description": "End page number if known"}},
                                               ]}},
                    ],
                    "relationships": [
                        {"HAS_SECTION": {"description": "Whitepaper contains a section",
                                         "source": "Whitepaper", "target": "WhitepaperSection"}}
                    ],
                }

            schemas.append(data)
            self._write_cache_yaml(key, data)

        if not schemas:
            # absolute fallback
            return {
                "nodes": [
                    {"Whitepaper": {"description": "A specific whitepaper document.",
                                    "attributes": [
                                        {"title": {"type": "str", "description": "Document title"}},
                                        {"sha1": {"type": "str", "description": "Document SHA1"}},
                                        {"pages": {"type": "int", "description": "Total pages"}},
                                        {"filename": {"type": "str", "description": "Filename if known"}},
                                    ]}},
                    {"WhitepaperSection": {"description": "A section/segment of a whitepaper.",
                                           "attributes": [
                                               {"section_id": {"type": "str", "description": "Section identifier or synthetic id"}},
                                               {"title": {"type": "str", "description": "Optional title, if identifiable"}},
                                               {"page_start": {"type": "int", "description": "Start page number if known"}},
                                               {"page_end": {"type": "int", "description": "End page number if known"}},
                                           ]}},
                ],
                "relationships": [
                    {"HAS_SECTION": {"description": "Whitepaper contains a section",
                                     "source": "Whitepaper", "target": "WhitepaperSection"}}
                ],
            }

        # merge per-segment schemas
        proposal = _merge_yaml_schemas(schemas)
        return proposal


def write_yaml(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def run_proposer_over_run(
    run_dir: Path,
    out_dir: Path,
    parallel_docs: int = 1,
    max_segments_per_doc: Optional[int] = None,
    sample_first_k_chars: Optional[int] = None,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    corpus = load_run_corpus(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    proposer = OntologyProposer(
        max_segments_per_doc=max_segments_per_doc,
        sample_first_k_chars=sample_first_k_chars,
        cache_dir=cache_dir,
        ollama_options={
            "num_predict": 768,
            "temperature": 0.2,
            "top_p": 5.0,
            "stop": ["```", "\n\n\n"],
        },
    )

    # serial fallback
    if parallel_docs <= 1:
        for doc_id, evidence in tqdm(corpus.items(), desc="Proposing ontologies"):
            proposal = proposer.propose_for_document(doc_id, evidence)
            write_yaml(proposal, out_dir / f"{_normalize(doc_id)}.yaml")
        return

    # threaded
    def _job(item):
        doc_id, evidence = item
        proposal = proposer.propose_for_document(doc_id, evidence)
        write_yaml(proposal, out_dir / f"{_normalize(doc_id)}.yaml")
        return doc_id

    items = list(corpus.items())
    with ThreadPoolExecutor(max_workers=parallel_docs) as ex:
        futs = [ex.submit(_job, it) for it in items]
        for _ in tqdm(as_completed(futs), total=len(futs), desc=f"Proposing ontologies (threads={parallel_docs})"):
            pass
