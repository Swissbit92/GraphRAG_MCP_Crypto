# src/ontology/consolidate.py
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _index_nodes(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for n in schema.get("nodes", []) or []:
        if not isinstance(n, dict) or len(n) != 1:
            continue
        name, spec = next(iter(n.items()))
        out[name] = spec
    return out


def _index_rels(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for r in schema.get("relationships", []) or []:
        if not isinstance(r, dict) or len(r) != 1:
            continue
        name, spec = next(iter(r.items()))
        out[name] = spec
    return out


def _merge_attributes(dst: Dict[str, Any], src: Dict[str, Any]):
    # attributes is a list of {attr_name: spec}
    dst_map = {name: spec for item in dst.get("attributes", []) or [] for name, spec in item.items()}
    for item in src.get("attributes", []) or []:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        name, spec = next(iter(item.items()))
        if name not in dst_map:
            dst_map[name] = spec
        else:
            # prefer existing; could merge examples later
            pass
    if dst_map:
        dst["attributes"] = [{k: v} for k, v in dst_map.items()]


def _similar(a: str, b: str) -> bool:
    # Simple name similarity for consolidation
    return _normalize(a) == _normalize(b)


def consolidate_proposals(
    proposals_dir: Path,
    grounded_dir: Path,
    min_docs_for_core_class: int = 3,
    allow_project_specific_namespace: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Gate 1 (Evidence):
      - If a node/rel name does not appear in grounded terms for that doc (or is a required baseline), reject it.
    Gate 2 (Corpus-level):
      - Merge same/similar node names across documents.
      - Keep nodes that appear in at least `min_docs_for_core_class` distinct docs (unless they are in allowlist).
    Returns (canonical_yaml_dict, rejections_dict).
    """
    # Load grounded evidence
    grounded_terms_by_doc = _load_grounded_terms(grounded_dir)

    allowlist_nodes = {"Whitepaper", "WhitepaperSection"}
    canonical_nodes: Dict[str, Dict[str, Any]] = {}     # node_name -> spec + docs set
    canonical_nodes_docs: Dict[str, set] = {}
    canonical_rels: Dict[str, Dict[str, Any]] = {}
    canonical_rels_docs: Dict[str, set] = {}
    rejections: Dict[str, List[str]] = {}  # doc_id -> reasons

    for yp in sorted(proposals_dir.glob("*.yaml")):
        doc_id = yp.stem
        schema = _load_yaml(yp)
        grounded = set(grounded_terms_by_doc.get(doc_id, []))
        nodes = _index_nodes(schema)
        rels = _index_rels(schema)

        # Gate 1 — Evidence check
        # Keep required baseline regardless; others need to be grounded (name or fragments in grounded terms)
        def grounded_ok(name: str) -> bool:
            if name in allowlist_nodes:
                return True
            # accept if any token of PascalCase name appears in grounded tokens
            parts = re.findall(r"[A-Z][a-z0-9]+|[A-Z]+(?![a-z])", name) or [name]
            parts_norm = {_normalize(p) for p in parts}
            grounded_norm = {_normalize(t) for t in grounded}
            return len(parts_norm & grounded_norm) > 0

        # Nodes
        for nname, nspec in nodes.items():
            if not grounded_ok(nname):
                rejections.setdefault(doc_id, []).append(f"Node '{nname}' failed Gate1 grounding.")
                continue
            # Gate 2 — consolidate by name
            canon_key = None
            for existing in canonical_nodes.keys():
                if _similar(existing, nname):
                    canon_key = existing
                    break
            if canon_key is None:
                canon_key = nname
                canonical_nodes[canon_key] = {"description": nspec.get("description"), "attributes": nspec.get("attributes", [])}
                canonical_nodes_docs[canon_key] = set()
            else:
                # merge attributes
                _merge_attributes(canonical_nodes[canon_key], nspec)
            canonical_nodes_docs[canon_key].add(doc_id)

        # Relationships
        for rname, rspec in rels.items():
            # relationships are allowed if source/target nodes passed Gate1 (or are baseline)
            src = rspec.get("source")
            tgt = rspec.get("target")
            if not src or not tgt:
                rejections.setdefault(doc_id, []).append(f"Relationship '{rname}' missing source/target.")
                continue
            if not (src in canonical_nodes or src in allowlist_nodes or grounded_ok(src)):
                rejections.setdefault(doc_id, []).append(f"Relationship '{rname}' failed Gate1 (source '{src}' not grounded).")
                continue
            if not (tgt in canonical_nodes or tgt in allowlist_nodes or grounded_ok(tgt)):
                rejections.setdefault(doc_id, []).append(f"Relationship '{rname}' failed Gate1 (target '{tgt}' not grounded).")
                continue

            canon_key = None
            for existing in canonical_rels.keys():
                if _similar(existing, rname):
                    canon_key = existing
                    break
            if canon_key is None:
                canon_key = rname
                canonical_rels[canon_key] = {"description": rspec.get("description"), "source": src, "target": tgt}
                canonical_rels_docs[canon_key] = set()
            canonical_rels_docs[canon_key].add(doc_id)

    # Gate 2 — prune nodes/rels by doc frequency (except baseline)
    final_nodes = []
    for nname, nspec in canonical_nodes.items():
        if nname in allowlist_nodes or len(canonical_nodes_docs.get(nname, set())) >= min_docs_for_core_class:
            final_nodes.append({nname: nspec})

    final_rels = []
    for rname, rspec in canonical_rels.items():
        # keep if both ends kept and frequency ok
        keep = (len(canonical_rels_docs.get(rname, set())) >= max(1, min_docs_for_core_class - 1))  # a bit softer for rels
        if keep and any(n for n in final_nodes if list(n.keys())[0] == rspec.get("source")) and any(n for n in final_nodes if list(n.keys())[0] == rspec.get("target")):
            final_rels.append({rname: rspec})

    canonical_yaml = {"nodes": final_nodes, "relationships": final_rels}
    return canonical_yaml, rejections


def _load_grounded_terms(grounded_dir: Path) -> Dict[str, List[str]]:
    """
    Map doc_id (normalized filename stem used in proposals) -> grounded term list.
    We build it from the existing run outputs at `outputs/run_simple/labels/*.jsonl`.
    Here we just require that `grounded_dir` points to the parent of labels (i.e., run dir).
    """
    labels_dir = grounded_dir / "labels"
    out: Dict[str, List[str]] = {}
    def _norm_doc_id(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

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
                nid = _norm_doc_id(doc_id)
                terms = out.setdefault(nid, [])
                # collect entities + keyphrases
                ents = row.get("entities") or {}
                for _, arr in ents.items():
                    if isinstance(arr, list):
                        terms.extend([str(v) for v in arr if isinstance(v, str)])
                kps = row.get("keyphrases") or []
                if isinstance(kps, list):
                    terms.extend([kp for kp in kps if isinstance(kp, str)])
    # unique
    for k in list(out.keys()):
        out[k] = sorted(set(out[k]))
    return out


def write_yaml(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def write_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
