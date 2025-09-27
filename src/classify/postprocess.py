# src/classify/postprocess.py
import re
from typing import Dict, Any, List, Tuple

_WORD = re.compile(r"\w+", re.UNICODE)

def _find_spans(text: str, needle: str, case_insensitive: bool = True) -> List[Tuple[int,int]]:
    if not needle or not text:
        return []
    flags = re.IGNORECASE if case_insensitive else 0
    # prefer whole-token-ish matches first, then fallback to substring
    patt_word = re.compile(rf"\b{re.escape(needle)}\b", flags)
    spans = [(m.start(), m.end()) for m in patt_word.finditer(text)]
    if spans:
        return spans
    patt_sub = re.compile(re.escape(needle), flags)
    return [(m.start(), m.end()) for m in patt_sub.finditer(text)]

def _norm_entity(e: str) -> str:
    return " ".join(_WORD.findall(e)).strip()

def keep_if_present(text: str, items: List[str], top_k: int = 10) -> List[str]:
    """Return only items that literally appear in text (case-insensitive)."""
    seen = set()
    kept = []
    for raw in items:
        e = (raw or "").strip()
        if not e:
            continue
        if e.lower() in seen:
            continue
        if _find_spans(text, e):
            kept.append(e)
            seen.add(e.lower())
    # keep the list compact
    return kept[:top_k]

def filter_entities_relations(obj: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Drop entities/relations that aren't grounded in the text."""
    out = dict(obj)
    ents = out.get("entities", {}) or {}

    tokens = keep_if_present(text, ents.get("token", []), top_k=20)
    protocols = keep_if_present(text, ents.get("protocol", []), top_k=20)
    components = keep_if_present(text, ents.get("component", []), top_k=40)
    orgs = keep_if_present(text, ents.get("organization", []), top_k=20)

    out["entities"] = {
        "token": tokens,
        "protocol": protocols,
        "component": components,
        "organization": orgs,
    }

    # Filter relations: subject & object must be visible in the text (as literals)
    rels = out.get("relations", []) or []
    grounded_rels = []
    for r in rels:
        subj = (r.get("subject") or "").strip()
        objv = (r.get("object") or "").strip()
        if subj and objv and (_find_spans(text, subj) and _find_spans(text, objv)):
            # ensure evidence span is actually inside text; if not, regenerate from object
            span = r.get("evidence_span") or []
            if not (isinstance(span, list) and len(span) == 2 and 0 <= span[0] < span[1] <= len(text)):
                spans = _find_spans(text, objv)
                if spans:
                    span = list(spans[0])
                else:
                    # fallback to subject
                    sps = _find_spans(text, subj)
                    span = list(sps[0]) if sps else [0, min(20, len(text))]
            grounded = dict(r)
            grounded["evidence_span"] = span
            grounded_rels.append(grounded)

    out["relations"] = grounded_rels

    # Keyphrases: keep only those seen
    out["keyphrases"] = keep_if_present(text, out.get("keyphrases", []), top_k=20)

    return out
