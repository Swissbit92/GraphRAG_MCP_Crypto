# src/schema/contracts.py
"""
Lightweight JSON contracts (v0.1) for whitepaper chunk labels.

No external deps (jsonschema/pydantic). We validate+normalize dicts
and fail closed on obviously-wrong shapes while preserving existing
pipeline behavior.

Contract (v0.1) for a *label record* written to *.labels.jsonl:

{
  "schema_version": "0.1",              # added by validator
  "doc_id": "str",                      # required
  "chunk_id": "str",                    # required
  "section_type": ["str", ...],         # array of strings
  "content_role": ["str", ...],         # array of strings
  "entities": {                         # required subkeys (arrays of strings)
      "token": ["str", ...],
      "protocol": ["str", ...],
      "component": ["str", ...],
      "organization": ["str", ...]
  },
  "relations": [                        # optional
      {
        "subject": "str",
        "predicate": "str",
        "object": "str",
        "confidence": float in [0,1],
        "evidence_span": [start:int, end:int]   # optional but if present must be valid
      },
      ...
  ],
  "keyphrases": ["str", ...],
  "confidence_overall": float in [0,1]
}

NOTE: We *optionally* receive the original chunk text to validate evidence_span bounds.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

LABEL_SCHEMA_VERSION = "0.1"

# --------- helpers


def _as_list_of_str(x: Any, max_len: int = 1000) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for item in x:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
        # silently drop non-strings
        if len(out) >= max_len:
            break
    return out


def _as_float_01(x: Any, default: float = 0.6) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _as_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


def _cap_list(items: List[Any], cap: int) -> List[Any]:
    if len(items) <= cap:
        return items
    return items[:cap]


def _validate_span(span: Any, text_len: int) -> Optional[Tuple[int, int]]:
    """Return (start, end) if valid; else None."""
    if not isinstance(span, list) or len(span) != 2:
        return None
    try:
        s = int(span[0])
        e = int(span[1])
    except Exception:
        return None
    if s < 0 or e < 0 or s >= e:
        return None
    if text_len >= 0:  # -1 means unknown
        if e > text_len:
            return None
    return (s, e)


# --------- normalization / validation


def normalize_label_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an LLM/heuristic payload (no doc_id/chunk_id here).
    Ensures required keys exist with correct shapes/types.
    """
    out: Dict[str, Any] = {}

    out["section_type"] = _cap_list(_as_list_of_str(payload.get("section_type")), 20)
    out["content_role"] = _cap_list(_as_list_of_str(payload.get("content_role")), 20)

    ents = payload.get("entities") or {}
    if not isinstance(ents, dict):
        ents = {}
    out["entities"] = {
        "token": _cap_list(_as_list_of_str(ents.get("token")), 100),
        "protocol": _cap_list(_as_list_of_str(ents.get("protocol")), 100),
        "component": _cap_list(_as_list_of_str(ents.get("component")), 200),
        "organization": _cap_list(_as_list_of_str(ents.get("organization")), 100),
    }

    # relations: optional, each with required string fields + confidence + optional evidence_span
    rels_in = payload.get("relations") or []
    out_rels: List[Dict[str, Any]] = []
    if isinstance(rels_in, list):
        for r in rels_in:
            if not isinstance(r, dict):
                continue
            subj = _as_str(r.get("subject")).strip()
            pred = _as_str(r.get("predicate")).strip()
            objv = _as_str(r.get("object")).strip()
            if not (subj and pred and objv):
                continue
            conf = _as_float_01(r.get("confidence"), default=0.6)
            rel_obj: Dict[str, Any] = {
                "subject": subj,
                "predicate": pred,
                "object": objv,
                "confidence": conf,
            }
            span = r.get("evidence_span")
            # DO NOT validate span here (need text length); attach raw span if shape looks like [int,int]
            if isinstance(span, list) and len(span) == 2:
                try:
                    rel_obj["evidence_span"] = [int(span[0]), int(span[1])]
                except Exception:
                    pass
            out_rels.append(rel_obj)
            if len(out_rels) >= 200:
                break
    out["relations"] = out_rels

    out["keyphrases"] = _cap_list(_as_list_of_str(payload.get("keyphrases")), 100)
    out["confidence_overall"] = _as_float_01(payload.get("confidence_overall"), default=0.6)

    return out


def normalize_label_record(
    record: Dict[str, Any],
    *,
    text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize a full *label record* (includes doc_id & chunk_id) and enforce schema v0.1.

    If `text` is provided, we validate/clip evidence_span to be within [0, len(text)].
    """
    out: Dict[str, Any] = {}

    # required id fields
    out["doc_id"] = _as_str(record.get("doc_id")).strip()
    out["chunk_id"] = _as_str(record.get("chunk_id")).strip()

    # normalize payload fields
    payload_norm = normalize_label_payload(record)

    # fix evidence spans if text length is known
    text_len = len(text) if isinstance(text, str) else -1
    fixed_rels: List[Dict[str, Any]] = []
    for r in payload_norm.get("relations", []):
        r2 = dict(r)
        span = r.get("evidence_span", None)
        if span is not None:
            valid = _validate_span(span, text_len)
            if valid is not None:
                r2["evidence_span"] = [valid[0], valid[1]]
            else:
                # drop invalid span, keep relation
                r2.pop("evidence_span", None)
        fixed_rels.append(r2)
    payload_norm["relations"] = fixed_rels

    out.update(payload_norm)
    out["schema_version"] = LABEL_SCHEMA_VERSION

    return out


def validate_label_record(
    record: Dict[str, Any],
    *,
    text: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Strict-ish validation:
      - doc_id, chunk_id non-empty
      - arrays/dicts have correct shapes (post normalization)
      - confidence in [0,1]
      - evidence_span within text bounds if provided

    Returns: (ok: bool, normalized_record: dict, error_message: str|None)
    """
    try:
        norm = normalize_label_record(record, text=text)

        if not norm["doc_id"]:
            return False, norm, "doc_id missing/empty"
        if not norm["chunk_id"]:
            return False, norm, "chunk_id missing/empty"

        # sanity checks on shapes (already normalized, just double-check types)
        if not isinstance(norm["section_type"], list):
            return False, norm, "section_type must be a list"
        if not isinstance(norm["content_role"], list):
            return False, norm, "content_role must be a list"
        ents = norm.get("entities")
        if not (isinstance(ents, dict) and all(k in ents for k in ("token", "protocol", "component", "organization"))):
            return False, norm, "entities missing required keys"
        for key in ("token", "protocol", "component", "organization"):
            if not isinstance(ents.get(key), list):
                return False, norm, f"entities.{key} must be a list"

        # relations
        rels = norm.get("relations", [])
        if not isinstance(rels, list):
            return False, norm, "relations must be a list"
        for r in rels:
            if not isinstance(r, dict):
                return False, norm, "relation item must be an object"
            for k in ("subject", "predicate", "object", "confidence"):
                if k not in r:
                    return False, norm, f"relation missing {k}"
            conf = r.get("confidence")
            if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
                return False, norm, "relation.confidence out of range [0,1]"
            if "evidence_span" in r:
                span = r["evidence_span"]
                # after normalize, span should already be valid
                if not (isinstance(span, list) and len(span) == 2 and isinstance(span[0], int) and isinstance(span[1], int) and span[0] < span[1]):
                    return False, norm, "relation.evidence_span invalid"

        # top-level confidence
        co = norm.get("confidence_overall")
        if not isinstance(co, (int, float)) or not (0.0 <= float(co) <= 1.0):
            return False, norm, "confidence_overall out of range [0,1]"

        # everything looks good
        return True, norm, None

    except Exception as e:
        return False, record, f"exception during validation: {e}"
