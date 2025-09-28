# src/schema/contracts.py
"""
Label record schema/normalization (v0.1).

This module provides:
- validate_label_record(rec: dict, text: Optional[str] = None) -> tuple[bool, dict, str | None]
- validate_label_record_stream(path_or_iter, limit_bad: int = 20) -> dict
- schema_version() -> str

Design goals:
- Keep zero extra dependencies (no jsonschema).
- Normalize gently: coerce missing fields to reasonable defaults.
- Be resilient to LLM output variance (strings vs lists, None, etc.).
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import json
from pathlib import Path

SCHEMA_VERSION = "0.1"

_ALLOWED_ENTITY_KEYS = {"token", "protocol", "component", "organization"}
# Hard caps to avoid pathological records
_MAX_SECTION_TYPES = 8
_MAX_KEYPHRASES = 64
_MAX_ENTITIES_PER_KIND = 128
_MAX_LABEL_LEN = 512  # individual label strings
_MAX_TEXT_SNIPPET = 20000  # if we ever choose to inspect chunk text

def _as_str_list(x: Any) -> List[str]:
    """Normalize value into a list[str] (split single strings; ignore Nones)."""
    if x is None:
        return []
    if isinstance(x, list):
        vals = x
    elif isinstance(x, str):
        # treat as a single label; do not split on commas (labels may contain commas)
        vals = [x]
    else:
        # try to coerce simple scalars
        vals = [str(x)]
    out: List[str] = []
    seen = set()
    for v in vals:
        if not isinstance(v, str):
            v = str(v)
        vv = v.strip()
        if not vv:
            continue
        if len(vv) > _MAX_LABEL_LEN:
            vv = vv[:_MAX_LABEL_LEN]
        if vv not in seen:
            seen.add(vv)
            out.append(vv)
    return out

def _ensure_entities(obj: Any) -> Dict[str, List[str]]:
    """Normalize entities map and clamp counts."""
    result: Dict[str, List[str]] = {}
    if not isinstance(obj, dict):
        obj = {}
    for kind in _ALLOWED_ENTITY_KEYS:
        vals = _as_str_list(obj.get(kind))
        if len(vals) > _MAX_ENTITIES_PER_KIND:
            vals = vals[:_MAX_ENTITIES_PER_KIND]
        result[kind] = vals
    # Drop any unexpected keys silently (forward-compat)
    return result

def _coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return None

def schema_version() -> str:
    return SCHEMA_VERSION

def validate_label_record(
    rec: Dict[str, Any],
    text: Optional[str] = None,  # <- NEW: optional, accepted & currently informational
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Validate and normalize a single label record (schema v0.1).

    Returns (ok, normalized_record, error_message).

    Required minimal fields:
      - doc_id (str), chunk_id (str)
    Optional fields (normalized):
      - section_type: list[str]
      - content_role: list[str]
      - entities: {token|protocol|component|organization: list[str]}
      - relations: list[dict]  (passed through as-is, but must be list)
      - keyphrases: list[str]
      - confidence_overall: float (0..1 recommended but not enforced)

    The `text` parameter is accepted for potential future checks (e.g., length),
    but the schema does not persist it; we use it only for sanity hints.
    """
    if not isinstance(rec, dict):
        return False, {}, "Record must be an object"

    norm: Dict[str, Any] = {}

    # Required ids
    doc_id = rec.get("doc_id")
    chunk_id = rec.get("chunk_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        return False, {}, "Missing or invalid 'doc_id'"
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        return False, {}, "Missing or invalid 'chunk_id'"
    norm["doc_id"] = doc_id.strip()
    norm["chunk_id"] = chunk_id.strip()

    # section_type, content_role
    section_type = _as_str_list(rec.get("section_type"))
    if len(section_type) > _MAX_SECTION_TYPES:
        section_type = section_type[:_MAX_SECTION_TYPES]
    norm["section_type"] = section_type

    content_role = _as_str_list(rec.get("content_role"))
    norm["content_role"] = content_role

    # entities
    norm["entities"] = _ensure_entities(rec.get("entities"))

    # relations (opaque; require list)
    relations = rec.get("relations")
    if relations is None:
        relations = []
    if not isinstance(relations, list):
        return False, {}, "Field 'relations' must be a list if present"
    norm["relations"] = relations

    # keyphrases
    keyphrases = _as_str_list(rec.get("keyphrases"))
    if len(keyphrases) > _MAX_KEYPHRASES:
        keyphrases = keyphrases[:_MAX_KEYPHRASES]
    norm["keyphrases"] = keyphrases

    # confidence_overall
    conf = _coerce_float(rec.get("confidence_overall"))
    if conf is not None:
        # Keep as-is; we won't clamp but we validate it's a number
        norm["confidence_overall"] = conf

    # Optional, soft sanity check: if text is absurdly long, cap what we might inspect
    if text is not None and isinstance(text, str):
        # currently we do not reject by length; this is future-proofing
        _ = text[:_MAX_TEXT_SNIPPET]

    # Attach schema version for downstream consumers (non-strict)
    norm["_schema_version"] = SCHEMA_VERSION

    return True, norm, None

def validate_label_record_stream(
    path_or_iter: Union[str, Path, Iterable[str]],
    limit_bad: int = 20
) -> Dict[str, Any]:
    """
    Validate a JSONL stream of label records.

    - path_or_iter can be a file path or an iterable of JSON lines.
    - Returns a summary dict: {ok, total, bad_count, examples_bad: [...], version}

    We stop collecting examples after `limit_bad`, but continue counting.
    """
    bad_examples: List[Dict[str, Any]] = []
    total = 0
    bad_count = 0

    def _iter_lines() -> Iterable[str]:
        if isinstance(path_or_iter, (str, Path)):
            p = Path(path_or_iter)
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    yield line
        else:
            for line in path_or_iter:
                yield line

    for line in _iter_lines():
        total += 1
        line_s = line.strip()
        if not line_s:
            continue
        try:
            obj = json.loads(line_s)
        except Exception as e:
            bad_count += 1
            if len(bad_examples) < limit_bad:
                bad_examples.append({"line": total, "error": f"invalid json: {e}", "raw": line_s[:500]})
            continue

        ok, norm, err = validate_label_record(obj)
        if not ok:
            bad_count += 1
            if len(bad_examples) < limit_bad:
                bad_examples.append({"line": total, "error": err or "invalid record", "raw": line_s[:500]})

    return {
        "ok": bad_count == 0,
        "total": total,
        "bad_count": bad_count,
        "examples_bad": bad_examples,
        "version": SCHEMA_VERSION,
    }
