# src/ontology/validators.py
import re
from typing import Dict, Any, List, Tuple


PASCAL = re.compile(r"^[A-Z][A-Za-z0-9]*(_[0-9]+)?$")
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
SCREAM = re.compile(r"^[A-Z0-9_]+$")


def validate_yaml_schema(schema: Dict[str, Any]) -> List[str]:
    """
    Lightweight structural checks.
    """
    errors: List[str] = []
    if not isinstance(schema, dict):
        return ["Schema is not a dict"]

    nodes = schema.get("nodes", [])
    rels = schema.get("relationships", [])
    if not isinstance(nodes, list) or not isinstance(rels, list):
        errors.append("nodes or relationships not lists")

    # Node structure
    for node in nodes:
        if not isinstance(node, dict) or len(node) != 1:
            errors.append(f"Bad node entry: {node}")
            continue
        (name, spec), = node.items()
        if not PASCAL.match(name):
            errors.append(f"Node '{name}' not PascalCase")
        if not isinstance(spec, dict):
            errors.append(f"Node '{name}' spec not a dict")
            continue
        attrs = spec.get("attributes", [])
        if attrs and not isinstance(attrs, list):
            errors.append(f"Node '{name}' attributes not a list")
        if attrs:
            for a in attrs:
                if not isinstance(a, dict) or len(a) != 1:
                    errors.append(f"Node '{name}' attribute bad: {a}")
                    continue
                (aname, aspec), = a.items()
                if not SNAKE.match(aname):
                    errors.append(f"Attr '{name}.{aname}' not snake_case")
                if not isinstance(aspec, dict) or "type" not in aspec:
                    errors.append(f"Attr '{name}.{aname}' missing type")

    # Relationship structure
    for rel in rels:
        if not isinstance(rel, dict) or len(rel) != 1:
            errors.append(f"Bad relationship entry: {rel}")
            continue
        (rname, rspec), = rel.items()
        if not SCREAM.match(rname):
            errors.append(f"Rel '{rname}' not SCREAMING_SNAKE_CASE")
        if not isinstance(rspec, dict):
            errors.append(f"Rel '{rname}' spec not a dict")
            continue
        if "source" not in rspec or "target" not in rspec:
            errors.append(f"Rel '{rname}' missing source/target")
        else:
            if not PASCAL.match(rspec["source"]):
                errors.append(f"Rel '{rname}' source not PascalCase")
            if not PASCAL.match(rspec["target"]):
                errors.append(f"Rel '{rname}' target not PascalCase")

    return errors


def gate1_evidence_filter(
    schema: Dict[str, Any],
    doc_text: str,
    grounded_vocab: Dict[str, List[str]],
    min_hits_node: int = 1,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Gate 1: remove nodes/relationships that lack textual evidence in doc context or grounded entities.
    - Node names must appear in the doc text (case-insensitive) OR be in grounded vocab sets.
    """
    reasons: List[str] = []
    text = doc_text.lower()

    grounded_terms = set()
    for k, vals in grounded_vocab.items():
        grounded_terms.update([v.lower() for v in vals])

    keep_nodes = []
    kept_names = set()

    for node in schema.get("nodes", []):
        (name, spec), = node.items()
        lname = name.lower()
        evidence = (lname in text) or (lname in grounded_terms)
        if evidence:
            keep_nodes.append(node)
            kept_names.add(name)
        else:
            reasons.append(f"Drop Node '{name}' — no evidence")

    keep_rels = []
    for rel in schema.get("relationships", []):
        (rname, rspec), = rel.items()
        s = rspec.get("source")
        t = rspec.get("target")
        if s in kept_names and t in kept_names:
            keep_rels.append(rel)
        else:
            reasons.append(f"Drop Rel '{rname}' — disconnected or node dropped")

    filtered = {"nodes": keep_nodes, "relationships": keep_rels}
    return filtered, reasons
