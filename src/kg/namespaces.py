# src/kg/namespaces.py
from urllib.parse import quote

# Clean, stable prefixes for the KG
PREFIXES = {
    # Standards
    "rdf":   "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs":  "http://www.w3.org/2000/01/rdf-schema#",
    "xsd":   "http://www.w3.org/2001/XMLSchema#",
    "dct":   "http://purl.org/dc/terms/",
    "foaf":  "http://xmlns.com/foaf/0.1/",
    "skos":  "http://www.w3.org/2004/02/skos/core#",
    "org":   "http://www.w3.org/ns/org#",
    "prov":  "http://www.w3.org/ns/prov#",

    # MCP KG vocabularies
    "mcp":   "https://kg.mcp.ai/core/",
    "crypto":"https://kg.mcp.ai/crypto/",

    # Instance IRIs (your IDs)
    "ids":   "https://kg.mcp.ai/id/",
}

def sparql_prefix_block() -> str:
    return "\n".join([f"PREFIX {p}: <{iri}>" for p, iri in PREFIXES.items()])

def iri_entity(kind: str, value: str) -> str:
    """
    Build a compact, stable entity IRI under the ids: namespace.
    kind: 'token' | 'protocol' | 'component' | 'organization' | 'doc'
    value: free text; will be url-encoded and lowercased
    """
    slug = quote(value.strip().lower().replace(" ", "_"))
    return f"<{PREFIXES['ids']}{kind}/{slug}>"

# Explicit namespace helpers (clearer than 'ex:')
def iri_prop(ns: str, local: str) -> str:
    """Return a CURIE for a property, e.g., iri_prop('mcp','pageCount') -> mcp:pageCount"""
    return f"{ns}:{local}"

def iri_cls(ns: str, local: str) -> str:
    """Return a CURIE for a class, e.g., iri_cls('crypto','Token') -> crypto:Token"""
    return f"{ns}:{local}"

# --- Backwards-compat no-ops (if old code still imports these) ----------
def iri_property(local: str) -> str:
    # Prefer iri_prop('mcp', local) in new code.
    return iri_prop("mcp", local)

def iri_class(local: str) -> str:
    # Prefer iri_cls('mcp', local) or iri_cls('crypto', local) explicitly.
    return iri_cls("mcp", local)
