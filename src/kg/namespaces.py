# src/kg/namespaces.py
from urllib.parse import quote

# Base prefixes (adjust freely as your ontology matures)
PREFIXES = {
    "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl":  "http://www.w3.org/2002/07/owl#",
    "xsd":  "http://www.w3.org/2001/XMLSchema#",
    "ex":   "https://mcp.example.org/vocab/",        # classes/properties
    "ent":  "https://mcp.example.org/entity/",       # instances
}

def sparql_prefix_block() -> str:
    return "\n".join([f"PREFIX {p}: <{iri}>" for p, iri in PREFIXES.items()])

def iri_entity(kind: str, value: str) -> str:
    """
    Build a compact, stable entity IRI:
    kind: 'token' | 'protocol' | 'component' | 'organization' | 'chunk' | 'doc'
    value: free text; will be url-encoded and lowercased
    """
    slug = quote(value.strip().lower().replace(" ", "_"))
    return f"<{PREFIXES['ent']}{kind}/{slug}>"

def iri_property(local: str) -> str:
    return f"ex:{local}"

def iri_class(local: str) -> str:
    return f"ex:{local}"
