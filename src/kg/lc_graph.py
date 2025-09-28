# src/kg/lc_graph.py
import os
from langchain_community.graphs import OntotextGraphDBGraph  # Query-only (for QA etc.)

def graph_from_env() -> OntotextGraphDBGraph:
    """
    Returns a LangChain OntotextGraphDBGraph configured from env.
    If you keep your ontology in a named graph, set GRAPHDB_SCHEMA_GRAPH to that IRI.
    Otherwise, set GRAPHDB_SCHEMA_FILE to a local TTL/N-Triples file.
    """
    query_endpoint = f"{os.getenv('GRAPHDB_URL','http://localhost:7200').rstrip('/')}/repositories/{os.getenv('GRAPHDB_REPOSITORY','mcp_kg')}"
    schema_graph = os.getenv("GRAPHDB_SCHEMA_GRAPH")
    schema_file = os.getenv("GRAPHDB_SCHEMA_FILE")
    kwargs = {"query_endpoint": query_endpoint}

    if schema_graph:
        # Recommend storing ontology in its own named graph; see Ontotext docs.
        kwargs["query_ontology"] = f"CONSTRUCT {{ ?s ?p ?o }} FROM <{schema_graph}> WHERE {{ ?s ?p ?o }}"
    elif schema_file:
        kwargs["local_file"] = schema_file

    # If secured, set GRAPHDB_USERNAME/PASSWORD in the environment (class reads them)
    return OntotextGraphDBGraph(**kwargs)
