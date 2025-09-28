# src/kg/graphdb_sink.py
import os
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple
import requests

from .namespaces import sparql_prefix_block, iri_entity, iri_property, iri_class

# --- config helpers ---
def _endpoints(base_url: str, repo: str) -> Tuple[str, str]:
    # SPARQL query endpoint (GET/POST) and Update endpoint (POST statements)
    # GraphDB conventions:
    query = f"{base_url.rstrip('/')}/repositories/{repo}"
    update = f"{base_url.rstrip('/')}/repositories/{repo}/statements"
    return query, update

def _auth():
    user = os.getenv("GRAPHDB_USERNAME")
    pwd = os.getenv("GRAPHDB_PASSWORD")
    return (user, pwd) if user and pwd else None

# --- minimal mapping from our labels JSON to triples ---
# Our label records look like:
# {
#   "doc_id": "...",
#   "chunk_id": "...",
#   "section_type": [...],
#   "entities": {"token":[...], "protocol":[...], "component":[...], "organization":[...]},
#   "relations": [...],
#   "keyphrases": [...],
#   "confidence_overall": 0.42
# }
#
# We’ll emit:
# ent:doc/<id>   a ex:Document ; rdfs:label "Title" .
# ent:chunk/<id> a ex:Chunk ; ex:partOf ent:doc/<id> ; ex:hasText "...".
# ent:token/<t>  a ex:Token .
# ent:component/<c> a ex:Component .
# ent:chunk/<id> ex:mentions ent:token/<t> (etc).
# confidence stored as xsd:decimal literal on the labeling edge via a reified blank node is overkill now;
# we’ll attach it to the chunk for now: ex:labelConfidence "0.42"^^xsd:decimal .

def _triples_for_record(meta_index: Dict[str, Dict[str, Any]], rec: Dict[str, Any]) -> str:
    pre = sparql_prefix_block()
    doc_id = rec["doc_id"]
    chunk_id = rec["chunk_id"]
    entities = rec.get("entities", {})
    section_types = rec.get("section_type", [])
    confidence = rec.get("confidence_overall")

    doc_iri = iri_entity("doc", doc_id)
    chunk_iri = iri_entity("chunk", chunk_id)

    # Document triple (with label if we have meta)
    title = meta_index.get(doc_id, {}).get("title")
    pages = meta_index.get(doc_id, {}).get("pages")

    inserts = []

    inserts.append(f"{doc_iri} a {iri_class('Document')} .")
    if title:
        t = title.replace('"', '\\"')
        inserts.append(f'{doc_iri} rdfs:label "{t}" .')
    if pages is not None:
        inserts.append(f"{doc_iri} {iri_property('pageCount')} \"{int(pages)}\"^^xsd:integer .")

    # Chunk triples
    inserts.append(f"{chunk_iri} a {iri_class('Chunk')} ; {iri_property('partOf')} {doc_iri} .")

    # Section types as tags (lightweight)
    for st in section_types:
        st_lit = st.replace('"', '\\"')
        inserts.append(f'{chunk_iri} {iri_property("sectionType")} "{st_lit}" .')

    # Mentioned entities
    def _emit(kind: str, vals: Iterable[str]):
        for v in vals or []:
            v_clean = v.strip()
            if not v_clean:
                continue
            ent_iri = iri_entity(kind, v_clean)
            ent_class = {
                "token": "Token",
                "protocol": "Protocol",
                "component": "Component",
                "organization": "Organization"
            }.get(kind, "Thing")
            inserts.append(f"{ent_iri} a {iri_class(ent_class)} ; rdfs:label \"{v_clean.replace('\"','\\\\\"')}\" .")
            inserts.append(f"{chunk_iri} {iri_property('mentions')} {ent_iri} .")

    _emit("token", entities.get("token"))
    _emit("protocol", entities.get("protocol"))
    _emit("component", entities.get("component"))
    _emit("organization", entities.get("organization"))

    # Confidence (optional)
    if isinstance(confidence, (int, float)):
        inserts.append(f"{chunk_iri} {iri_property('labelConfidence')} \"{confidence}\"^^xsd:decimal .")

    body = "\n".join(inserts)
    if not body.strip():
        return ""
    # Use INSERT with FILTER NOT EXISTS to keep idempotency
    # We guard by trying to avoid re-inserting chunk type triple.
    guard = f"FILTER NOT EXISTS {{ {chunk_iri} a {iri_class('Chunk')} }}"
    return f"""{pre}

INSERT {{
{body}
}}
WHERE {{
  {guard}
}}"""

def _read_meta_index(docs_dir: Path) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for p in docs_dir.glob("*.meta.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            idx[data["doc_id"]] = data
        except Exception:
            pass
    return idx

def push_labels_dir(
    out_dir: Path,
    graphdb_url: str,
    repository: str,
    batch_size: int = 100
):
    """
    Read *.labels.jsonl under outputs/.../labels and push triples into GraphDB.
    """
    _, update_endpoint = _endpoints(graphdb_url, repository)
    labels_dir = out_dir / "labels"
    docs_dir = out_dir / "docs"
    meta_index = _read_meta_index(docs_dir)

    auth = _auth()
    headers = {"Content-Type": "application/sparql-update"}

    # Collect SPARQL updates in mini-batches to reduce HTTP overhead
    batch: List[str] = []
    def _flush():
        if not batch:
            return
        payload = ";\n".join(q for q in batch if q.strip())
        resp = requests.post(update_endpoint, data=payload.encode("utf-8"), headers=headers, auth=auth)
        resp.raise_for_status()
        batch.clear()

    for lab_file in labels_dir.glob("*.labels.jsonl"):
        with lab_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                q = _triples_for_record(meta_index, rec)
                if not q:
                    continue
                batch.append(q)
                if len(batch) >= batch_size:
                    _flush()
    _flush()
