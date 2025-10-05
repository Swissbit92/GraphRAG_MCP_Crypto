# src/kg/graphdb_sink.py
import os
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional
import requests

from .namespaces import sparql_prefix_block, iri_entity, iri_prop, iri_cls

__all__ = ["GraphDB", "push_labels_dir"]

# Feature flag: entity-only KG (default true)
ENTITY_ONLY = os.getenv("KG_ENTITY_ONLY", "true").strip().lower() in ("1", "true", "yes")

# --- config helpers ---
def _endpoints(base_url: str, repo: str) -> Tuple[str, str]:
    """Return (SPARQL query endpoint, SPARQL update endpoint)."""
    query = f"{base_url.rstrip('/')}/repositories/{repo}"
    update = f"{base_url.rstrip('/')}/repositories/{repo}/statements"
    return query, update

def _auth() -> Optional[Tuple[str, str]]:
    user = os.getenv("GRAPHDB_USERNAME")
    pwd = os.getenv("GRAPHDB_PASSWORD")
    return (user, pwd) if user and pwd else None


# === Minimal client class expected by src.mcp.tools ==========================
class GraphDB:
    """
    Very small GraphDB HTTP client:
      - query(): SPARQL SELECT/ASK/DESCRIBE/CONSTRUCT
      - update(): SPARQL UPDATE (INSERT/DELETE)
      - health(): quick connectivity check
      - push_labels_dir(): convenience wrapper using this client config
    """
    def __init__(
        self,
        base_url: Optional[str] = None,
        repository: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30,
        verify: bool = True,
    ):
        self.base_url = (base_url or os.getenv("GRAPHDB_URL") or "").strip()
        self.repository = (repository or os.getenv("GRAPHDB_REPOSITORY") or "").strip()
        if not self.base_url or not self.repository:
            raise ValueError("GraphDB base URL and repository are required (GRAPHDB_URL, GRAPHDB_REPOSITORY).")

        self.query_endpoint, self.update_endpoint = _endpoints(self.base_url, self.repository)
        self.auth = (username, password) if (username and password) else _auth()
        self.timeout = timeout
        self.verify = verify  # allow overriding TLS verification if ever needed

    # --- basic ops ---
    def query(self, sparql: str, accept: str = "application/sparql-results+json") -> requests.Response:
        """
        Execute a SPARQL query. Returns the raw Response so callers can .json() or .text.
        """
        headers = {"Accept": accept}
        resp = requests.post(
            self.query_endpoint,
            data={"query": sparql},
            headers=headers,
            auth=self.auth,
            timeout=self.timeout,
            verify=self.verify,
        )
        resp.raise_for_status()
        return resp

    def update(self, sparql_update: str) -> requests.Response:
        """
        Execute a SPARQL UPDATE.
        """
        headers = {"Content-Type": "application/sparql-update"}
        resp = requests.post(
            self.update_endpoint,
            data=sparql_update.encode("utf-8"),
            headers=headers,
            auth=self.auth,
            timeout=self.timeout,
            verify=self.verify,
        )
        resp.raise_for_status()
        return resp

    def health(self) -> Dict[str, Any]:
        """
        Minimal health check: run a trivial ASK and return basic info.
        """
        try:
            r = self.query("ASK {}")
            ok = r.json().get("boolean", False)
            return {
                "ok": bool(ok),
                "repository": self.repository,
                "endpoint": self.query_endpoint,
            }
        except Exception as e:
            return {
                "ok": False,
                "repository": self.repository,
                "endpoint": self.query_endpoint,
                "error": str(e),
            }

    # --- convenience wrapper mirroring the module-level function -------------
    def push_labels_dir(self, out_dir: Path, batch_size: int = 100):
        """
        Call the module-level push_labels_dir using this client's configuration.
        """
        return push_labels_dir(out_dir=out_dir,
                               graphdb_url=self.base_url,
                               repository=self.repository,
                               batch_size=batch_size)


# --- mapping from our labels JSON to triples --------------------------------
# Label record shape (unchanged):
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
# ENTITY_ONLY = true:
#   ids:doc/<id>    a mcp:Document ; dct:title "Title" ; mcp:pageCount N .
#   ids:token/<t>   a crypto:Token ; rdfs:label "..." .
#   ids:protocol/<p> a crypto:Protocol ; rdfs:label "..." .
#   ids:component/<c> a crypto:Component ; rdfs:label "..." .
#   ids:organization/<o> a org:Organization ; rdfs:label "..." .
#
# ENTITY_ONLY = false (back-compat):
#   (legacy) also writes Chunk, mentions, sectionType, labelConfidence.

def _triples_for_record(meta_index: Dict[str, Dict[str, Any]], rec: Dict[str, Any]) -> str:
    pre = sparql_prefix_block()
    doc_id = rec.get("doc_id")
    entities = rec.get("entities", {}) or {}

    inserts: List[str] = []

    # Optional Document (nice to have in KG for UI & joins)
    if doc_id:
        doc_iri = iri_entity("doc", doc_id)
        title = meta_index.get(doc_id, {}).get("title")
        pages = meta_index.get(doc_id, {}).get("pages")

        inserts.append(f"{doc_iri} a {iri_cls('mcp','Document')} .")
        if title:
            t = str(title).replace('"', '\\"')
            inserts.append(f'{doc_iri} dct:title "{t}" .')
        if pages is not None:
            try:
                inserts.append(f'{doc_iri} {iri_prop("mcp","pageCount")} "{int(pages)}"^^xsd:integer .')
            except Exception:
                pass

    # Entities (Token/Protocol/Component/Organization)
    def _emit(kind: str, vals: Iterable[str]):
        ns, cls = {
            "token":        ("crypto", "Token"),
            "protocol":     ("crypto", "Protocol"),
            "component":    ("crypto", "Component"),
            "organization": ("org",    "Organization"),
        }.get(kind, ("mcp", "Entity"))
        for v in vals or []:
            v_clean = (v or "").strip()
            if not v_clean:
                continue
            ent_iri = iri_entity(kind, v_clean)
            inserts.append(f"{ent_iri} a {iri_cls(ns, cls)} ; rdfs:label \"{v_clean.replace('\"','\\\\\"')}\" .")

    _emit("token",        entities.get("token"))
    _emit("protocol",     entities.get("protocol"))
    _emit("component",    entities.get("component"))
    _emit("organization", entities.get("organization"))

    # If legacy mode is desired, add chunks/mentions here
    if not ENTITY_ONLY:
        # Legacy path (kept for backward compatibility)
        chunk_id = rec.get("chunk_id")
        section_types = rec.get("section_type", []) or []
        confidence = rec.get("confidence_overall")

        if doc_id and chunk_id:
            chunk_iri = iri_entity("chunk", chunk_id)
            # chunk & partOf
            inserts.append(
                f"{chunk_iri} a {iri_cls('mcp','Chunk')} ; "
                f"{iri_prop('mcp','partOf')} {iri_entity('doc', doc_id)} ."
            )
            # section types
            for st in section_types:
                st_lit = str(st).replace('"', '\\"')
                inserts.append(f'{chunk_iri} {iri_prop("mcp","sectionType")} "{st_lit}" .')
            # mentions (emit links to any entities just created)
            for kind, vals in (entities or {}).items():
                for v in vals or []:
                    v_clean = (v or "").strip()
                    if not v_clean:
                        continue
                    ent_iri = iri_entity(kind, v_clean)
                    inserts.append(f"{chunk_iri} {iri_prop('mcp','mentions')} {ent_iri} .")
            # confidence
            if isinstance(confidence, (int, float)):
                inserts.append(f'{chunk_iri} {iri_prop("mcp","labelConfidence")} "{confidence}"^^xsd:decimal .')

    body = "\n".join(inserts)
    if not body.strip():
        return ""

    # RDF stores are set-based; INSERT DATA is idempotent in practice.
    return f"""{pre}

INSERT DATA {{
{body}
}}"""

def _read_meta_index(docs_dir: Path) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for p in docs_dir.glob("*.meta.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if "doc_id" in data:
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
    In ENTITY_ONLY mode, only entities (and optional documents) are written.
    """
    _, update_endpoint = _endpoints(graphdb_url, repository)
    labels_dir = out_dir / "labels"
    docs_dir = out_dir / "docs"
    meta_index = _read_meta_index(docs_dir)

    auth = _auth()
    headers = {"Content-Type": "application/sparql-update"}

    batch: List[str] = []

    def _flush():
        if not batch:
            return
        payload = ";\n".join(q for q in batch if q.strip())
        resp = requests.post(
            update_endpoint,
            data=payload.encode("utf-8"),
            headers=headers,
            auth=auth,
            timeout=60,
        )
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
