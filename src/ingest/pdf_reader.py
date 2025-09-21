import hashlib
from dataclasses import dataclass
from pathlib import Path
import fitz  # PyMuPDF

@dataclass
class DocMeta:
    doc_id: str
    title: str
    pages: int
    sha1: str
    filename: str

def sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()

def load_pdf(path: Path):
    data = path.read_bytes()
    doc = fitz.open(stream=data, filetype="pdf")
    sha = sha1_bytes(data)
    title = doc.metadata.get("title") or path.stem
    meta = DocMeta(
        doc_id=f"{path.stem}-{sha[:8]}",
        title=title,
        pages=len(doc),
        sha1=sha,
        filename=path.name,
    )
    pages = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text")
        pages.append(text)
    doc.close()
    return meta, pages
