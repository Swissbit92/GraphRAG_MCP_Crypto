# src/pipeline.py
import os
import json
from pathlib import Path
from dataclasses import asdict
from dotenv import load_dotenv
from tqdm import tqdm

# ✅ package-relative imports
from .ingest.pdf_reader import load_pdf
from .chunking.semantic_splitter import split_pages_to_chunks
from .classify.llm_chunk_tagger import tag_chunk
from .cluster.simple_communities import kmeans_text_clusters, co_mention_graph
from .reports.summary import write_overview, write_json
from .schema.contracts import validate_label_record  # ← NEW


def _is_boilerplate(ch_text: str) -> bool:
    """
    Heuristics to drop low-signal pages/chunks (licenses, references, footers).
    This reduces noise in clustering without affecting labeling.
    """
    t = ch_text.lower()
    # creative commons / license pages
    if "creative commons" in t or "attribution" in t or "license" in t:
        return True
    # references sections: many bracketed citations or the word 'references'
    if "references" in t and t.strip().startswith("references"):
        return True
    # footers with almost no letters
    alpha_ratio = sum(c.isalpha() for c in t) / max(1, len(t))
    if alpha_ratio < 0.25 and len(t) > 400:
        return True
    return False


def process_pdf(pdf_path: Path, out_dir: Path):
    """
    Read a single PDF → chunk → LLM tag → persist chunks + labels (validated).
    Returns (meta, chunks, labels_path).
    """
    meta, pages = load_pdf(pdf_path)

    # docs/
    doc_dir = out_dir / "docs"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / f"{meta.doc_id}.meta.json").write_text(
        json.dumps(asdict(meta), indent=2),
        encoding="utf-8"
    )

    # chunk
    raw_chunks = split_pages_to_chunks(pages)
    chunks = []
    for ch in raw_chunks:
        chunks.append({
            "doc_id": meta.doc_id,
            "chunk_id": f"{meta.doc_id}:{ch['page']:03d}:{ch['chunk_idx']:03d}",
            "page": ch["page"],
            "char_start": ch["char_start"],
            "char_end": ch["char_end"],
            "text": ch["text"]
        })

    # persist chunks jsonl
    chunks_path = doc_dir / f"{meta.doc_id}.chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # classify (LLM via LangChain + Ollama; internal fallback exists)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    lab_path = labels_dir / f"{meta.doc_id}.labels.jsonl"

    dropped = 0
    with lab_path.open("w", encoding="utf-8") as f:
        for ch in tqdm(chunks, desc=f"Tagging {meta.doc_id}"):
            lab = tag_chunk(ch["text"], title=meta.title)
            # stitch record
            lab_rec = {"doc_id": meta.doc_id, "chunk_id": ch["chunk_id"], **lab}
            # ✅ validate & normalize against schema v0.1 (uses text for span bounds)
            ok, norm, err = validate_label_record(lab_rec, text=ch["text"])
            if not ok:
                # Fail closed for structurally broken rows; log to sidecar file
                dropped += 1
                (labels_dir / f"{meta.doc_id}.labels.errors.log").open("a", encoding="utf-8").write(
                    json.dumps({"chunk_id": ch["chunk_id"], "error": err, "raw": lab_rec}, ensure_ascii=False) + "\n"
                )
                # continue without writing the bad row
                continue
            f.write(json.dumps(norm, ensure_ascii=False) + "\n")

    if dropped:
        print(f"  - {dropped} label row(s) dropped for {meta.doc_id} due to schema v0.1 validation")

    return meta, chunks, lab_path


def load_all_labels(labels_dir: Path):
    """Read back all label JSONL files to feed the co-mention graph."""
    all_labels = []
    for p in labels_dir.glob("*.labels.jsonl"):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                all_labels.append(json.loads(line))
    return all_labels


def main(pdf_dir: str = "data/pdfs", out_dir: str = "outputs/run_simple"):
    # Load .env so OLLAMA_MODEL / OLLAMA_BASE are available for LangChain Ollama
    load_dotenv()

    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the LLM label cache (used by llm_chunk_tagger) is colocated with this run
    os.environ.setdefault(
        "WP_EXPLORE_CACHE_DIR",
        str(out_dir / ".cache" / "labels")
    )

    # Process PDFs → chunks + labels
    doc_summaries = []
    all_chunks = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        meta, chunks, _ = process_pdf(pdf, out_dir)
        doc_summaries.append({
            "doc_id": meta.doc_id,
            "title": meta.title,
            "pages": meta.pages,
            "chunks": len(chunks)
        })
        all_chunks.extend(chunks)

    # Filter boilerplate before clustering (labels remain unfiltered)
    filtered_chunks = [c for c in all_chunks if not _is_boilerplate(c["text"])]
    kmeans = kmeans_text_clusters(filtered_chunks, k=6)

    # Build co-mention graph communities from labels
    labels = load_all_labels(out_dir / "labels")
    graph_comm = co_mention_graph(labels)

    # Persist communities
    write_json(out_dir / "communities" / "embedding_clusters.json", kmeans)
    write_json(out_dir / "communities" / "graph_communities.json", graph_comm)

    # Write the human overview report
    write_overview(out_dir, doc_summaries, kmeans, graph_comm)
    if os.getenv("GRAPHDB_PUSH", "").lower() in ("1", "true", "yes"):
        from .kg.graphdb_sink import push_labels_dir
        push_labels_dir(
            out_dir=out_dir,
            graphdb_url=os.getenv("GRAPHDB_URL", "http://localhost:7200"),
            repository=os.getenv("GRAPHDB_REPOSITORY", "mcp_kg"),
            batch_size=int(os.getenv("GRAPHDB_BATCH_SIZE", "100"))
        )
    print(f"\nDone. See report at: {out_dir}/reports/overview.md")


if __name__ == "__main__":
    main()
