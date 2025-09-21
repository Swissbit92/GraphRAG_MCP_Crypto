import json
from pathlib import Path
from dataclasses import asdict
from tqdm import tqdm

from src.ingest.pdf_reader import load_pdf
from src.chunking.semantic_splitter import split_pages_to_chunks
from src.classify.llm_chunk_tagger import tag_chunk
from src.cluster.simple_communities import kmeans_text_clusters, co_mention_graph
from src.reports.summary import write_overview, write_json

def process_pdf(pdf_path: Path, out_dir: Path):
    meta, pages = load_pdf(pdf_path)
    doc_dir = out_dir / "docs"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / f"{meta.doc_id}.meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

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
    # save chunks jsonl
    chunks_path = doc_dir / f"{meta.doc_id}.chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # classify (LLM optional)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    lab_path = labels_dir / f"{meta.doc_id}.labels.jsonl"
    with lab_path.open("w", encoding="utf-8") as f:
        for ch in tqdm(chunks, desc=f"Tagging {meta.doc_id}"):
            lab = tag_chunk(ch["text"], title=meta.title)
            lab_rec = {"doc_id": meta.doc_id, "chunk_id": ch["chunk_id"], **lab}
            f.write(json.dumps(lab_rec, ensure_ascii=False) + "\n")

    return meta, chunks, lab_path

def load_all_labels(labels_dir: Path):
    all_labels = []
    for p in labels_dir.glob("*.labels.jsonl"):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                all_labels.append(json.loads(line))
    return all_labels

def main(pdf_dir: str = "data/pdfs", out_dir: str = "outputs/run_simple"):
    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_summaries = []
    all_chunks = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        meta, chunks, _ = process_pdf(pdf, out_dir)
        doc_summaries.append({"doc_id": meta.doc_id, "title": meta.title, "pages": meta.pages, "chunks": len(chunks)})
        all_chunks.extend(chunks)

    # global clustering over all chunks (text-based KMeans)
    kmeans = kmeans_text_clusters(all_chunks, k=6)

    # load labels for co-mention graph
    labels = load_all_labels(out_dir / "labels")
    graph_comm = co_mention_graph(labels)

    # persist communities
    write_json(out_dir / "communities" / "embedding_clusters.json", kmeans)
    write_json(out_dir / "communities" / "graph_communities.json", graph_comm)

    # report
    write_overview(out_dir, doc_summaries, kmeans, graph_comm)
    print(f"\nDone. See report at: {out_dir}/reports/overview.md")

if __name__ == "__main__":
    main()
