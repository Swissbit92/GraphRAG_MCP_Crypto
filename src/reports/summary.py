from pathlib import Path
import json

def write_overview(out_dir: Path, doc_summaries, kmeans, graph_comm):
    report = out_dir / "reports" / "overview.md"
    report.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Whitepaper Explore — Overview\n"]
    lines.append("## Documents\n")
    for d in doc_summaries:
        lines.append(f"- **{d['title']}** — id `{d['doc_id']}` — pages: {d['pages']} — chunks: {d['chunks']}")
    lines.append("\n## KMeans text clusters\n")
    for c in kmeans.get("clusters", []):
        lines.append(f"- **{c['cluster_id']}**: {len(c['members'])} chunks | terms: {', '.join(c['label_terms'][:8])}")
    lines.append("\n## Co-mention graph communities\n")
    for g in graph_comm.get("communities", []):
        tops = ", ".join(g.get("top_entities", [])[:6])
        lines.append(f"- **{g['community_id']}**: {len(g['entities'])} entities | top: {tops}")

    report.write_text("\n".join(lines), encoding="utf-8")

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
