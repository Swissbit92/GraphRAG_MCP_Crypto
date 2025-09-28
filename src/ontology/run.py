# src/ontology/run.py
import argparse
from pathlib import Path

from .proposer import run_proposer_over_run
from .consolidate import consolidate_proposals, write_yaml, write_json


def main():
    parser = argparse.ArgumentParser(prog="ontology-cli", description="Schema proposer + consolidation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("propose", help="Generate per-document YAML proposals")
    p1.add_argument("--run-dir", type=str, default="outputs/run_simple", help="Path to a pipeline run dir (with docs/ and labels/)")
    p1.add_argument("--out-dir", type=str, default=None, help="Where to write per-doc proposals (defaults to <run-dir>/ontology/proposals)")
    p1.add_argument("--parallel", type=int, default=1, help="Number of docs to process in parallel (threads). Try 2-4.")
    p1.add_argument("--max-segments", type=int, default=None, help="Cap number of LLM segments per doc (speed). E.g., 2.")
    p1.add_argument("--sample-first", type=int, default=None, help="Only read first K chars of each doc (pre-batching).")

    p2 = sub.add_parser("consolidate", help="Consolidate proposals into canonical ontology.v0.1.yaml")
    p2.add_argument("--run-dir", type=str, default="outputs/run_simple", help="Run dir (used for grounded terms)")
    p2.add_argument("--proposals-dir", type=str, default=None, help="Where proposals are (defaults to <run-dir>/ontology/proposals)")
    p2.add_argument("--out-yaml", type=str, default=None, help="Output canonical YAML file (defaults to <run-dir>/ontology/canonical/ontology.v0.1.yaml)")
    p2.add_argument("--out-rejections", type=str, default=None, help="Output rejections JSON (defaults to <run-dir>/ontology/canonical/rejections.json)")
    p2.add_argument("--min-docs", type=int, default=3, help="Min distinct docs required to promote a node to core class")

    args = parser.parse_args()

    run_dir = Path(getattr(args, "run_dir"))
    if args.cmd == "propose":
        out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "ontology" / "proposals")
        run_proposer_over_run(
            run_dir=run_dir,
            out_dir=out_dir,
            parallel_docs=args.parallel,
            max_segments_per_doc=args.max_segments,
            sample_first_k_chars=args.sample_first,
        )
        print(f"✅ Proposals written to: {out_dir}")

    elif args.cmd == "consolidate":
        proposals_dir = Path(args.proposals_dir) if args.proposals_dir else (run_dir / "ontology" / "proposals")
        out_yaml = Path(args.out_yaml) if args.out_yaml else (run_dir / "ontology" / "canonical" / "ontology.v0.1.yaml")
        out_rej = Path(args.out_rejections) if args.out_rejections else (run_dir / "ontology" / "canonical" / "rejections.json")

        canonical, rejections = consolidate_proposals(
            proposals_dir=proposals_dir,
            grounded_dir=run_dir,
            min_docs_for_core_class=args.min_docs
        )
        write_yaml(canonical, out_yaml)
        write_json(rejections, out_rej)

        kept_nodes = len(canonical.get("nodes", []))
        kept_rels = len(canonical.get("relationships", []))
        print(f"✅ Canonical ontology written: {out_yaml} (nodes={kept_nodes}, relationships={kept_rels})")
        print(f"🧹 Rejections log: {out_rej}")


if __name__ == "__main__":
    main()
