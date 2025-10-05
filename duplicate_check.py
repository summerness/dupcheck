#!/usr/bin/env python3
"""Entrypoint for the duplicate image checking skeleton.

This script wires the components together and provides a simple CLI.
"""
import argparse
import os
from pathlib import Path

# ...existing code...
def parse_args():
    p = argparse.ArgumentParser(description="Duplicate image check skeleton")
    p.add_argument("--db_dir", required=True, help="Path to image database directory")
    p.add_argument("--input_dir", required=True, help="Path to new images to check")
    p.add_argument("--out_dir", required=True, help="Output reports directory")
    p.add_argument("--topk", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    db_dir = Path(args.db_dir)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports to keep CLI responsive if modules missing
    from duplicate_check import indexer, features, matcher, report

    print(f"Indexing DB: {db_dir}")
    idx = indexer.build_index(db_dir)

    print(f"Processing inputs from: {input_dir}")
    results = []
    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file():
            continue
        print(f"Checking {img_path.name}...")
        feats = features.compute_features(img_path)
        cand = matcher.recall_candidates(feats, idx, topk=args.topk)
        detailed = matcher.rerank_and_verify(img_path, cand)
        results.extend(detailed)

    csv_path = out_dir / "dup_report.csv"
    report.write_csv(results, csv_path)
    print(f"Done. Report: {csv_path}")


if __name__ == "__main__":
    main()
