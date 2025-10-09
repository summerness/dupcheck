#!/usr/bin/env python3
"""Simple CLI for running duplicate detection pipeline.

Usage example:
  python dupcheck_cli.py --db_dir ./images_db --input_dir ./images_new --out_dir ./reports
"""
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db_dir", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--index_db", default="./index.db", help="Path to sqlite index DB")
    p.add_argument("--rebuild_index", action="store_true", help="Rebuild sqlite index from db_dir")
    p.add_argument("--phash_thresh", type=int, default=10)
    p.add_argument("--orb_inliers_thresh", type=int, default=25)
    p.add_argument("--ncc_thresh", type=float, default=0.92)
    p.add_argument("--vector_score_thresh", type=float, default=0.0, help="Minimum FAISS similarity to accept a vector candidate")
    return p.parse_args()


def main():
    args = parse_args()
    db_dir = Path(args.db_dir)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from duplicate_check import indexer, features, matcher, report
    # import modules from package
    # 从包中导入模块

    # use sqlite-backed index if available
    # 优先使用 SQLite 索引以支持持久化和增量更新
    idx = None
    db_path = Path(args.index_db)
    if db_path.exists() and not args.rebuild_index:
        print(f"Loading index from {db_path}...")
        try:
            idx = indexer.load_index_from_db(db_path)
        except Exception:
            idx = None

    if idx is None:
        if args.rebuild_index or not db_path.exists():
            print("Building sqlite index...")
            indexer.build_index_db(db_dir, db_path)
        else:
            print("Building in-memory index...")
        idx = indexer.load_index_from_db(db_path) if db_path.exists() else indexer.build_index(db_dir)

    results = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        print(f"Checking {p.name}...")
        feats = features.compute_features(p)
        cands = matcher.recall_candidates(
            feats,
            idx,
            topk=args.topk,
            phash_thresh=args.phash_thresh,
            vector_score_thresh=args.vector_score_thresh,
        )
        rows = matcher.rerank_and_verify(p, cands, idx, orb_inliers_thresh=args.orb_inliers_thresh, ncc_thresh=args.ncc_thresh)
        # generate evidence images for rows
        for r in rows:
            if r.get("matched_image"):
                dbp = Path(idx["by_id"][r["matched_image"]]["path"])
                evid = out_dir / f"{p.stem}__VS__{dbp.stem}.jpg"
                report.make_evidence_image(p, dbp, evid, draw_matches=True, matches=r.get("match_pairs"))
                r["evidence_img_path"] = str(evid)
        results.extend(rows)

    csvp = out_dir / "dup_report.csv"
    report.write_csv(results, csvp)
    print(f"Done. Report: {csvp}")


if __name__ == "__main__":
    main()
