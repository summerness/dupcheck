"""Threshold tuning helper.

Usage:
    python tools/tune_thresholds.py --labels labels.csv --db_dir ./images_db --input_dir ./images_new --out_dir ./reports

labels.csv should contain columns: new_image, matched_image, label (unique/partial_duplicate/exact_patch)

This script sweeps phash_thresh, orb_inliers_thresh, ncc_thresh and reports simple match rate vs ground truth.

阈值调优脚本。

用法：
    python tools/tune_thresholds.py --labels labels.csv --db_dir ./images_db --input_dir ./images_new --out_dir ./reports

labels.csv 应包含列：new_image, matched_image, label（unique/partial_duplicate/exact_patch）

本脚本对 phash_thresh、orb_inliers_thresh、ncc_thresh 做网格搜索，并报告与标注的 TP/FP/FN 统计。
"""
import sys
import argparse
import csv
from pathlib import Path

# Ensure repo root is on sys.path so `duplicate_check` package is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from duplicate_check import indexer, features, matcher


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--db_dir", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def load_labels(path):
    rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows[row['new_image']] = row
    return rows


def main():
    args = parse_args()
    labels = load_labels(args.labels)
    db_dir = Path(args.db_dir)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    idx = indexer.build_index(db_dir)

    # simple sweep
    phash_range = [6,8,10,12]
    orb_range = [10,25,50]
    ncc_range = [0.85,0.9,0.92,0.95]

    results = []
    for ph in phash_range:
        for orb_th in orb_range:
            for ncc in ncc_range:
                tp=0; fp=0; fn=0
                for p in input_dir.iterdir():
                    if not p.is_file():
                        continue
                    feats = features.compute_features(p)
                    cands = matcher.recall_candidates(feats, idx, phash_thresh=ph)
                    rows = matcher.rerank_and_verify(p, cands, idx, orb_inliers_thresh=orb_th, ncc_thresh=ncc)
                    predicted = rows[0]['matched_image'] if rows else None
                    gt = labels.get(p.name, {}).get('matched_image')
                    if gt and predicted == gt:
                        tp+=1
                    elif gt and predicted != gt:
                        fn+=1
                    elif not gt and predicted:
                        fp+=1
                results.append((ph,orb_th,ncc,tp,fp,fn))
    # write out
    outp = out_dir / 'tune_results.csv'
    with open(outp, 'w', newline='', encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['phash','orb','ncc','tp','fp','fn'])
        for r in results:
            w.writerow(r)
    print('Done. Results:', outp)

if __name__=='__main__':
    main()
