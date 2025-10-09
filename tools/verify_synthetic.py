"""Quick evaluator for the synthetic DupCheck dataset.

Usage example:
    python tools/verify_synthetic.py \
      --db_dir data/synth_db \
      --input_dir data/synth_new \
      --labels data/synth_labels.csv

The script runs the duplicate detection pipeline against the synthetic
dataset and reports how many annotated duplicates / uniques are detected
correctly along with any mismatches it finds.
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplicate_check import features, indexer, matcher


def check_dependencies() -> List[str]:
    missing: List[str] = []
    if not getattr(features, "PIL_AVAILABLE", False) or getattr(features, "imagehash", None) is None:
        missing.append("Pillow + imagehash (needed for perceptual hash and tile hashing)")
    if getattr(features, "cv2", None) is None:
        missing.append("opencv-python (needed for ORB matching and NCC verification)")
    return missing


def load_labels(path: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["new_image"]] = row
    return rows


def evaluate(
    db_dir: Path,
    input_dir: Path,
    labels_path: Path,
    *,
    topk: int,
    phash_thresh: int,
    orb_inliers_thresh: int,
    ncc_thresh: float,
    vector_score_thresh: float,
    roi_margin_ratio: float,
    max_roi_matches: int,
) -> Dict[str, object]:
    labels = load_labels(labels_path)
    idx = indexer.build_index(db_dir)

    stats = {
        "duplicate_total": 0,
        "duplicate_hits": 0,
        "unique_total": 0,
        "unique_hits": 0,
        "mismatches": [],
    }

    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file():
            continue
        feats = features.compute_features(img_path)
        cands = matcher.recall_candidates(
            feats,
            idx,
            topk=topk,
            phash_thresh=phash_thresh,
            vector_score_thresh=vector_score_thresh,
        )
        rows = matcher.rerank_and_verify(
            img_path,
            cands,
            idx,
            orb_inliers_thresh=orb_inliers_thresh,
            ncc_thresh=ncc_thresh,
            roi_margin_ratio=roi_margin_ratio,
            max_roi_matches=max_roi_matches,
        )

        meta = labels.get(img_path.name, {"matched_image": "", "label": "unique"})
        gt_match = meta.get("matched_image") or ""
        gt_label = meta.get("label", "unique")

        predicted_label = rows[0]["final_label"] if rows else "unique"
        predicted_match = rows[0]["matched_image"] if rows else ""
        if predicted_label == "unique":
            predicted_match = ""

        if gt_match:
            stats["duplicate_total"] += 1
            if predicted_match == gt_match:
                stats["duplicate_hits"] += 1
            else:
                stats["mismatches"].append(
                    {
                        "image": img_path.name,
                        "expected_match": gt_match,
                        "expected_label": gt_label,
                        "predicted_match": rows[0]["matched_image"] if rows else "",
                        "predicted_label": predicted_label,
                    }
                )
        else:
            stats["unique_total"] += 1
            if not predicted_match:
                stats["unique_hits"] += 1
            else:
                stats["mismatches"].append(
                    {
                        "image": img_path.name,
                        "expected_match": "",
                        "expected_label": gt_label,
                        "predicted_match": rows[0]["matched_image"] if rows else "",
                        "predicted_label": predicted_label,
                    }
                )

    return stats


def format_summary(stats: Dict[str, object]) -> str:
    dup_total = stats["duplicate_total"] or 1
    uniq_total = stats["unique_total"] or 1
    lines: List[str] = []
    lines.append(
        f"Duplicate accuracy: {stats['duplicate_hits']}/{stats['duplicate_total']}"
        f" ({stats['duplicate_hits']/dup_total:.1%})"
    )
    lines.append(
        f"Unique accuracy: {stats['unique_hits']}/{stats['unique_total']}"
        f" ({stats['unique_hits']/uniq_total:.1%})"
    )
    mismatches = stats["mismatches"]
    if mismatches:
        lines.append("\nMismatches:")
        for miss in mismatches:
            lines.append(
                f" - {miss['image']}: expected {miss['expected_match'] or 'unique'}"
                f" → predicted {miss['predicted_match'] or miss['predicted_label']}"
            )
    else:
        lines.append("\nAll samples matched expected labels.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate synthetic DupCheck dataset")
    p.add_argument("--db_dir", default="data/synth_db")
    p.add_argument("--input_dir", default="data/synth_new")
    p.add_argument("--labels", default="data/synth_labels.csv")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--phash_thresh", type=int, default=10)
    p.add_argument("--orb_inliers_thresh", type=int, default=25)
    p.add_argument("--ncc_thresh", type=float, default=0.92)
    p.add_argument("--vector_score_thresh", type=float, default=0.0)
    p.add_argument("--roi_margin_ratio", type=float, default=0.12)
    p.add_argument("--max_roi_matches", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    missing = check_dependencies()
    if missing:
        print("Warning: required imaging dependencies missing; results will be unreliable.")
        for item in missing:
            print(f" - {item}")
        print("Install them via `pip install -r requirements.txt` and re-run this script.")
        return
    stats = evaluate(
        Path(args.db_dir),
        Path(args.input_dir),
        Path(args.labels),
        topk=args.topk,
        phash_thresh=args.phash_thresh,
        orb_inliers_thresh=args.orb_inliers_thresh,
        ncc_thresh=args.ncc_thresh,
        vector_score_thresh=args.vector_score_thresh,
        roi_margin_ratio=args.roi_margin_ratio,
        max_roi_matches=args.max_roi_matches,
    )
    print(format_summary(stats))


if __name__ == "__main__":
    main()
