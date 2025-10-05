# DupCheck — Duplicate & Tamper Detection

## Overview
DupCheck addresses a recurring issue in repair-claim workflows: contractors reuse or patch previous photos to obtain duplicate reimbursements. The tool analyses an upload against a curated gallery and flags exact copies, tight crops, rotations, flips, and lightly edited versions.

The pipeline is entirely Python based and keeps dependencies minimal so it can be embedded in existing intake or review systems.

## Detection flow
1. **Index build** – each gallery image is converted to multiple perceptual hashes (original, rotations, flips), tile hashes, cached ORB descriptors, and optional ResNet-18 embeddings to support geometric and coarse semantic changes.
2. **Candidate recall** – a new upload is compared with the index via pHash buckets, tile voting, and optional FAISS (ResNet-18) vector search; if needed, multi-orientation ORB matching pulls in additional suspects.
3. **Verification** – the best orientation pair runs ORB + RANSAC. When the homography is reliable, NCC on the corresponding patch upgrades matches to `exact_patch`.
4. **Reporting** – results are written to `dup_report.csv`, and the CLI can render side-by-side evidence images for manual review.

> **Scaling tip:** If the gallery grows beyond what in-process FAISS can handle, replace the FAISS block in `duplicate_check/indexer.py` / `load_index_from_db` with writes to Milvus, Qdrant, Pinecone, etc., and query that service from `matcher.recall_candidates` before ORB reranking.

## Project layout
- `duplicate_check/` — core library modules (`features`, `indexer`, `matcher`, `report`).
- `dupcheck_cli.py` — main CLI wrapper supporting in-memory or SQLite indices.
- `duplicate_check.py` — minimal entry point kept for backwards compatibility.
- `tools/` — utilities for synthetic data generation and threshold tuning.
- `tests/` & `run_smoke.py` — quick smoke coverage for the end-to-end flow.
- `data/` — sample synthetic dataset used by the documentation examples.

## Requirements
Install dependencies listed in `requirements.txt` inside a Python 3.9+ environment. OpenCV, Pillow, imagehash, `torch`, `torchvision`, and (optionally) `faiss-cpu` enable the full feature set; the pipeline falls back gracefully if some extras are unavailable.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start
1. Generate the demo dataset:
   ```bash
   python tools/generate_synthetic.py --out_dir data --count 5
   ```
2. Rebuild the SQLite index and run detection:
   ```bash
   python dupcheck_cli.py \
     --db_dir data/synth_db \
     --input_dir data/synth_new \
     --out_dir reports \
     --index_db ./index.db \
     --rebuild_index
   ```
3. Inspect `reports/dup_report.csv` and the generated evidence JPEGs.
4. (Optional) Benchmark on the labelled synthetic set and review mismatches:
   ```bash
   python tools/verify_synthetic.py \
     --db_dir data/synth_db \
     --input_dir data/synth_new \
     --labels data/synth_labels.csv \
     --phash_thresh 16 \
     --orb_inliers_thresh 6 \
     --ncc_thresh 0.85
   ```

To reuse an existing index, drop the `--rebuild_index` flag. Tweak `--phash_thresh`, `--orb_inliers_thresh`, and `--ncc_thresh` to experiment with precision/recall.

## CLI examples
```bash
# Rebuild index for fresh data
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --index_db ./index.db --rebuild_index

# Run with custom thresholds
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --phash_thresh 12 --orb_inliers_thresh 30 --ncc_thresh 0.94

# Quick scan using the cached index
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --index_db ./index.db
```

## Threshold tuning
Use `tools/tune_thresholds.py` with the synthetic labels to grid-search thresholds:

```bash
python tools/tune_thresholds.py \
  --labels data/synth_labels.csv \
  --db_dir data/synth_db \
  --input_dir data/synth_new \
  --out_dir reports/tune_out
```

The script writes `tune_results.csv` with TP/FP/FN counts for each parameter combo, making it easy to lock in settings for your own data.
