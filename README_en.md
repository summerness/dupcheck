# DupCheck — Duplicate & Tamper Detection

## Overview
DupCheck solves broad “duplicate / tamper detection” needs: It works in insurance claim review, content moderation, e-commerce authenticity checks, and copyright protection. It began as a submodule designed to stop third-party repair contractors from re-uploading maintenance photos to claim duplicate reimbursements; I later spun it out, optimised it, and expanded it into a general-purpose toolkit. Uploads are compared against a reference gallery to flag exact copies, crops, rotations, flips, and lightly edited variants, producing reviewer-friendly evidence.

The pipeline is pure Python with minimal dependencies, making it easy to embed into intake pipelines or back-office review systems.

## Detection flow
1. **Index build** – each gallery image is converted to multiple perceptual hashes (original, rotations, flips), multi-scale tile hashes, cached ORB descriptors, and optional ResNet-18 / CLIP embeddings to support geometric and coarse semantic changes.
2. **Candidate recall** – a new upload is compared with the index via pHash buckets, tile voting, and optional FAISS (ResNet-18/CLIP) vector search; if needed, multi-orientation ORB matching pulls in additional suspects.
3. **Verification** – the best orientation pair runs ORB + RANSAC. When the homography is reliable, NCC on the corresponding patch upgrades matches to `exact_patch`.
4. **Reporting** – results are written to `dup_report.csv`, and the CLI can render side-by-side evidence images for manual review.
5. **Threshold tuning** – optionally run `tools/tune_thresholds.py` to grid-search `phash/ORB/NCC` thresholds and pick the best configuration for your data.

> **Scaling tip:** Set `DUPC_VECTOR_INDEX=ivf_pq` or `hnsw` to switch the built-in FAISS index; for even larger deployments, replace the FAISS block in `duplicate_check/indexer.py` / `load_index_from_db` with writes to Milvus, Qdrant, Pinecone, etc., and query that service from `matcher.recall_candidates` before ORB reranking.
> **Performance tip:** Tune `DUPC_TILE_SCALES` (e.g., `1.0,0.6`) and `DUPC_TILE_GRID` to trade multi-scale robustness for runtime when processing massive galleries.

## Project layout
- `duplicate_check/` — core library modules (`features`, `indexer`, `matcher`, `report`).
- `dupcheck_cli.py` — main CLI wrapper supporting in-memory or SQLite indices.
- `duplicate_check.py` — minimal entry point kept for backwards compatibility.
- `tools/` — utilities for synthetic data generation and threshold tuning.
- `tests/` — quick test.
- `data/` — sample synthetic dataset used by the documentation examples.

## Requirements
Install dependencies listed in `requirements.txt` inside a Python 3.9+ environment. OpenCV, Pillow, imagehash, `torch`, `torchvision`, and (optionally) `faiss-cpu` enable the full feature set; the pipeline falls back gracefully if some extras are unavailable.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras: install `faiss-cpu` (for ANN recall) and either `open-clip-torch` or `clip` if you want CLIP-ViT embeddings in addition to ResNet.

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
5. (Optional) Launch a grid search over thresholds:
   ```bash
   python tools/tune_thresholds.py \
     --labels data/synth_labels.csv \
     --db_dir data/synth_db \
     --input_dir data/synth_new \
     --out_dir reports/tune_out
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

## License

This project is released under the [MIT License](LICENSE).
