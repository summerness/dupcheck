# DupCheck — Duplicate & Tamper Detection / 图片重复与伪造检测

<div align="right">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</div>

---

<details open>
<summary id="english"><strong>English</strong></summary>

### Overview
DupCheck targets a recurring fraud scenario in repair claims: contractors upload recycled or lightly edited photos to claim duplicate reimbursements. The system compares every new image against a reference gallery, flags exact copies, crops, rotations/flips, and subtle edits, then produces reviewer-friendly evidence.

The implementation is pure Python and depends only on widely available imaging libraries, which keeps integration with existing intake or back-office pipelines straightforward.

### Detection flow
1. **Index build** – gallery images are converted to multi-orientation pHash, multi-scale tile hashes, cached ORB descriptors, and optional ResNet-18 / CLIP embeddings so geometric tweaks and coarse semantics remain discoverable.
2. **Candidate recall** – a new upload is matched through pHash buckets, tile voting, and optional FAISS (ResNet-18/CLIP) vector search; if necessary, orientation-aware ORB matching pulls in additional suspects.
3. **Verification** – the best orientation pair runs ORB + RANSAC. When the homography is reliable, NCC on the corresponding patch promotes the match to `exact_patch`.
4. **Reporting** – matches are recorded in `dup_report.csv`, and the CLI can render side-by-side evidence images for manual review.

> **Scaling tip:** Set `DUPC_VECTOR_INDEX=ivf_pq` or `hnsw` to switch the built-in FAISS index; for very large galleries or cluster deployments, replace the in-process FAISS index with an external vector database (e.g., Milvus, Qdrant, Pinecone). A natural hook is the `duplicate_check/indexer.py::build_index` / `load_index_from_db` functions—swap the FAISS creation for remote writes, and query that service inside `matcher.recall_candidates` before running ORB reranking.

### Project layout
- `duplicate_check/` — core modules (`features`, `indexer`, `matcher`, `report`).
- `dupcheck_cli.py` — main CLI with in-memory and SQLite index support.
- `duplicate_check.py` — legacy entrypoint kept for backward compatibility.
- `tools/` — helpers for synthetic data generation and threshold tuning.
- `tests/` & `run_smoke.py` — minimal smoke coverage for the end-to-end flow.
- `data/` — synthetic dataset used in docs and examples.

### Requirements
Install the dependencies from `requirements.txt` inside a Python 3.9+ environment. Pillow, OpenCV, imagehash, `torch`, `torchvision`, and (optionally) `faiss-cpu` unlock the full feature set; the code degrades gracefully if some extras are missing.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras: install `faiss-cpu` (for ANN recall) and either `open-clip-torch` or `clip` if you want CLIP-ViT embeddings in addition to ResNet.

### Quick start
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
3. Inspect `reports/dup_report.csv` alongside the generated evidence JPEGs.
4. (Optional) Benchmark on the synthetic labels and inspect mismatches:
   ```bash
   python tools/verify_synthetic.py \
     --db_dir data/synth_db \
     --input_dir data/synth_new \
     --labels data/synth_labels.csv \
     --phash_thresh 16 \
     --orb_inliers_thresh 6 \
     --ncc_thresh 0.85
   ```

Drop `--rebuild_index` to reuse a cached index. Tune `--phash_thresh`, `--orb_inliers_thresh`, and `--ncc_thresh` to explore different precision/recall tradeoffs.

### CLI examples
```bash
# Rebuild index for fresh data
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --index_db ./index.db --rebuild_index

# Run with custom thresholds
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --phash_thresh 12 --orb_inliers_thresh 30 --ncc_thresh 0.94

# Quick scan using the cached index
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --index_db ./index.db
```

### Threshold tuning
Use `tools/tune_thresholds.py` with the synthetic labels to sweep detection thresholds:

```bash
python tools/tune_thresholds.py \
  --labels data/synth_labels.csv \
  --db_dir data/synth_db \
  --input_dir data/synth_new \
  --out_dir reports/tune_out
```

The script writes `tune_results.csv` containing TP/FP/FN counts for each parameter combo so you can lock in thresholds for your own dataset.

</details>

<details>
<summary id="中文"><strong>中文</strong></summary>

### 项目简介
DupCheck 聚焦理赔审核中的骗赔套路：重复提交、裁剪拼接、亮度/压缩篡改等。系统会把新上传图片与历史图库逐一比对，识别完全重复、局部重复、旋转/翻转及轻度改动的图像，并输出便于人工复核的证据。

项目依赖常见的 Python 图像 / 深度学习库，可嵌入现有的上传或后台审核流程。

### 检测流程
1. **构建索引**：对图库图片计算多姿态 pHash（原图、旋转、翻转）、多尺度块哈希、缓存 ORB 关键点，并可生成 ResNet-18 / CLIP 嵌入，确保几何和粗语义变化也能被召回。
2. **召回候选**：新图片通过 pHash/块哈希匹配，并可结合基于 ResNet-18/CLIP 的 FAISS 向量检索；如有需要再执行多姿态 ORB 匹配，把旋转、翻转的嫌疑图拉入候选集。
3. **精排验证**：对最佳姿态组合执行 ORB + RANSAC，若单应关系稳定，则在对应区域做 NCC，判断是否为 `exact_patch`。
4. **结果输出**：检测结论写入 `dup_report.csv`，命令行可生成对照证据图，辅助人工审核。

> **扩展建议**：可通过设置环境变量 `DUPC_VECTOR_INDEX=ivf_pq` 或 `hnsw` 切换内置 FAISS 索引；若图库规模巨大或需集群部署，可在 `duplicate_check/indexer.py` / `load_index_from_db` 中替换 FAISS，为 Milvus、Qdrant、Pinecone 等外部向量库写入，并在 `matcher.recall_candidates` 中调用该服务。

### 目录结构
- `duplicate_check/` —— 核心模块（`features`、`indexer`、`matcher`、`report`）。
- `dupcheck_cli.py` —— 主命令行工具，支持内存或 SQLite 索引。
- `duplicate_check.py` —— 保留的兼容性入口脚本。
- `tools/` —— 合成数据生成、阈值调参等辅助脚本。
- `tests/` 与 `run_smoke.py` —— 端到端冒烟验证。
- `data/` —— 文档示例所用的合成数据集。

### 环境依赖
建议在 Python 3.9+ 中创建虚拟环境，并安装 `requirements.txt` 列出的依赖。OpenCV、Pillow、imagehash、`torch`、`torchvision` 与可选的 `faiss-cpu` 能启用全部功能，缺失时流程会自动降级。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

可选依赖：`faiss-cpu`（向量召回），以及 `open-clip-torch` 或 `clip`（启用 CLIP-ViT 向量）。

### 快速体验
1. 生成示例数据集：
   ```bash
   python tools/generate_synthetic.py --out_dir data --count 5
   ```
2. 重建 SQLite 索引并运行检测：
   ```bash
   python dupcheck_cli.py \
     --db_dir data/synth_db \
     --input_dir data/synth_new \
     --out_dir reports \
     --index_db ./index.db \
     --rebuild_index
   ```
3. 查看 `reports/dup_report.csv` 及生成的证据图片。
4. （可选）对合成标注集进行评估，查看召回差异：
   ```bash
   python tools/verify_synthetic.py \
     --db_dir data/synth_db \
     --input_dir data/synth_new \
     --labels data/synth_labels.csv \
     --phash_thresh 16 \
     --orb_inliers_thresh 6 \
     --ncc_thresh 0.85
   ```

如需复用已有索引，可省略 `--rebuild_index`。可通过 `--phash_thresh`、`--orb_inliers_thresh`、`--ncc_thresh` 调整查准率与召回率之间的权衡。

### 常用命令
```bash
# 重建索引
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --index_db ./index.db --rebuild_index

# 自定义阈值运行
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --phash_thresh 12 --orb_inliers_thresh 30 --ncc_thresh 0.94

# 使用已有索引快速扫描
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --index_db ./index.db
```

### 阈值调参
使用 `tools/tune_thresholds.py` 对阈值组合进行网格搜索：

```bash
python tools/tune_thresholds.py \
  --labels data/synth_labels.csv \
  --db_dir data/synth_db \
  --input_dir data/synth_new \
  --out_dir reports/tune_out
```

脚本会输出 `tune_results.csv`，包含每组参数的 TP/FP/FN 统计，可据此锁定适合业务数据的阈值。

</details>
