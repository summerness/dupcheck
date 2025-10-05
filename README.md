# DupCheck — 图片重复与伪造检测（中英双语说明）

English
-------
DupCheck is a minimal, extensible prototype for duplicate / tamper detection
on uploaded repair images. It provides a pipeline that performs:

- fast recall (perceptual hash, tile-hash)
- fine-grained verification (ORB feature matching + RANSAC)
- precise patch equality check (NCC template matching)
- report generation (CSV + evidence images)


中文
----
DupCheck 是一个用于检测维修图片重复与伪造的最小可扩展原型。它实现了：

- 快速召回（感知哈希 pHash、块哈希 Tile-hash）
- 精排验证（ORB 特征匹配 + RANSAC）
- 精确子图一致性判断（标准化互相关 NCC）
- 报告输出（CSV + 证据图片）

仓库结构 / Repository layout
-----------------------------
- `duplicate_check/` — 核心模块
  - `features.py` — pHash、tile-hash、ORB 特征提取
  - `indexer.py` — 索引构建（内存/SQLite）与持久化接口
  - `matcher.py` — 召回、精排、判定逻辑
  - `report.py` — CSV 与证据图生成
- `dupcheck_cli.py` — 命令行入口（索引构建 + 批量检测）
- `tools/` — 辅助脚本（合成数据、阈值调优）
- `data/` — 示例/合成数据（推荐放图库与待测图片）
- `requirements.txt` — 推荐依赖

快速开始 / Quick Start
-----------------------
1. 创建并激活虚拟环境（推荐）：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

2. 生成或准备示例数据（可用 `tools/generate_synthetic.py`）：

```bash
python tools/generate_synthetic.py --out_dir data --count 5
```

3. 重建并使用 SQLite 索引，然后运行检测：

```bash
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --index_db ./index.db --rebuild_index
```

4. 输出：`reports/dup_report.csv`（CSV）与证据图片（reports/*.jpg）

重要参数 / Key thresholds
---------------------------
- `phash_thresh`（pHash Hamming distance）: 默认 10（召回阈值，可通过 CLI `--phash_thresh` 调整）
- `orb_inliers_thresh`（ORB 内点数阈值）: 默认 25（精排阈值，可通过 `--orb_inliers_thresh` 调整）
- `ncc_thresh`（NCC 峰值）: 默认 0.92（判断 exact_patch 的阈值，可通过 `--ncc_thresh` 调整）

阈值调优 / Tuning thresholds
----------------------------
使用 `tools/tune_thresholds.py` 在带标注的合成/真实数据上做网格搜索：

```bash
. .venv/bin/activate
python tools/tune_thresholds.py --labels data/synth_labels.csv --db_dir data/synth_db --input_dir data/synth_new --out_dir /tmp/tune_out
```


