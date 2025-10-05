# DupCheck — 图片重复与伪造检测

## 项目简介
DupCheck 针对理赔审核环节中“重复上传、裁剪拼接、轻度篡改”等骗赔手段，自动比对新上传照片与历史图库，识别完全重复、局部重复以及轻度改动的图像，并输出可供人工复核的证据。

项目仅依赖常见的 Python 图像库，便于集成到现有的上传或后台审核系统。

## 检测流程
1. **构建索引**：对图库图片计算多姿态 pHash（原图、旋转、翻转）、块哈希，以及缓存 ORB 特征，确保几何变换仍可召回。
2. **召回候选**：新上传图片通过 pHash/块哈希匹配，如有需要再结合多姿态 ORB 比对，将旋转、翻转的嫌疑图拉入候选集。
3. **精排验证**：对最佳姿态组合执行 ORB + RANSAC，若单应关系可靠，则在对应区域做 NCC，判断是否为 `exact_patch`。
4. **结果输出**：检测结论写入 `dup_report.csv`，命令行可生成对照证据图，辅助人工审核。

## 目录结构
- `duplicate_check/` —— 核心库模块（`features`、`indexer`、`matcher`、`report`）。
- `dupcheck_cli.py` —— 主命令行工具，支持内存索引或 SQLite 索引。
- `duplicate_check.py` —— 兼容性入口脚本。
- `tools/` —— 合成数据生成、阈值调参等辅助脚本。
- `tests/` 与 `run_smoke.py` —— 端到端冒烟验证。
- `data/` —— 文档示例使用的合成数据集。

## 环境依赖
建议在 Python 3.9 及以上版本下创建虚拟环境，并安装 `requirements.txt` 中的依赖。若缺少 OpenCV、Pillow、imagehash 则会自动降级，但完整功能需要这些包。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速体验
1. 生成示例数据集：
   ```bash
   python tools/generate_synthetic.py --out_dir data --count 5
   ```
2. 重建 SQLite 索引并执行检测：
   ```bash
   python dupcheck_cli.py \
     --db_dir data/synth_db \
     --input_dir data/synth_new \
     --out_dir reports \
     --index_db ./index.db \
     --rebuild_index
   ```
3. 查看 `reports/dup_report.csv` 以及生成的证据图片。

若要复用已有索引，可省略 `--rebuild_index`。通过调整 `--phash_thresh`、`--orb_inliers_thresh`、`--ncc_thresh` 等参数探索查准率和召回率的平衡。

## 常用命令
```bash
# 重建索引
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --index_db ./index.db --rebuild_index

# 自定义阈值运行
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --phash_thresh 12 --orb_inliers_thresh 30 --ncc_thresh 0.94

# 直接使用缓存索引
python dupcheck_cli.py --db_dir data/synth_db --input_dir data/synth_new --out_dir reports --index_db ./index.db
```

## 阈值调参
使用 `tools/tune_thresholds.py` 对多个阈值组合做网格搜索：

```bash
python tools/tune_thresholds.py \
  --labels data/synth_labels.csv \
  --db_dir data/synth_db \
  --input_dir data/synth_new \
  --out_dir reports/tune_out
```

脚本会输出 `tune_results.csv`，其中包含每组参数的 TP/FP/FN 统计，可据此锁定最适合的数据集配置。
```
