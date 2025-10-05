"""Index building utilities storing phash and tile-hash index.

Current implementation builds an in-memory index with structure:
{
    'by_id': { img_id: {path, phash, tiles: [(tile_hash, bbox), ...]} },
    'by_phash': { phash: [img_id,...] },
    'by_tile': { tile_hash: [(img_id, bbox), ...] }
}

This is suitable for small-medium datasets and for unit testing. For
production use, persist to disk (SQLite/LevelDB) or use FAISS/Milvus for vectors.

索引构建工具：存储 phash 和 tile-hash 索引。

当前实现创建了一个内存索引，结构如下：
{
    'by_id': { img_id: {path, phash, tiles: [(tile_hash, bbox), ...]} },
    'by_phash': { phash: [img_id,...] },
    'by_tile': { tile_hash: [(img_id, bbox), ...] }
}

该实现适用于小到中等规模的数据集以及单元测试。生产环境建议将索引持久化到磁盘（SQLite/LevelDB），或使用 FAISS/Milvus 存储向量索引。
"""
from pathlib import Path
from typing import Dict, Any
from duplicate_check.features import compute_phash, compute_tile_hashes
import sqlite3
import json
from contextlib import closing


def build_index(db_dir: Path, tile_grid: int = 8) -> Dict[str, Any]:
    idx = {"by_id": {}, "by_phash": {}, "by_tile": {}}
    for p in sorted(db_dir.iterdir()):
        if not p.is_file():
            continue
        pid = p.name
        ph = compute_phash(p)
        tiles = compute_tile_hashes(p, grid=tile_grid)
        idx["by_id"][pid] = {"path": str(p), "phash": ph, "tiles": tiles}
        idx["by_phash"].setdefault(ph, []).append(pid)
        for th, bbox in tiles:
            idx["by_tile"].setdefault(th, []).append((pid, bbox))
    return idx


def load_index(path: Path):
    # load JSON index
    # 从 JSON 文件加载索引
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(idx: Dict[str, Any], path: Path):
    # save JSON index
    # 将内存索引保存为 JSON 文件
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx, f)


### SQLite-backed index persistence (simple)
def init_sqlite(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    with closing(conn):
        cur = conn.cursor()
        # enable WAL for better concurrency
        # 启用 WAL 模式以提高并发性能
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("CREATE TABLE IF NOT EXISTS images(img_id TEXT PRIMARY KEY, path TEXT, phash TEXT, w INTEGER, h INTEGER)")
        cur.execute("CREATE TABLE IF NOT EXISTS tiles(img_id TEXT, tile_hash TEXT, x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tiles_hash ON tiles(tile_hash)")
        conn.commit()


def add_image_to_db(db_path: Path, image_path: Path, tile_grid: int = 8):
    conn = sqlite3.connect(str(db_path))
    with closing(conn):
        cur = conn.cursor()
        # compute features
        # 计算 pHash 和块哈希
        ph = compute_phash(image_path)
        tiles = compute_tile_hashes(image_path, grid=tile_grid)
        stat = None
        try:
            from PIL import Image
            img = Image.open(str(image_path))
            w, h = img.size
        except Exception:
            w, h = 0, 0
        img_id = image_path.name
        cur.execute("INSERT OR REPLACE INTO images(img_id,path,phash,w,h) VALUES (?,?,?,?,?)", (img_id, str(image_path), ph, w, h))
        # delete existing tiles
        cur.execute("DELETE FROM tiles WHERE img_id = ?", (img_id,))
        cur.executemany("INSERT INTO tiles(img_id,tile_hash,x0,y0,x1,y1) VALUES (?,?,?,?,?,?)", [(img_id, th, bbox[0], bbox[1], bbox[2], bbox[3]) for th, bbox in tiles])
        conn.commit()


def build_index_db(db_dir: Path, db_path: Path, tile_grid: int = 8):
    init_sqlite(db_path)
    for p in sorted(db_dir.iterdir()):
        if not p.is_file():
            continue
        add_image_to_db(db_path, p, tile_grid=tile_grid)


def load_index_from_db(db_path: Path) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    idx = {"by_id": {}, "by_phash": {}, "by_tile": {}}
    with closing(conn):
        cur = conn.cursor()
        for row in cur.execute("SELECT img_id,path,phash,w,h FROM images"):
            img_id, path, phash, w, h = row
            idx["by_id"][img_id] = {"path": path, "phash": phash, "tiles": []}
            idx["by_phash"].setdefault(phash, []).append(img_id)
        for row in cur.execute("SELECT img_id,tile_hash,x0,y0,x1,y1 FROM tiles"):
            img_id, th, x0, y0, x1, y1 = row
            idx["by_id"].setdefault(img_id, {}).setdefault("tiles", []).append((th, (x0, y0, x1, y1)))
            idx["by_tile"].setdefault(th, []).append((img_id, (x0, y0, x1, y1)))
    return idx

"""Index building utilities (simple file-based index skeleton).

This module creates a minimal index mapping image ids to phash and path.
Real implementation should persist index to disk or use FAISS/DB.
"""
from pathlib import Path
from typing import Dict, Any
from duplicate_check.features import compute_phash


def build_index(db_dir: Path) -> Dict[str, Any]:
    idx = {"by_id": {}, "by_phash": {}}
    for p in sorted(db_dir.iterdir()):
        if not p.is_file():
            continue
        pid = p.name
        ph = compute_phash(p)
        idx["by_id"][pid] = {"path": str(p), "phash": ph}
        idx["by_phash"].setdefault(ph, []).append(pid)
    return idx
