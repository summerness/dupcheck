"""Index utilities with in-memory + SQLite backends supporting tile lookups."""
from contextlib import closing
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

from duplicate_check.features import compute_phash, compute_tile_hashes, compute_phash_variants


def build_index(db_dir: Path, tile_grid: int = 8) -> Dict[str, Any]:
    """Build an in-memory index containing phash and tile hashes."""
    idx = {"by_id": {}, "by_phash": {}, "by_tile": {}}
    for p in sorted(db_dir.iterdir()):
        if not p.is_file():
            continue
        pid = p.name
        ph_variants = compute_phash_variants(p)
        primary_ph = ph_variants[0]
        tiles = compute_tile_hashes(p, grid=tile_grid)
        idx["by_id"][pid] = {
            "path": str(p),
            "phash": primary_ph,
            "phash_variants": ph_variants,
            "tiles": tiles,
        }
        for ph in ph_variants:
            bucket = idx["by_phash"].setdefault(ph, [])
            if pid not in bucket:
                bucket.append(pid)
        for th, bbox in tiles:
            idx["by_tile"].setdefault(th, []).append((pid, bbox))
    return idx


def load_index(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(idx: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx, f)


def init_sqlite(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    with closing(conn):
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS images(\n"
            "    img_id TEXT PRIMARY KEY,\n"
            "    path TEXT,\n"
            "    phash TEXT,\n"
            "    w INTEGER,\n"
            "    h INTEGER\n"
            ")"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS tiles(\n"
            "    img_id TEXT,\n"
            "    tile_hash TEXT,\n"
            "    x0 INTEGER,\n"
            "    y0 INTEGER,\n"
            "    x1 INTEGER,\n"
            "    y1 INTEGER\n"
            ")"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tiles_hash ON tiles(tile_hash)")
        conn.commit()


def add_image_to_db(db_path: Path, image_path: Path, tile_grid: int = 8) -> None:
    conn = sqlite3.connect(str(db_path))
    with closing(conn):
        cur = conn.cursor()
        ph = compute_phash(image_path)
        tiles = compute_tile_hashes(image_path, grid=tile_grid)
        try:
            from PIL import Image

            w, h = Image.open(str(image_path)).size
        except Exception:
            w, h = 0, 0
        img_id = image_path.name
        cur.execute(
            "INSERT OR REPLACE INTO images(img_id,path,phash,w,h) VALUES (?,?,?,?,?)",
            (img_id, str(image_path), ph, w, h),
        )
        cur.execute("DELETE FROM tiles WHERE img_id = ?", (img_id,))
        cur.executemany(
            "INSERT INTO tiles(img_id,tile_hash,x0,y0,x1,y1) VALUES (?,?,?,?,?,?)",
            [(img_id, th, bbox[0], bbox[1], bbox[2], bbox[3]) for th, bbox in tiles],
        )
        conn.commit()


def build_index_db(db_dir: Path, db_path: Path, tile_grid: int = 8) -> None:
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
        for img_id, path, phash, w, h in cur.execute(
            "SELECT img_id,path,phash,w,h FROM images"
        ):
            idx["by_id"][img_id] = {"path": path, "phash": phash, "phash_variants": [phash], "tiles": []}
            idx["by_phash"].setdefault(phash, []).append(img_id)
        for img_id, th, x0, y0, x1, y1 in cur.execute(
            "SELECT img_id,tile_hash,x0,y0,x1,y1 FROM tiles"
        ):
            tile_info = (th, (x0, y0, x1, y1))
            idx["by_id"].setdefault(img_id, {}).setdefault("tiles", []).append(tile_info)
            idx["by_tile"].setdefault(th, []).append((img_id, tile_info[1]))

    # Augment with variant phashes for better recall
    for img_id, rec in list(idx["by_id"].items()):
        path = Path(rec.get("path", ""))
        try:
            variants = compute_phash_variants(path)
        except Exception:
            variants = [rec.get("phash")]
        rec["phash_variants"] = variants or [rec.get("phash")]
        for ph in rec["phash_variants"]:
            if not ph:
                continue
            bucket = idx["by_phash"].setdefault(ph, [])
            if img_id not in bucket:
                bucket.append(img_id)
    return idx
