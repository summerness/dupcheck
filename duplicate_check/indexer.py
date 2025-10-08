"""Index utilities with in-memory + SQLite backends supporting tile & vector lookups."""
from contextlib import closing
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:
    np = None

try:
    import faiss
except Exception:
    faiss = None

from duplicate_check.features import (
    compute_phash,
    compute_tile_hashes,
    compute_phash_variants,
    compute_embedding,
)

VECTOR_INDEX_TYPE = os.environ.get("DUPC_VECTOR_INDEX", "flat").lower()
VECTOR_INDEX_NLIST = int(os.environ.get("DUPC_VECTOR_NLIST", "1024"))
VECTOR_INDEX_PQ_M = int(os.environ.get("DUPC_VECTOR_PQ_M", "16"))
VECTOR_HNSW_M = int(os.environ.get("DUPC_VECTOR_HNSW_M", "32"))
VECTOR_HNSW_EF = int(os.environ.get("DUPC_VECTOR_HNSW_EF", "64"))


def _build_vector_index(embeddings: List[Any], ids: List[str]) -> Optional[Dict[str, Any]]:
    if not embeddings or faiss is None or np is None:
        return None
    try:
        mat = np.stack(embeddings).astype("float32")
    except Exception:
        return None
    if mat.size == 0:
        return None
    dim = mat.shape[1]
    index = None
    metric = "ip"
    index_type = VECTOR_INDEX_TYPE
    try:
        if index_type == "ivf_pq" and mat.shape[0] > VECTOR_INDEX_PQ_M:
            nlist = min(max(1, VECTOR_INDEX_NLIST), mat.shape[0])
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, VECTOR_INDEX_PQ_M, 8)
            index.train(mat)
            index.add(mat)
            index.nprobe = max(1, min(nlist, nlist // 10 or 1))
        elif index_type == "hnsw":
            hnsw_m = max(2, VECTOR_HNSW_M)
            index = faiss.IndexHNSWFlat(dim, hnsw_m)
            index.hnsw.efConstruction = max(hnsw_m, VECTOR_HNSW_EF)
            index.add(mat)
            index.hnsw.efSearch = max(hnsw_m, VECTOR_HNSW_EF)
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(mat)
            index_type = "flat"
    except Exception:
        index = None
    if index is None:
        return None
    return {"index": index, "ids": ids, "metric": metric, "type": index_type}


def build_index(db_dir: Path, tile_grid: int = 8) -> Dict[str, Any]:
    """Build an in-memory index containing multi-scale hashes and optional vectors."""
    idx = {"by_id": {}, "by_phash": {}, "by_tile": {}, "vector": None}
    use_vectors = faiss is not None and np is not None
    vector_embeddings = []
    vector_ids = []
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
        for tile in tiles:
            th = tile.get("hash")
            if not th:
                continue
            entry = {
                "img_id": pid,
                "bbox": tile.get("bbox", (0, 0, 0, 0)),
                "scale": tile.get("scale", 1.0),
            }
            idx["by_tile"].setdefault(th, []).append(entry)
        if use_vectors and np is not None:
            try:
                emb_val = compute_embedding(p)
                if emb_val is None:
                    continue
                emb = np.asarray(emb_val, dtype=np.float32)
                if emb.ndim == 1 and emb.size > 0:
                    vector_embeddings.append(emb)
                    vector_ids.append(pid)
            except Exception:
                continue
    if use_vectors and vector_embeddings:
        idx["vector"] = _build_vector_index(vector_embeddings, vector_ids)
    return idx


def load_index(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(idx: Dict[str, Any], path: Path) -> None:
    to_dump = dict(idx)
    if "vector" in to_dump:
        to_dump["vector"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dump, f)


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
            "    y1 INTEGER,\n"
            "    scale REAL DEFAULT 1.0\n"
            ")"
        )
        try:
            cur.execute("PRAGMA table_info(tiles)")
            existing_cols = {row[1] for row in cur.fetchall()}
            if "scale" not in existing_cols:
                cur.execute("ALTER TABLE tiles ADD COLUMN scale REAL DEFAULT 1.0")
        except Exception:
            pass
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
        tile_rows = []
        for tile in tiles:
            th = tile.get("hash")
            bbox = tile.get("bbox", (0, 0, 0, 0))
            scale = tile.get("scale", 1.0)
            tile_rows.append((img_id, th, bbox[0], bbox[1], bbox[2], bbox[3], float(scale)))
        cur.executemany(
            "INSERT INTO tiles(img_id,tile_hash,x0,y0,x1,y1,scale) VALUES (?,?,?,?,?,?,?)",
            tile_rows,
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
    idx = {"by_id": {}, "by_phash": {}, "by_tile": {}, "vector": None}
    with closing(conn):
        cur = conn.cursor()
        for img_id, path, phash, w, h in cur.execute(
            "SELECT img_id,path,phash,w,h FROM images"
        ):
            idx["by_id"][img_id] = {"path": path, "phash": phash, "phash_variants": [phash], "tiles": []}
            idx["by_phash"].setdefault(phash, []).append(img_id)
        try:
            tile_rows = cur.execute(
                "SELECT img_id,tile_hash,x0,y0,x1,y1,scale FROM tiles"
            )
            scale_included = True
        except sqlite3.OperationalError:
            tile_rows = cur.execute(
                "SELECT img_id,tile_hash,x0,y0,x1,y1 FROM tiles"
            )
            scale_included = False
        for row in tile_rows:
            if scale_included:
                img_id, th, x0, y0, x1, y1, scale = row
            else:
                img_id, th, x0, y0, x1, y1 = row
                scale = 1.0
            tile_entry = {
                "hash": th,
                "bbox": (x0, y0, x1, y1),
                "scale": float(scale),
            }
            rec = idx["by_id"].setdefault(
                img_id,
                {"path": "", "phash": "", "phash_variants": [], "tiles": []},
            )
            rec.setdefault("tiles", []).append(tile_entry)
            idx["by_tile"].setdefault(th, []).append(
                {"img_id": img_id, "bbox": tile_entry["bbox"], "scale": tile_entry["scale"]}
            )

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
    # Build vector index on demand
    use_vectors = faiss is not None and np is not None
    if use_vectors:
        vector_embeddings: List[np.ndarray] = []
        vector_ids: List[str] = []
        for img_id, rec in idx["by_id"].items():
            path = rec.get("path")
            if not path:
                continue
            try:
                emb_val = compute_embedding(Path(path))
            except Exception:
                emb_val = None
            if emb_val is None:
                continue
            try:
                arr = np.asarray(emb_val, dtype=np.float32)
            except Exception:
                continue
            if arr.ndim != 1 or arr.size == 0:
                continue
            vector_embeddings.append(arr)
            vector_ids.append(img_id)
        idx["vector"] = _build_vector_index(vector_embeddings, vector_ids)
    return idx
