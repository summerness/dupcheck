"""Feature extraction utilities: pHash, tile-hash, ORB descriptors.

This module implements:
- compute_phash(image_path) -> hex string
- compute_tile_hashes(image_path, grid) -> list of (hex, bbox)
- compute_orb_descriptors(image_path, max_features) -> {kps, descs}
- compute_features(image_path) -> ImageFeatures

If OpenCV/imagehash/Pillow are missing, functions will raise ImportError.

特征提取工具：pHash、块哈希（tile-hash）、ORB 特征。

本模块实现：
- compute_phash(image_path) -> 十六进制字符串
- compute_tile_hashes(image_path, grid) -> 返回 (hash, bbox) 列表
- compute_orb_descriptors(image_path, max_features) -> 返回 {kps, descs}
- compute_features(image_path) -> 返回 ImageFeatures

如果系统缺少 OpenCV/imagehash/Pillow，函数会进行降级或抛出异常。
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from pathlib import Path
import hashlib
import io
import os

# Optional dependencies
# 可选依赖
try:
    import imagehash
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    imagehash = None
    Image = None
    PIL_AVAILABLE = False

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None


@dataclass
class ImageFeatures:
    phash: str
    orb: Dict[str, Any]
    size: Tuple[int, int]


def compute_phash(image_path: Path, hash_size: int = 8) -> str:
    """Compute pHash for the image and return as hex string."""
    if PIL_AVAILABLE and imagehash is not None:
        img = Image.open(str(image_path)).convert("RGB")
        ph = imagehash.phash(img, hash_size=hash_size)
        return ph.__str__()
    # Fallback: use SHA1 of file contents and return truncated hex
    h = hashlib.sha1()
    with open(str(image_path), "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_tile_hashes(image_path: Path, grid: int = 8, hash_size: int = 8) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Split image into grid x grid tiles and compute pHash per tile.

    Returns list of (hex_hash, (x0,y0,x1,y1)).
    """
    tiles = []
    if not PIL_AVAILABLE or Image is None or imagehash is None:
        # Fallback: produce one tile using full-image phash fallback
        ph = compute_phash(image_path, hash_size=hash_size)
        tiles.append((ph, (0, 0, 0, 0)))
        return tiles
    img = Image.open(str(image_path)).convert("RGB")
    w, h = img.size
    gx = grid
    gy = grid
    tw = max(1, w // gx)
    th = max(1, h // gy)
    for yi in range(gy):
        for xi in range(gx):
            x0 = xi * tw
            y0 = yi * th
            x1 = x0 + tw if xi < gx - 1 else w
            y1 = y0 + th if yi < gy - 1 else h
            crop = img.crop((x0, y0, x1, y1))
            ph = imagehash.phash(crop, hash_size=hash_size)
            tiles.append((ph.__str__(), (x0, y0, x1, y1)))
    return tiles


def compute_orb_descriptors(image_path: Path, max_features: int = 2000) -> Dict:
    """Extract ORB keypoints and descriptors using OpenCV.

    Returns dict {"kps": list of cv2.KeyPoint, "descs": np.ndarray}
    """
    if cv2 is None:
        # Graceful fallback: return empty descriptors
        # 优雅降级：返回空的关键点/描述子
        return {"kps": [], "descs": None}
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"Unable to read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    kps, descs = orb.detectAndCompute(gray, None)
    return {"kps": kps, "descs": descs}


def compute_features(image_path: Path, orb_max_features: int = 2000, tile_grid: int = 8) -> ImageFeatures:
    ph = compute_phash(image_path)
    orb = {}
    try:
        orb = compute_orb_descriptors(image_path, max_features=orb_max_features)
    except Exception:
        orb = {"kps": [], "descs": None}
    size = (0, 0)
    if PIL_AVAILABLE and Image is not None:
        try:
            img = Image.open(str(image_path))
            size = img.size
        except Exception:
            size = (0, 0)
    feats = ImageFeatures(phash=ph, orb=orb, size=size)
    # attach source path for downstream tile recall
    # 将源路径附加到特征对象，以便后续 tile 召回使用
    try:
        feats._path = str(image_path)
    except Exception:
        pass
    return feats

