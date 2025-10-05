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
from typing import Any, Dict, List, Tuple, Optional
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

try:
    import torch
    from torchvision import models
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    models = None
    TORCH_AVAILABLE = False

_EMBED_MODEL = None
_EMBED_TRANSFORM = None

@dataclass
class ImageFeatures:
    phash: str
    orb: Dict[str, Any]
    size: Tuple[int, int]
    embedding: Optional[Any] = None


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


def compute_phash_variants(image_path: Path, hash_size: int = 8) -> List[str]:
    """Return a list of pHash values including simple geometric variants."""
    if not PIL_AVAILABLE or imagehash is None:
        return [compute_phash(image_path, hash_size=hash_size)]
    variants: List[str] = []
    with Image.open(str(image_path)) as img:
        base = img.convert("RGB")
        transforms = [
            base,
            base.rotate(90, expand=True),
            base.rotate(180, expand=True),
            base.rotate(270, expand=True),
            base.transpose(Image.FLIP_LEFT_RIGHT),
            base.transpose(Image.FLIP_TOP_BOTTOM),
        ]
        for im in transforms:
            variants.append(imagehash.phash(im, hash_size=hash_size).__str__())
    # deduplicate while preserving order
    seen: List[str] = []
    for v in variants:
        if v not in seen:
            seen.append(v)
    return seen


def _load_embedder():
    global _EMBED_MODEL, _EMBED_TRANSFORM
    if not TORCH_AVAILABLE:
        return None, None
    if _EMBED_MODEL is not None and _EMBED_TRANSFORM is not None:
        return _EMBED_MODEL, _EMBED_TRANSFORM
    try:
        weights = None
        try:
            weights = models.ResNet18_Weights.DEFAULT  # type: ignore[attr-defined]
        except Exception:
            weights = None
        if weights is not None:
            model = models.resnet18(weights=weights)
            transform = weights.transforms()
        else:
            model = models.resnet18(pretrained=True)
            from torchvision import transforms

            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        model.fc = torch.nn.Identity()
        model.eval()
        model.to("cpu")
        _EMBED_MODEL = model
        _EMBED_TRANSFORM = transform
    except Exception:
        _EMBED_MODEL = None
        _EMBED_TRANSFORM = None
    return _EMBED_MODEL, _EMBED_TRANSFORM


def _fallback_embedding(image_path: Path, size: int = 64) -> Optional[Any]:
    if np is None or not PIL_AVAILABLE or Image is None:
        return None
    try:
        img = Image.open(str(image_path)).convert("RGB")
        img = img.resize((size, size))
        arr = np.asarray(img, dtype=np.float32)
        if arr.size == 0:
            return None
        arr = arr / 255.0
        emb = arr.reshape(-1)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception:
        return None


def compute_embedding(image_path: Path) -> Optional[Any]:
    """Compute a ResNet18 embedding (falls back to RGB thumbnail)."""
    model, transform = _load_embedder()
    if model is not None and transform is not None and Image is not None:
        try:
            img = Image.open(str(image_path)).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                vec = model(tensor.to("cpu")).squeeze(0).numpy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.astype("float32")
        except Exception:
            pass
    return _fallback_embedding(image_path)


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
    embedding = None
    if PIL_AVAILABLE and Image is not None:
        try:
            img = Image.open(str(image_path))
            size = img.size
        except Exception:
            size = (0, 0)
    try:
        embedding = compute_embedding(image_path)
    except Exception:
        embedding = None
    feats = ImageFeatures(phash=ph, orb=orb, size=size, embedding=embedding)
    # attach source path for downstream tile recall
    # 将源路径附加到特征对象，以便后续 tile 召回使用
    try:
        feats._path = str(image_path)
    except Exception:
        pass
    feats.embedding = embedding
    return feats
