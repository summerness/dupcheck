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
    import clip
    CLIP_AVAILABLE = True
except Exception:
    clip = None
    CLIP_AVAILABLE = False

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
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_MODEL = None
_CLIP_PREPROCESS = None

MULTISCALE_LEVELS: Tuple[float, ...] = (1.0, 0.75, 0.5)

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


def compute_phash_variants(
    image_path: Path,
    hash_size: int = 8,
    scales: Tuple[float, ...] = MULTISCALE_LEVELS,
) -> List[str]:
    """Return a list of pHash values with multi-scale + orientation variants."""
    if not PIL_AVAILABLE or imagehash is None:
        return [compute_phash(image_path, hash_size=hash_size)]
    variants: List[str] = []
    with Image.open(str(image_path)) as img:
        base = img.convert("RGB")
        transforms: List[Image.Image] = []
        for scale in scales:
            if scale <= 0:
                continue
            if scale == 1.0:
                scaled = base
            else:
                w = max(1, int(base.width * scale))
                h = max(1, int(base.height * scale))
                scaled = base.resize((w, h))
            transforms.extend(
                [
                    scaled,
                    scaled.rotate(90, expand=True),
                    scaled.rotate(180, expand=True),
                    scaled.rotate(270, expand=True),
                    scaled.transpose(Image.FLIP_LEFT_RIGHT),
                    scaled.transpose(Image.FLIP_TOP_BOTTOM),
                ]
            )
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


def _load_clip_model():
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if not CLIP_AVAILABLE or not TORCH_AVAILABLE:
        return None, None
    if _CLIP_MODEL is not None and _CLIP_PREPROCESS is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS
    try:
        device = "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        _CLIP_MODEL = model
        _CLIP_PREPROCESS = preprocess
    except Exception:
        _CLIP_MODEL = None
        _CLIP_PREPROCESS = None
    return _CLIP_MODEL, _CLIP_PREPROCESS


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
    """Compute fused embeddings (ResNet18 + optional CLIP) for ANN recall."""
    if np is None:
        return _fallback_embedding(image_path)
    embeddings: List[np.ndarray] = []

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
            embeddings.append(vec.astype("float32"))
        except Exception:
            pass

    clip_model, clip_preprocess = _load_clip_model()
    if clip_model is not None and clip_preprocess is not None:
        try:
            img = Image.open(str(image_path)).convert("RGB")
            tensor = clip_preprocess(img).unsqueeze(0)
            with torch.no_grad():
                vec = clip_model.encode_image(tensor.to("cpu")).squeeze(0).cpu().numpy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.astype("float32"))
        except Exception:
            pass

    if embeddings:
        try:
            fused = np.concatenate(embeddings)
            norm = np.linalg.norm(fused)
            if norm > 0:
                fused = fused / norm
            return fused.astype("float32")
        except Exception:
            pass

    return _fallback_embedding(image_path)


def compute_tile_hashes(
    image_path: Path,
    grid: int = 8,
    hash_size: int = 8,
    scales: Tuple[float, ...] = MULTISCALE_LEVELS,
) -> List[Dict[str, Any]]:
    """Split image into grid x grid tiles across multiple scales and compute pHash per tile."""
    tiles: List[Dict[str, Any]] = []
    if not PIL_AVAILABLE or Image is None or imagehash is None:
        ph = compute_phash(image_path, hash_size=hash_size)
        tiles.append({"hash": ph, "bbox": (0, 0, 0, 0), "scale": 1.0})
        return tiles

    base = Image.open(str(image_path)).convert("RGB")
    w_base, h_base = base.size
    for scale in scales:
        if scale <= 0:
            continue
        if scale == 1.0:
            img = base
            w, h = w_base, h_base
        else:
            w = max(1, int(w_base * scale))
            h = max(1, int(h_base * scale))
            img = base.resize((w, h))
        if w == 0 or h == 0:
            continue

        tile_w = max(1, w // grid)
        tile_h = max(1, h // grid)
        for yi in range(grid):
            for xi in range(grid):
                x0 = xi * tile_w
                y0 = yi * tile_h
                x1 = x0 + tile_w if xi < grid - 1 else w
                y1 = y0 + tile_h if yi < grid - 1 else h
                crop = img.crop((x0, y0, x1, y1))
                ph = imagehash.phash(crop, hash_size=hash_size)
                inv = 1.0 / scale if scale != 0 else 1.0
                bbox = (
                    int(x0 * inv),
                    int(y0 * inv),
                    int(x1 * inv),
                    int(y1 * inv),
                )
                tiles.append({
                    "hash": ph.__str__(),
                    "bbox": bbox,
                    "scale": float(scale),
                })
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
