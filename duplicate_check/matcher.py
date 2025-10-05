"""Matcher: recall via phash and tile-hash, precise verification via ORB+RANSAC and NCC.

Matcher 模块：通过 phash 和 tile-hash 召回候选，使用 ORB+RANSAC 与 NCC 做精排与判定。
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple
from duplicate_check.features import ImageFeatures
import hashlib

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None


def hamming_distance_hex(a: str, b: str) -> int:
    # imagehash returns hex string; convert to int
    # imagehash 返回十六进制字符串；将其转换为整数并计算汉明距离
    ai = int(a, 16)
    bi = int(b, 16)
    x = ai ^ bi
    return x.bit_count()


def recall_candidates(features: ImageFeatures, index: Dict, topk: int = 50, phash_thresh: int = 10, tile_match_count: int = 3) -> List[Dict[str, Any]]:
    """Recall candidates by global pHash and tile-hash. Returns list of dicts with scores.

    通过全局 pHash 和块哈希召回候选，返回包含分数的字典列表。
    """
    ph = features.phash
    hits = {}
    # global phash exact-ish match
    for phash_key, ids in index.get("by_phash", {}).items():
        d = hamming_distance_hex(ph, phash_key)
        if d <= phash_thresh:
            for i in ids:
                hits.setdefault(i, {"score": 0.0, "reason": []})
                hits[i]["score"] += max(0, (phash_thresh - d) / phash_thresh)
                hits[i]["reason"].append(("phash", d))

    # tile recall: compute query tile hashes (if possible) and count matches
    try:
        from duplicate_check.features import compute_tile_hashes
        if hasattr(features, "_path") and features._path:
            q_tiles = compute_tile_hashes(Path(features._path), grid=8)
        else:
            q_tiles = None
    except Exception:
        q_tiles = None

    # 如果能够计算 query 的 tile-hash，则进行基于块的召回
    if q_tiles:
        tile_counts = {}
        for th, bbox in q_tiles:
            for (img_id, tbbox) in index.get("by_tile", {}).get(th, []):
                tile_counts.setdefault(img_id, 0)
                tile_counts[img_id] += 1
        # add tile-derived scores
        for img_id, cnt in tile_counts.items():
            hits.setdefault(img_id, {"score": 0.0, "reason": []})
            hits[img_id]["score"] += cnt / (len(q_tiles) or 1)
            hits[img_id]["reason"].append(("tiles", cnt))

    # Convert hits to sorted list
    # 将命中结果转换为按分数排序的列表
    out = []
    for img_id, v in hits.items():
        out.append({"db_id": img_id, "score": v["score"], "reason": v["reason"]})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:topk]


def _orb_ransac_inliers(kps1, desc1, kps2, desc2, ratio=0.75, ransac_thresh=5.0):
    """Match descriptors using BFMatcher and compute RANSAC homography inliers.

    Returns (inlier_count, inlier_ratio, matches_mask, H)
    """
    if cv2 is None or np is None or desc1 is None or desc2 is None:
        return 0, 0.0, None, None, []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 4:
        return 0, 0.0, None, None, good
    src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return 0, 0.0, None, H, good
    inliers = int(mask.sum())
    inlier_ratio = inliers / max(1, len(good))
    return inliers, inlier_ratio, mask.ravel().tolist(), H, good


def _ncc_peak(img_query_path: Path, db_path: Path, bbox: Tuple[int, int, int, int]) -> float:
    """Compute normalized cross correlation between query and db patch.

    For simplicity, we load images via OpenCV, extract the db bbox, resize query to same
    and compute cv2.matchTemplate with TM_CCOEFF_NORMED.
    """
    if cv2 is None or np is None:
        return 0.0
    q = cv2.imread(str(img_query_path), cv2.IMREAD_COLOR)
    d = cv2.imread(str(db_path), cv2.IMREAD_COLOR)
    if q is None or d is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    patch = d[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    # Resize query to patch size
    q_resized = cv2.resize(q, (patch.shape[1], patch.shape[0]))
    qf = cv2.cvtColor(q_resized, cv2.COLOR_BGR2GRAY)
    pf = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(pf, qf, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) if res.size else 0.0


def rerank_and_verify(input_path: Path, candidates: List[Dict[str, Any]], index: Dict, orb_inliers_thresh: int = 25, orb_inlier_ratio: float = 0.25, ncc_thresh: float = 0.92) -> List[Dict[str, Any]]:
    """For each candidate, run ORB matching + RANSAC and NCC to generate final decision rows."""
    rows = []
    # precompute query descriptors
    try:
        from duplicate_check.features import compute_orb_descriptors
        q_orb = compute_orb_descriptors(input_path)
    except Exception:
        q_orb = {"kps": [], "descs": None}

    for c in candidates:
        db_id = c["db_id"]
        db_rec = index["by_id"].get(db_id)
        if db_rec is None:
            continue
        db_path = Path(db_rec["path"])
        # attempt to load db descriptors on the fly
        try:
            d_orb = compute_orb_descriptors(db_path)
        except Exception:
            d_orb = {"kps": [], "descs": None}

        inliers, inlier_ratio, mask, H, good_matches = _orb_ransac_inliers(q_orb.get("kps"), q_orb.get("descs"), d_orb.get("kps"), d_orb.get("descs"))

        label = "unique"
        score = c.get("score", 0.0)
        ncc_peak = 0.0
        evidence = ""

        if inliers >= orb_inliers_thresh and inlier_ratio >= orb_inlier_ratio:
            # candidate partial duplicate; attempt NCC using bbox from db tiles if available
            label = "partial_duplicate"
            score = max(score, min(0.99, 0.5 + inlier_ratio))
            # try NCC on first tile if available
            tiles = db_rec.get("tiles", [])
            if tiles:
                # pick center tile bbox
                _, bbox = tiles[len(tiles) // 2]
                try:
                    ncc_peak = _ncc_peak(input_path, db_path, bbox)
                except Exception:
                    ncc_peak = 0.0
                if ncc_peak >= ncc_thresh:
                    label = "exact_patch"
                    score = 0.99

        # build match_pairs for visualization (list of ((x1,y1),(x2,y2)))
        match_pairs = []
        try:
            if good_matches and q_orb.get("kps") and d_orb.get("kps"):
                for m in good_matches:
                    pt_q = q_orb.get("kps")[m.queryIdx].pt
                    pt_d = d_orb.get("kps")[m.trainIdx].pt
                    match_pairs.append(((float(pt_q[0]), float(pt_q[1])), (float(pt_d[0]), float(pt_d[1]))))
        except Exception:
            match_pairs = []

        rows.append({
            "new_image": str(input_path.name),
            "matched_image": db_id,
            "final_label": label,
            "score": float(score),
            "inliers": int(inliers),
            "inlier_ratio": float(inlier_ratio),
            "ncc_peak": float(ncc_peak),
            "evidence_img_path": evidence,
            "match_pairs": match_pairs,
        })

    return rows

