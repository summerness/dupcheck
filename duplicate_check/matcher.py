"""Matcher: recall via phash and tile-hash, precise verification via ORB+RANSAC and NCC.

Matcher 模块：通过 phash 和 tile-hash 召回候选，使用 ORB+RANSAC 与 NCC 做精排与判定。
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
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

try:
    import faiss
except Exception:
    faiss = None


_DB_ORB_VARIANT_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def hamming_distance_hex(a: str, b: str) -> int:
    # imagehash returns hex string; convert to int
    # imagehash 返回十六进制字符串；将其转换为整数并计算汉明距离
    ai = int(a, 16)
    bi = int(b, 16)
    x = ai ^ bi
    return x.bit_count()


def _has_descriptors(variant: Dict[str, Any]) -> bool:
    desc = variant.get("descs")
    try:
        return desc is not None and len(desc) > 0
    except Exception:
        return False


def _count_good_matches(desc1, desc2, ratio: float = 0.75) -> int:
    if cv2 is None or np is None:
        return 0
    if desc1 is None or desc2 is None:
        return 0
    try:
        if len(desc1) == 0 or len(desc2) == 0:
            return 0
    except Exception:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = 0
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good += 1
    return good


def _compute_orb_variants_for_path(path: Path, cache: Dict[str, List[Dict[str, Any]]] | None = None, max_features: int = 2000) -> List[Dict[str, Any]]:
    key = str(path)
    if cache is not None and key in cache:
        return cache[key]

    variants: List[Dict[str, Any]] = []
    if cv2 is None:
        variants.append({"name": "rot0", "kps": [], "descs": None})
    else:
        img = cv2.imread(str(path))
        if img is None:
            variants.append({"name": "rot0", "kps": [], "descs": None})
        else:
            orb = cv2.ORB_create(nfeatures=max_features)
            transforms = [
                ("rot0", img),
                ("rot90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
                ("rot180", cv2.rotate(img, cv2.ROTATE_180)),
                ("rot270", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ]
            flip = cv2.flip(img, 1)
            transforms.extend([
                ("flip0", flip),
                ("flip90", cv2.rotate(flip, cv2.ROTATE_90_CLOCKWISE)),
            ])

            seen = set()
            for name, mat in transforms:
                if mat is None or name in seen:
                    continue
                seen.add(name)
                try:
                    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
                except Exception:
                    variants.append({"name": name, "kps": [], "descs": None})
                    continue
                kps, descs = orb.detectAndCompute(gray, None)
                variants.append({"name": name, "kps": kps or [], "descs": descs})

    if cache is not None:
        cache[key] = variants
    return variants


def _get_db_orb_variants(path: Path) -> List[Dict[str, Any]]:
    return _compute_orb_variants_for_path(path, _DB_ORB_VARIANT_CACHE)


def _best_orb_match(q_variants: List[Dict[str, Any]], db_variants: List[Dict[str, Any]]) -> Tuple[int, int, Optional[Tuple[str, str]]]:
    best_good = 0
    best_len = 1
    best_pair: Optional[Tuple[str, str]] = None
    for q_var in q_variants:
        if not _has_descriptors(q_var):
            continue
        q_desc = q_var.get("descs")
        q_len = len(q_desc)
        for d_var in db_variants:
            if not _has_descriptors(d_var):
                continue
            good = _count_good_matches(q_desc, d_var.get("descs"))
            if good > best_good:
                best_good = good
                best_len = max(1, q_len)
                best_pair = (q_var.get("name"), d_var.get("name"))
    return best_good, best_len, best_pair

def recall_candidates(features: ImageFeatures, index: Dict, topk: int = 50, phash_thresh: int = 10, tile_match_count: int = 3) -> List[Dict[str, Any]]:
    """Recall candidates by global pHash and tile-hash. Returns list of dicts with scores.

    通过全局 pHash 和块哈希召回候选，返回包含分数的字典列表。
    """
    ph = features.phash
    hits: Dict[str, Dict[str, Any]] = {}
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

    # Vector-based recall via FAISS (optional)
    vector_index = index.get("vector") if isinstance(index, dict) else None
    if vector_index and np is not None and faiss is not None:
        q_emb = getattr(features, "embedding", None)
        if q_emb is None and hasattr(features, "_path"):
            try:
                from duplicate_check.features import compute_embedding

                q_emb = compute_embedding(Path(features._path))
            except Exception:
                q_emb = None
        try:
            if q_emb is not None:
                vec = np.asarray(q_emb, dtype=np.float32)
                if vec.ndim == 1 and vec.size > 0:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    vec = vec.reshape(1, -1)
                    index_obj = vector_index.get("index")
                    ids = vector_index.get("ids", [])
                    metric = vector_index.get("metric", "ip")
                    if index_obj is not None and len(ids):
                        topn = min(max(topk, 10), len(ids))
                        D, I = index_obj.search(vec, topn)
                        for dist, idx_id in zip(D[0], I[0]):
                            if idx_id < 0 or idx_id >= len(ids):
                                continue
                            db_id = ids[idx_id]
                            if metric == "ip":
                                score = float(dist)
                            else:
                                score = float(1.0 / (1.0 + dist))
                            if score <= 0:
                                continue
                            entry = hits.setdefault(db_id, {"score": 0.0, "reason": []})
                            entry["score"] += score
                            entry.setdefault("reason", []).append(("vector", score))
        except Exception:
            pass

    # Orientation-aware ORB scoring
    query_path = None
    if hasattr(features, "_path") and features._path:
        try:
            query_path = Path(features._path)
        except Exception:
            query_path = None

    q_variants: List[Dict[str, Any]] = []
    if query_path is not None:
        try:
            q_variants = _compute_orb_variants_for_path(query_path)
        except Exception:
            q_variants = []

    has_query_orb = any(_has_descriptors(v) for v in q_variants)

    if has_query_orb:
        for img_id in list(hits.keys()):
            rec = index.get("by_id", {}).get(img_id)
            if rec is None:
                continue
            db_variants = _get_db_orb_variants(Path(rec["path"]))
            best_good, best_len, best_pair = _best_orb_match(q_variants, db_variants)
            if best_good <= 0 or best_pair is None:
                continue
            entry = hits.setdefault(img_id, {"score": 0.0, "reason": []})
            entry["score"] += min(1.0, best_good / max(1, best_len))
            entry.setdefault("reason", []).append(("orb", best_good))
            entry["best_orient"] = best_pair

        # Fallback: add strong ORB matches not yet recalled
        if len(hits) < topk:
            ORB_FALLBACK_MIN = 25
            for img_id, rec in index.get("by_id", {}).items():
                if img_id in hits:
                    continue
                db_variants = _get_db_orb_variants(Path(rec["path"]))
                best_good, best_len, best_pair = _best_orb_match(q_variants, db_variants)
                if best_good < ORB_FALLBACK_MIN or best_pair is None:
                    continue
                entry = hits.setdefault(img_id, {"score": 0.0, "reason": []})
                entry["score"] += min(1.0, best_good / max(1, best_len))
                entry.setdefault("reason", []).append(("orb", best_good))
                entry["best_orient"] = best_pair
                if len(hits) >= topk:
                    break

    # Convert hits to sorted list
    out = []
    for img_id, v in hits.items():
        out.append(
            {
                "db_id": img_id,
                "score": v.get("score", 0.0),
                "reason": v.get("reason", []),
                "orientation": v.get("best_orient"),
            }
        )
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


def rerank_and_verify(
    input_path: Path,
    candidates: List[Dict[str, Any]],
    index: Dict,
    orb_inliers_thresh: int = 25,
    orb_inlier_ratio: float = 0.25,
    ncc_thresh: float = 0.92,
) -> List[Dict[str, Any]]:
    """For each candidate, run ORB matching + RANSAC and NCC to generate final decision rows."""
    rows: List[Dict[str, Any]] = []

    try:
        q_variants = _compute_orb_variants_for_path(input_path)
    except Exception:
        q_variants = []
    q_map = {var.get("name"): var for var in q_variants}
    has_query_orb = any(_has_descriptors(v) for v in q_variants)

    for c in candidates:
        db_id = c.get("db_id")
        db_rec = index.get("by_id", {}).get(db_id) if db_id else None
        if db_rec is None:
            continue
        db_path = Path(db_rec["path"])
        db_variants = _get_db_orb_variants(db_path)
        d_map = {var.get("name"): var for var in db_variants}
        has_db_orb = any(_has_descriptors(v) for v in db_variants)

        orientation_hint = c.get("orientation")
        pair_order: List[Tuple[str, str]] = []
        if orientation_hint and isinstance(orientation_hint, (tuple, list)) and len(orientation_hint) == 2:
            q_name, d_name = orientation_hint
            if q_name in q_map and d_name in d_map:
                pair_order.append((q_name, d_name))

        for q_var in q_variants:
            for d_var in db_variants:
                pair = (q_var.get("name"), d_var.get("name"))
                if pair not in pair_order:
                    pair_order.append(pair)

        best = None
        for q_name, d_name in pair_order:
            q_var = q_map.get(q_name)
            d_var = d_map.get(d_name)
            if not q_var or not d_var:
                continue
            if not _has_descriptors(q_var) or not _has_descriptors(d_var):
                continue
            inliers, inlier_ratio, mask, H, good_matches = _orb_ransac_inliers(
                q_var["kps"],
                q_var["descs"],
                d_var["kps"],
                d_var["descs"],
            )
            if best is None or inliers > best["inliers"]:
                best = {
                    "q": q_var,
                    "d": d_var,
                    "q_name": q_name,
                    "d_name": d_name,
                    "inliers": inliers,
                    "inlier_ratio": inlier_ratio,
                    "matches": good_matches,
                }

        has_descriptors = has_query_orb and has_db_orb

        if best is None:
            if not has_descriptors:
                reasons = {r[0] for r in c.get("reason", [])}
                if "phash" in reasons:
                    rows.append(
                        {
                            "new_image": str(input_path.name),
                            "matched_image": db_id,
                            "final_label": "phash_duplicate",
                            "score": float(max(c.get("score", 0.5), 0.5)),
                            "inliers": 0,
                            "inlier_ratio": 0.0,
                            "ncc_peak": 0.0,
                            "evidence_img_path": "",
                            "match_pairs": [],
                            "orientation": "",
                        }
                    )
            continue

        label = "unique"
        score = c.get("score", 0.0)
        ncc_peak = 0.0
        evidence = ""

        if (
            best["inliers"] >= orb_inliers_thresh
            and best["inlier_ratio"] >= orb_inlier_ratio
        ):
            label = "partial_duplicate"
            score = max(score, min(0.99, 0.5 + best["inlier_ratio"]))
            tiles = db_rec.get("tiles", [])
            if (
                tiles
                and best["q_name"] == "rot0"
                and best["d_name"] == "rot0"
            ):
                _, bbox = tiles[len(tiles) // 2]
                try:
                    ncc_peak = _ncc_peak(input_path, db_path, bbox)
                except Exception:
                    ncc_peak = 0.0
                if ncc_peak >= ncc_thresh:
                    label = "exact_patch"
                    score = 0.99
        else:
            continue

        match_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        try:
            matches = best.get("matches") or []
            q_kps = best["q"]["kps"]
            d_kps = best["d"]["kps"]
            if matches and q_kps and d_kps:
                for m in matches:
                    pt_q = q_kps[m.queryIdx].pt
                    pt_d = d_kps[m.trainIdx].pt
                    match_pairs.append(((float(pt_q[0]), float(pt_q[1])), (float(pt_d[0]), float(pt_d[1]))))
        except Exception:
            match_pairs = []

        rows.append(
            {
                "new_image": str(input_path.name),
                "matched_image": db_id,
                "final_label": label,
                "score": float(score),
                "inliers": int(best["inliers"]),
                "inlier_ratio": float(best["inlier_ratio"]),
                "ncc_peak": float(ncc_peak),
                "evidence_img_path": evidence,
                "match_pairs": match_pairs,
                "orientation": f"{best['q_name']}->{best['d_name']}",
            }
        )

    return rows
