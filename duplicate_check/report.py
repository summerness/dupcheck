"""Reporting utilities: CSV output and evidence image generation (stub).

报告模块：生成 CSV 报表并创建证据图（并排显示、可绘制匹配连线）。
"""
import csv
from pathlib import Path
from typing import List, Dict
from shutil import copyfile

try:
    import cv2
except Exception:
    cv2 = None


CSV_FIELDS = [
    "new_image",
    "matched_image",
    "final_label",
    "score",
    "inliers",
    "inlier_ratio",
    "ncc_peak",
    "evidence_img_path",
]


def write_csv(rows: List[Dict], out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def make_evidence_image(new_img_path: Path, db_img_path: Path, out_path: Path, draw_matches: bool = False, matches=None):
    """Create a side-by-side evidence image. If cv2 and matches provided, draw matches."""
    if cv2 is None:
        # fallback: copy new image
        # 若未安装 OpenCV，则回退为直接复制新图作为证据图
        try:
            copyfile(str(new_img_path), str(out_path))
        except Exception:
            pass
        return

    na = cv2.imread(str(new_img_path))
    db = cv2.imread(str(db_img_path))
    if na is None or db is None:
        try:
            copyfile(str(new_img_path), str(out_path))
        except Exception:
            pass
        return

    # Resize to same height
    h = max(na.shape[0], db.shape[0])
    def resize_keep(asrc, height):
        h0, w0 = asrc.shape[:2]
        scale = height / h0
        return cv2.resize(asrc, (int(w0 * scale), height))

    na_r = resize_keep(na, h)
    db_r = resize_keep(db, h)

    if draw_matches and matches:
        # matches: list of ((xq,yq),(xd,yd)) pairs
        # build a canvas that is na_r + db_r side-by-side and draw lines
        concat = cv2.hconcat([na_r, db_r])
        wq = na_r.shape[1]
        # compute scale factors from original images to resized ones
        hq_orig = na.shape[0]
        wq_orig = na.shape[1]
        hd_orig = db.shape[0]
        wd_orig = db.shape[1]
        h_res = h
        na_scale_x = na_r.shape[1] / max(1, wq_orig)
        na_scale_y = na_r.shape[0] / max(1, hq_orig)
        db_scale_x = db_r.shape[1] / max(1, wd_orig)
        db_scale_y = db_r.shape[0] / max(1, hd_orig)
        for (xq, yq), (xd, yd) in matches:
            pt1 = (int(xq * na_scale_x), int(yq * na_scale_y))
            pt2 = (int(wq + xd * db_scale_x), int(yd * db_scale_y))
            cv2.line(concat, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(concat, pt1, 3, (0, 0, 255), -1)
            cv2.circle(concat, pt2, 3, (0, 0, 255), -1)
        try:
            cv2.imwrite(str(out_path), concat)
        except Exception:
            try:
                copyfile(str(new_img_path), str(out_path))
            except Exception:
                pass
        return


    concat = cv2.hconcat([na_r, db_r])
    try:
        cv2.imwrite(str(out_path), concat)
    except Exception:
        try:
            copyfile(str(new_img_path), str(out_path))
        except Exception:
            pass
