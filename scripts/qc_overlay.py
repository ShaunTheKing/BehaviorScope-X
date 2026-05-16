#!/usr/bin/env python3
"""
qc_overlay.py
=============
Visual sanity-check for the YOLO-pose dataset built by mars_to_yolo_pose.py.

For each sampled image, draws:
  * the bounding box for each mouse (white)
  * the class label in the corner of each bbox
  * the seven top-view keypoints, color-coded by body part
  * keypoint dot size encodes visibility (v=2 large, v=1 small, v=0 not drawn)
  * the image basename in the top-left corner of the frame

Outputs:
  <out>/individual/   one annotated JPG per sampled frame
  <out>/grid.jpg      composite of all sampled frames (auto-sized rows x cols)
  <out>/qc_summary.csv  one row per sample with class, split, mouse rows,
                        per-mouse visible-keypoint count

Usage (Windows, IntegraPose conda env):
    python qc_overlay.py ^
        --dataset-root path\\to\\mars_yolo_pose ^
        --num 24 ^
        --seed 0
"""

import argparse
import csv
import glob
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Match the order used in mars_to_yolo_pose.py.  Class 3 'other' is present
# only when the dataset was built with the (default) 4-class layout; for
# legacy 3-class datasets the id-3 entries simply never appear.
CLASS_NAMES = ["investigation", "mount", "attack", "other"]
KPT_NAMES = ["nose", "ear_1", "ear_2", "neck", "hip_1", "hip_2", "tail_base"]
KPT_COLORS = [
    (255, 0, 0),     # nose       - red
    (0, 255, 0),     # ear_1      - green
    (0, 0, 255),     # ear_2      - blue
    (255, 255, 0),   # neck       - yellow
    (255, 0, 255),   # hip_1      - magenta
    (0, 255, 255),   # hip_2      - cyan
    (255, 128, 0),   # tail_base  - orange
]
CLASS_COLORS = {
    0: (0, 200, 255),    # investigation - light blue
    1: (0, 255, 100),    # mount         - green
    2: (255, 80, 80),    # attack        - red
    3: (200, 200, 200),  # other         - gray
}


def _load_label(path):
    """Return list of dicts: {class_id, bbox(xc,yc,w,h normalized), kpts list[(xn,yn,v)]}."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 26:
                continue
            cls = int(parts[0])
            xc, yc, w, h = (float(x) for x in parts[1:5])
            kpts = []
            for k in range(7):
                xn = float(parts[5 + 3 * k])
                yn = float(parts[5 + 3 * k + 1])
                v = int(parts[5 + 3 * k + 2])
                kpts.append((xn, yn, v))
            rows.append({"class": cls, "bbox": (xc, yc, w, h), "kpts": kpts})
    return rows


def _font_or_default(size):
    """Try to load a TrueType font; fall back to PIL's default bitmap font."""
    windir = os.environ.get("WINDIR")
    candidates = [
        str(Path(windir) / "Fonts" / "arial.ttf") if windir else "",
        str(Path(windir) / "Fonts" / "segoeui.ttf") if windir else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate_image(img_path, lbl_path, font_lg, font_sm):
    """Open image, draw bboxes / keypoints / class labels, return RGB PIL Image
    along with summary stats."""
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    rows = _load_label(lbl_path)

    n_mouse = len(rows)
    visible_counts = []
    classes_present = set()

    for ridx, r in enumerate(rows):
        cls = r["class"]
        classes_present.add(cls)
        cls_color = CLASS_COLORS.get(cls, (255, 255, 255))
        xc, yc, w, h = r["bbox"]
        x1 = (xc - w / 2) * W
        y1 = (yc - h / 2) * H
        x2 = (xc + w / 2) * W
        y2 = (yc + h / 2) * H
        draw.rectangle([x1, y1, x2, y2], outline=cls_color, width=2)

        # Class tag with mouse-row index
        tag = f"{CLASS_NAMES[cls]} (m{ridx})"
        # text background
        try:
            tw, th = draw.textbbox((0, 0), tag, font=font_sm)[2:]
        except Exception:
            tw, th = font_sm.getsize(tag) if hasattr(font_sm, "getsize") else (len(tag) * 7, 12)
        pad = 2
        draw.rectangle([x1, y1 - th - 2 * pad, x1 + tw + 2 * pad, y1], fill=cls_color)
        draw.text((x1 + pad, y1 - th - pad), tag, fill=(0, 0, 0), font=font_sm)

        n_vis = 0
        for k, (xn, yn, v) in enumerate(r["kpts"]):
            if v == 0:
                continue
            n_vis += 1
            x = xn * W
            y = yn * H
            radius = 5 if v == 2 else 3
            color = KPT_COLORS[k]
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         fill=color, outline=(0, 0, 0))
        visible_counts.append(n_vis)

    # Image basename top-left
    base = os.path.basename(img_path)
    txt = base
    try:
        tw, th = draw.textbbox((0, 0), txt, font=font_sm)[2:]
    except Exception:
        tw, th = font_sm.getsize(txt) if hasattr(font_sm, "getsize") else (len(txt) * 7, 12)
    draw.rectangle([0, 0, tw + 8, th + 6], fill=(0, 0, 0))
    draw.text((4, 3), txt, fill=(255, 255, 255), font=font_sm)

    return im, {
        "n_mouse_rows": n_mouse,
        "visible_kpts_per_mouse": visible_counts,
        "classes_present": sorted(classes_present),
    }


def discover_samples(dataset_root, num, splits, seed, stratify):
    """Return list of dicts {image, label, split, class_id} sampled according
    to stratify flag."""
    rng = random.Random(seed)
    pool = []
    for split in splits:
        img_dir = Path(dataset_root) / "images" / split
        lbl_dir = Path(dataset_root) / "labels" / split
        if not img_dir.is_dir():
            continue
        for ip in sorted(img_dir.glob("*.jpg")):
            lp = lbl_dir / (ip.stem + ".txt")
            if not lp.is_file():
                continue
            # Read first-row class for stratification (fast)
            cls = None
            try:
                with open(lp, "r") as f:
                    first = f.readline().split()
                    if first:
                        cls = int(first[0])
            except Exception:
                continue
            pool.append({"image": str(ip), "label": str(lp), "split": split, "class_id": cls})

    if not pool:
        return []

    if not stratify:
        rng.shuffle(pool)
        return pool[:num]

    by_class = defaultdict(list)
    for item in pool:
        by_class[item["class_id"]].append(item)

    classes = sorted(by_class.keys())
    if not classes:
        return pool[:num]

    target_per_class = max(1, num // len(classes))
    picked = []
    for c in classes:
        rng.shuffle(by_class[c])
        picked.extend(by_class[c][:target_per_class])
    # Fill remainder if num > picked due to a small pool in some class
    if len(picked) < num:
        leftover = [it for c in classes for it in by_class[c][target_per_class:]]
        rng.shuffle(leftover)
        picked.extend(leftover[: num - len(picked)])
    rng.shuffle(picked)
    return picked[:num]


def make_grid(images, max_cols=4):
    """Composite a list of same-sized PIL images into one grid image."""
    if not images:
        return None
    W, H = images[0].size
    cols = min(max_cols, len(images))
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (W * cols, H * rows), (0, 0, 0))
    for i, im in enumerate(images):
        c = i % cols
        r = i // cols
        out.paste(im, (c * W, r * H))
    return out


def main():
    p = argparse.ArgumentParser(description="QC overlays for YOLO-pose dataset.")
    p.add_argument("--dataset-root", required=True,
                   help="Root of the YOLO-pose dataset (contains images/, labels/, data.yaml).")
    p.add_argument("--num", type=int, default=12, help="Total images to render (default 12).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="Output dir. Default: <dataset-root>/qc/")
    p.add_argument("--splits", nargs="+", default=["train", "val"],
                   help="Which splits to sample from (default both).")
    p.add_argument("--no-stratify", action="store_true",
                   help="Disable per-class stratification when picking samples.")
    p.add_argument("--no-grid", action="store_true",
                   help="Skip writing the composite grid image.")
    p.add_argument("--no-individual", action="store_true",
                   help="Skip writing individual annotated JPGs.")
    p.add_argument("--grid-cols", type=int, default=4)
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out) if args.out else dataset_root / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_individual:
        (out_dir / "individual").mkdir(parents=True, exist_ok=True)

    samples = discover_samples(
        str(dataset_root), args.num, args.splits,
        seed=args.seed, stratify=not args.no_stratify,
    )
    if not samples:
        print(f"ERROR: no labeled images found under {dataset_root}/images/{{train,val}}", file=sys.stderr)
        sys.exit(1)

    font_lg = _font_or_default(20)
    font_sm = _font_or_default(14)

    summary_rows = []
    grid_imgs = []
    for s in samples:
        im, info = annotate_image(s["image"], s["label"], font_lg, font_sm)
        # Save individual
        if not args.no_individual:
            base = os.path.basename(s["image"]).replace(".jpg", "")
            cls_name = CLASS_NAMES[s["class_id"]] if s["class_id"] is not None else "unk"
            out_name = f"{s['split']}__{cls_name}__{base}.jpg"
            im.save(out_dir / "individual" / out_name, quality=88)
        if not args.no_grid:
            grid_imgs.append(im)
        summary_rows.append({
            "split": s["split"],
            "class_id": s["class_id"],
            "class_name": CLASS_NAMES[s["class_id"]] if s["class_id"] is not None else "",
            "image": s["image"],
            "label": s["label"],
            "n_mouse_rows": info["n_mouse_rows"],
            "visible_kpts_per_mouse": "/".join(str(x) for x in info["visible_kpts_per_mouse"]),
            "classes_present_in_label": "/".join(str(c) for c in info["classes_present"]),
        })

    # Composite grid
    if not args.no_grid and grid_imgs:
        grid = make_grid(grid_imgs, max_cols=args.grid_cols)
        if grid is not None:
            grid.save(out_dir / "grid.jpg", quality=85)

    # CSV summary
    with open(out_dir / "qc_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "class_id", "class_name", "image", "label",
            "n_mouse_rows", "visible_kpts_per_mouse", "classes_present_in_label",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    # Console summary
    by_class = defaultdict(int)
    by_split = defaultdict(int)
    for r in summary_rows:
        by_class[r["class_name"]] += 1
        by_split[r["split"]] += 1
    print(f"Rendered {len(summary_rows)} images.")
    print(f"  by class: " + ", ".join(f"{k}={v}" for k, v in sorted(by_class.items())))
    print(f"  by split: " + ", ".join(f"{k}={v}" for k, v in sorted(by_split.items())))
    print(f"Output dir: {out_dir}")
    if not args.no_individual:
        print(f"  individual: {out_dir / 'individual'}")
    if not args.no_grid and grid_imgs:
        print(f"  grid:       {out_dir / 'grid.jpg'}")
    print(f"  csv:        {out_dir / 'qc_summary.csv'}")


if __name__ == "__main__":
    main()
