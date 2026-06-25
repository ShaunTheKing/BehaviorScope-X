#!/usr/bin/env python3
"""Audit held-out MARS annotation provenance for controlled comparisons."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from controlled_common import add_shared_scripts_to_path, load_config, output_root, resolve_repo_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Separate accepted human-GT annotations from pred-named/uncertain held-out annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    p.add_argument("--splits", nargs="+", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    add_shared_scripts_to_path(cfg)
    from batch_infer_eval import find_annot, find_seq  # noqa: WPS433

    mars_root = resolve_repo_path(cfg["paths"]["mars_data_root"])
    splits = args.splits or list(cfg["evaluation"]["splits"])
    rows = []
    for split in splits:
        split_dir = mars_root / split
        if not split_dir.is_dir():
            rows.append({"split": split, "video_id": "", "status": "missing_split_dir", "accepted_annot": "", "pred_named_annot": "", "all_annots": ""})
            continue
        for video_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            all_annots = sorted(video_dir.glob("*.annot"))
            accepted = find_annot(video_dir, allow_pred_named=False)
            pred_allowed = find_annot(video_dir, allow_pred_named=True)
            pred_named = pred_allowed if accepted is None and pred_allowed is not None else None
            status = "human_gt_accepted" if accepted is not None else "excluded_pred_named_or_missing"
            if find_seq(video_dir) is None:
                status = f"{status};missing_seq"
            rows.append({
                "split": split,
                "video_id": video_dir.name,
                "status": status,
                "accepted_annot": str(accepted or ""),
                "pred_named_or_uncertain_annot": str(pred_named or ""),
                "all_annots": ";".join(p.name for p in all_annots),
            })

    out_path = output_root(cfg) / "audits" / "mars_annotation_provenance.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["split"])
        writer.writeheader()
        writer.writerows(rows)
    n_human = sum(1 for r in rows if str(r.get("status", "")).startswith("human_gt_accepted"))
    n_excluded = len(rows) - n_human
    print(f"[audit] rows={len(rows)} human_gt={n_human} excluded_or_uncertain={n_excluded} out={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
