#!/usr/bin/env python
"""Bundle a BehaviorScope-Y classifier checkpoint and YOLO-pose weights.

The output is a single .pt file that can be passed to infer_x.py without a
separate --yolo_weights argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def safe_torch_load(path: Path | str, *, map_location=None):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a single-file BehaviorScope-Y .pt bundle.")
    p.add_argument("--classifier_checkpoint", required=True, help="best_model_macro_f1.pt or best_model.pt")
    p.add_argument("--yolo_weights", required=True, help="YOLO-pose .pt used during training")
    p.add_argument("--model_config", default=None, help="Optional config.json; otherwise uses embedded checkpoint config")
    p.add_argument("--output", required=True, help="Output bundled .pt path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    classifier_checkpoint = Path(args.classifier_checkpoint)
    yolo_weights = Path(args.yolo_weights)
    output = Path(args.output)
    if not classifier_checkpoint.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_checkpoint}")
    if not yolo_weights.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

    checkpoint = safe_torch_load(classifier_checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        checkpoint = {"model_state": checkpoint}
    else:
        checkpoint = dict(checkpoint)

    cfg = checkpoint.get("config") or checkpoint.get("model_config") or {}
    if args.model_config:
        with Path(args.model_config).open("r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
    cfg = dict(cfg)

    raw_yolo = yolo_weights.read_bytes()
    digest = hashlib.sha256(raw_yolo).hexdigest()
    cfg["yolo_weights"] = f"<embedded:{yolo_weights.name}>"
    cfg["yolo_weights_embedded"] = True

    checkpoint["config"] = cfg
    checkpoint["embedded_yolo_weights"] = {
        "format": "raw_file_bytes",
        "filename": yolo_weights.name,
        "sha256": digest,
        "num_bytes": len(raw_yolo),
        "bytes": raw_yolo,
    }
    checkpoint["bundle_metadata"] = {
        "version": "behaviorscope-y-single-model-v1",
        "classifier_checkpoint": classifier_checkpoint.name,
        "yolo_filename": yolo_weights.name,
        "yolo_sha256": digest,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"[single model] wrote {output}")
    print(f"[single model] embedded YOLO {yolo_weights.name} sha256={digest}")


if __name__ == "__main__":
    main()


