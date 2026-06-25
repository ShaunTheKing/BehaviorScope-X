"""Native MobileNetV3 pose inference for BehaviorScope window-cache builds."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from torchvision.models import mobilenet_v3_large
except Exception as exc:  # pragma: no cover
    raise ImportError("MobileNetV3 pose inference requires torchvision.") from exc

try:  # pragma: no cover - script/package dual use
    from cropping_x import (
        NCropDetection,
        compute_fixed_square_xyxy,
        crop_coverage_conf,
        crop_resize_pad_rgb,
    )
    from utils.pose_features_x import extract_relational_features
except Exception:  # pragma: no cover
    from .cropping_x import (
        NCropDetection,
        compute_fixed_square_xyxy,
        crop_coverage_conf,
        crop_resize_pad_rgb,
    )
    from .utils.pose_features_x import extract_relational_features


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
PARAM_ANCHOR_OFFSET_BOX_REL_KPT = "anchor_offset_box_relative_kpt"
PARAM_LOCAL_BOX_REL_KPT = "local_box_relative_kpt"
PARAM_ABSOLUTE = "absolute"


class ConvBNAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, groups: int = 1):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileNetV3LargeFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = mobilenet_v3_large(weights=None).features
        self.lat5 = ConvBNAct(960, 256, k=1)
        self.lat4 = ConvBNAct(112, 256, k=1)
        self.lat3 = ConvBNAct(40, 128, k=1)
        self.out4 = ConvBNAct(256, 256, k=3)
        self.out3 = ConvBNAct(128 + 256, 128, k=3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p3_raw = p4_raw = p5_raw = None
        y = x
        for i, layer in enumerate(self.features):
            y = layer(y)
            if i == 6:
                p3_raw = y
            elif i == 12:
                p4_raw = y
            elif i == 16:
                p5_raw = y
        if p3_raw is None or p4_raw is None or p5_raw is None:
            raise RuntimeError("MobileNetV3 feature tap extraction failed.")
        p5 = self.lat5(p5_raw)
        p4 = self.out4(self.lat4(p4_raw) + F.interpolate(p5, size=p4_raw.shape[-2:], mode="nearest"))
        p3 = self.out3(torch.cat([self.lat3(p3_raw), F.interpolate(p4, size=p3_raw.shape[-2:], mode="nearest")], dim=1))
        return {"p3": p3, "p4": p4, "p5": p5, "visual": p5}


class DecoupledPoseHeadV3(nn.Module):
    def __init__(self, num_classes: int, num_keypoints: int):
        super().__init__()
        self.stems = nn.ModuleDict()
        self.box_towers = nn.ModuleDict()
        self.cls_towers = nn.ModuleDict()
        self.kpt_towers = nn.ModuleDict()
        self.box_preds = nn.ModuleDict()
        self.obj_cls_preds = nn.ModuleDict()
        self.kpt_preds = nn.ModuleDict()
        for name, c_in in {"p3": 128, "p4": 256, "p5": 256}.items():
            c_mid = max(min(c_in, 256), 128)
            self.stems[name] = ConvBNAct(c_in, c_mid, k=1)
            self.box_towers[name] = nn.Sequential(ConvBNAct(c_mid, c_mid, k=3), ConvBNAct(c_mid, c_mid, k=3))
            self.cls_towers[name] = nn.Sequential(ConvBNAct(c_mid, c_mid, k=3), ConvBNAct(c_mid, c_mid, k=3))
            self.kpt_towers[name] = nn.Sequential(
                ConvBNAct(c_mid, c_mid, k=3),
                ConvBNAct(c_mid, c_mid, k=3),
                ConvBNAct(c_mid, c_mid, k=3),
            )
            self.box_preds[name] = nn.Conv2d(c_mid, 4, kernel_size=1)
            self.obj_cls_preds[name] = nn.Conv2d(c_mid, 1 + int(num_classes), kernel_size=1)
            self.kpt_preds[name] = nn.Conv2d(c_mid, int(num_keypoints) * 3, kernel_size=1)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for name in self.stems.keys():
            x = self.stems[name](features[name])
            out[name] = torch.cat(
                [
                    self.box_preds[name](self.box_towers[name](x)),
                    self.obj_cls_preds[name](self.cls_towers[name](x)),
                    self.kpt_preds[name](self.kpt_towers[name](x)),
                ],
                dim=1,
            )
        return out


class MobileNetV3PoseModel(nn.Module):
    def __init__(self, num_classes: int, num_keypoints: int):
        super().__init__()
        self.backbone = MobileNetV3LargeFPN()
        self.pose_head = DecoupledPoseHeadV3(num_classes=num_classes, num_keypoints=num_keypoints)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.pose_head(self.backbone(frames))


@dataclass(frozen=True)
class MobileNetV3PoseRuntime:
    model: MobileNetV3PoseModel
    image_size: int
    normalization: str
    num_keypoints: int
    num_classes: int
    parameterization: str
    config: dict


def load_mobilenetv3_pose_model(checkpoint: str | Path, device: str = "cpu") -> MobileNetV3PoseRuntime:
    path = Path(checkpoint)
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"{path} is not a BehaviorScope-X MobileNetV3 pose checkpoint.")
    cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    backbone_cfg = model_cfg.get("backbone", {}) if isinstance(model_cfg.get("backbone", {}), dict) else {}
    pose_head_cfg = model_cfg.get("pose_head", {}) if isinstance(model_cfg.get("pose_head", {}), dict) else {}
    backbone_name = str(backbone_cfg.get("name", ""))
    if backbone_name and backbone_name not in {"mobilenet_v3_large", "mobilenetv3_large", "mobilenetv3"}:
        raise ValueError(f"{path} uses backbone={backbone_name!r}; expected MobileNetV3-large.")
    num_keypoints = int(model_cfg.get("num_keypoints", 7))
    num_classes = int(model_cfg.get("num_pose_classes", 1))
    image_size = int(cfg.get("data", {}).get("image_size", 640))
    normalization = str(cfg.get("data", {}).get("normalization", "imagenet"))
    parameterization = str(pose_head_cfg.get("parameterization", PARAM_ANCHOR_OFFSET_BOX_REL_KPT))

    model = MobileNetV3PoseModel(num_classes=num_classes, num_keypoints=num_keypoints)
    state = ckpt["model"]
    pose_state = {k: v for k, v in state.items() if k.startswith(("backbone.", "pose_head."))}
    missing, unexpected = model.load_state_dict(pose_state, strict=False)
    allowed_missing_prefixes = ("visual_token_head.", "temporal_head.")
    unexpected_bad = [k for k in unexpected if not k.startswith(allowed_missing_prefixes)]
    if missing or unexpected_bad:
        raise RuntimeError(
            "MobileNetV3 pose checkpoint does not match the packaged decoder: "
            f"missing={missing}, unexpected={unexpected_bad}"
        )
    dev = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    model.to(dev).eval()
    for param in model.parameters():
        param.requires_grad = False
    return MobileNetV3PoseRuntime(model, image_size, normalization, num_keypoints, num_classes, parameterization, cfg)


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([(cx - w / 2).clamp(0, 1), (cy - h / 2).clamp(0, 1), (cx + w / 2).clamp(0, 1), (cy + h / 2).clamp(0, 1)], dim=-1)


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, max_detections: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if len(keep) >= int(max_detections) or order.numel() == 1:
            break
        ious = _box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= float(iou_threshold)]
    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


@torch.no_grad()
def decode_pose_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    num_keypoints: int,
    num_classes: int,
    conf_threshold: float,
    nms_iou: float,
    max_detections: int,
    pre_nms_topk: int,
    parameterization: str,
) -> list[dict[str, torch.Tensor]]:
    per_image: list[list[dict[str, torch.Tensor]]] | None = None
    for pred_bchw in outputs.values():
        pred = pred_bchw.permute(0, 2, 3, 1).contiguous()
        b, h, w = pred.shape[0], pred.shape[1], pred.shape[2]
        if per_image is None:
            per_image = [[] for _ in range(b)]
        flat = pred.reshape(b, -1, pred.shape[-1])
        box_raw = flat[..., 0:4].sigmoid()
        if parameterization == PARAM_ANCHOR_OFFSET_BOX_REL_KPT:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=pred.device, dtype=pred.dtype),
                torch.arange(w, device=pred.device, dtype=pred.dtype),
                indexing="ij",
            )
            cx = (box_raw[..., 0] * 2.0 - 0.5 + xx.reshape(1, -1)) / float(w)
            cy = (box_raw[..., 1] * 2.0 - 0.5 + yy.reshape(1, -1)) / float(h)
            boxes = torch.stack([cx, cy, box_raw[..., 2], box_raw[..., 3]], dim=-1).clamp(0.0, 1.0)
        elif parameterization == PARAM_LOCAL_BOX_REL_KPT:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=pred.device, dtype=pred.dtype),
                torch.arange(w, device=pred.device, dtype=pred.dtype),
                indexing="ij",
            )
            cx = (box_raw[..., 0] + xx.reshape(1, -1)) / float(w)
            cy = (box_raw[..., 1] + yy.reshape(1, -1)) / float(h)
            boxes = torch.stack([cx, cy, box_raw[..., 2], box_raw[..., 3]], dim=-1).clamp(0.0, 1.0)
        elif parameterization == PARAM_ABSOLUTE:
            boxes = box_raw
        else:
            raise ValueError(f"Unsupported pose parameterization: {parameterization}")
        obj = flat[..., 4].sigmoid()
        cls = flat[..., 5 : 5 + num_classes].sigmoid()
        cls_score, cls_id = cls.max(dim=-1)
        scores = obj * cls_score
        base = 5 + num_classes
        kpt_raw = flat[..., base : base + num_keypoints * 3].reshape(b, -1, num_keypoints, 3)
        kpt_xy = kpt_raw[..., 0:2].sigmoid()
        if parameterization in {PARAM_LOCAL_BOX_REL_KPT, PARAM_ANCHOR_OFFSET_BOX_REL_KPT}:
            x1 = (boxes[..., 0] - boxes[..., 2] / 2).unsqueeze(-1)
            y1 = (boxes[..., 1] - boxes[..., 3] / 2).unsqueeze(-1)
            bw = boxes[..., 2].unsqueeze(-1)
            bh = boxes[..., 3].unsqueeze(-1)
            keypoints = torch.stack([x1 + kpt_xy[..., 0] * bw, y1 + kpt_xy[..., 1] * bh], dim=-1).clamp(0.0, 1.0)
        else:
            keypoints = kpt_xy
        kpt_conf = kpt_raw[..., 2].sigmoid()
        for bi in range(b):
            mask = scores[bi] >= float(conf_threshold)
            if bool(mask.any()):
                idx = mask.nonzero(as_tuple=False).squeeze(1)
                if int(pre_nms_topk) > 0 and idx.numel() > int(pre_nms_topk):
                    idx = idx[scores[bi, idx].topk(int(pre_nms_topk)).indices]
                per_image[bi].append(
                    {
                        "boxes_xywh": boxes[bi, idx],
                        "boxes_xyxy": _xywh_to_xyxy(boxes[bi, idx]),
                        "scores": scores[bi, idx],
                        "classes": cls_id[bi, idx],
                        "keypoints": keypoints[bi, idx],
                        "keypoint_scores": kpt_conf[bi, idx],
                    }
                )
    assert per_image is not None
    decoded = []
    for chunks in per_image:
        if not chunks:
            decoded.append(_empty_prediction(num_keypoints))
            continue
        merged = {k: torch.cat([c[k] for c in chunks], dim=0) for k in chunks[0].keys()}
        if int(pre_nms_topk) > 0 and merged["scores"].numel() > int(pre_nms_topk):
            keep_pre = merged["scores"].topk(int(pre_nms_topk)).indices
            merged = {k: v[keep_pre] for k, v in merged.items()}
        keep = _nms(merged["boxes_xyxy"], merged["scores"], float(nms_iou), int(max_detections))
        decoded.append({k: v[keep] for k, v in merged.items()})
    return decoded


def _empty_prediction(num_keypoints: int) -> dict[str, torch.Tensor]:
    return {
        "boxes_xywh": torch.zeros((0, 4)),
        "boxes_xyxy": torch.zeros((0, 4)),
        "scores": torch.zeros((0,)),
        "classes": torch.zeros((0,), dtype=torch.long),
        "keypoints": torch.zeros((0, num_keypoints, 2)),
        "keypoint_scores": torch.zeros((0, num_keypoints)),
    }


def _preprocess_batch(frames_bgr: list[np.ndarray], image_size: int, normalization: str, device: torch.device) -> torch.Tensor:
    tensors = []
    for frame_bgr in frames_bgr:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != int(image_size) or rgb.shape[1] != int(image_size):
            rgb = cv2.resize(rgb, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)
        arr = rgb.astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    batch = torch.stack(tensors, dim=0).to(device)
    if str(normalization).lower() == "imagenet":
        batch = (batch - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return batch


def _prediction_to_arrays(
    pred: dict[str, torch.Tensor],
    n_animals: int,
    frame_shape: Tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_img, w_img = frame_shape[:2]
    k = int(pred["keypoints"].shape[1]) if pred["keypoints"].ndim == 3 else 0
    boxes = np.zeros((n_animals, 4), dtype=np.float32)
    conf = np.zeros((n_animals,), dtype=np.float32)
    track_ids = np.full((n_animals,), -1, dtype=np.int32)
    kxy = np.zeros((n_animals, k, 2), dtype=np.float32)
    kconf = np.zeros((n_animals, k), dtype=np.float32)
    present = np.zeros((n_animals,), dtype=bool)

    scores = pred["scores"].detach().cpu().numpy().astype(np.float32)
    if scores.size == 0:
        return boxes, conf, track_ids, kxy, kconf, present
    order = np.argsort(-scores)
    slot_ids_t = pred.get("slot_ids")
    if slot_ids_t is not None and slot_ids_t.numel() == scores.size:
        slot_ids = slot_ids_t.detach().cpu().numpy().astype(np.int32)
        order = np.argsort(slot_ids)
    else:
        slot_ids = np.arange(scores.size, dtype=np.int32)
    xyxy_norm = pred["boxes_xyxy"].detach().cpu().numpy().astype(np.float32)
    kxy_norm = pred["keypoints"].detach().cpu().numpy().astype(np.float32)
    kconf_np = pred["keypoint_scores"].detach().cpu().numpy().astype(np.float32)
    for det_idx in order:
        slot = int(slot_ids[det_idx]) if slot_ids.size else int(det_idx)
        if slot < 0 or slot >= int(n_animals) or present[slot]:
            empty = np.where(~present)[0]
            if empty.size == 0:
                break
            slot = int(empty[0])
        box = xyxy_norm[det_idx].copy()
        box[[0, 2]] *= float(w_img)
        box[[1, 3]] *= float(h_img)
        kp = kxy_norm[det_idx].copy()
        kp[:, 0] *= float(w_img)
        kp[:, 1] *= float(h_img)
        boxes[slot] = box
        conf[slot] = scores[det_idx]
        track_ids[slot] = slot
        kxy[slot] = kp
        kconf[slot] = kconf_np[det_idx]
        present[slot] = True
    return boxes, conf, track_ids, kxy, kconf, present


def _build_detection(
    frame_idx: int,
    frame_bgr: np.ndarray,
    boxes: np.ndarray,
    conf: np.ndarray,
    track_ids: np.ndarray,
    kxy: np.ndarray,
    kconf: np.ndarray,
    present: np.ndarray,
    *,
    n_animals: int,
    crop_size: int,
    group_crop_size: int,
    animal_scale_factor: float,
    group_scale_factor: float,
    body_length_px: Optional[float],
    pose_conf_threshold: float,
    fps: float,
    track_first_seen: dict[int, int],
    prev_boxes: Optional[np.ndarray],
    prev_kxy: Optional[np.ndarray],
) -> tuple[NCropDetection, np.ndarray, np.ndarray]:
    centers = np.array([(bb[0] + bb[2], bb[1] + bb[3]) for bb in boxes], dtype=np.float32) * 0.5
    valid_centers = centers[present]
    if valid_centers.size:
        group_center = valid_centers.mean(axis=0)
    else:
        h, w = frame_bgr.shape[:2]
        group_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    body_len = float(body_length_px or 0.0)
    if body_len <= 0:
        valid_w = boxes[present, 2] - boxes[present, 0] if present.any() else np.array([crop_size], dtype=np.float32)
        valid_h = boxes[present, 3] - boxes[present, 1] if present.any() else np.array([crop_size], dtype=np.float32)
        body_len = float(np.nanmedian(np.maximum(valid_w, valid_h))) if valid_w.size else float(crop_size)
        body_len = max(body_len, 1.0)

    group_xyxy = compute_fixed_square_xyxy(frame_bgr.shape, (float(group_center[0]), float(group_center[1])), body_len * float(group_scale_factor))
    animal_xyxys = np.array(
        [
            compute_fixed_square_xyxy(frame_bgr.shape, (float(centers[i, 0]), float(centers[i, 1])), body_len * float(animal_scale_factor))
            for i in range(n_animals)
        ],
        dtype=np.int32,
    )
    group_rgb = crop_resize_pad_rgb(frame_bgr, group_xyxy, group_crop_size)
    animal_rgbs = np.stack([crop_resize_pad_rgb(frame_bgr, tuple(xyxy), crop_size) for xyxy in animal_xyxys], axis=0)

    pose_conf = np.nan_to_num(kconf.mean(axis=1), nan=0.0).astype(np.float32) if kconf.size else np.zeros((n_animals,), dtype=np.float32)
    pose_conf = np.where(present, pose_conf, 0.0).astype(np.float32)
    pose_mask = (pose_conf >= float(pose_conf_threshold)) & present
    crop_conf = np.array([crop_coverage_conf(tuple(xyxy), frame_bgr.shape) for xyxy in animal_xyxys], dtype=np.float32)
    track_conf = np.where(present, conf, 0.0).astype(np.float32)
    track_age = np.zeros((n_animals,), dtype=np.float32)
    for i, tid in enumerate(track_ids.tolist()):
        if tid >= 0 and present[i]:
            track_first_seen.setdefault(int(tid), int(frame_idx))
            denom = float(fps) if float(fps) > 1e-6 else 30.0
            track_age[i] = min(float(frame_idx - track_first_seen[int(tid)] + 1) / denom, 1.0)

    keypoints_crop_norm = np.zeros((n_animals, kxy.shape[1], 3), dtype=np.float32)
    if kxy.ndim == 3 and kxy.shape[1] > 0:
        for i in range(n_animals):
            x1, y1, x2, y2 = [float(v) for v in animal_xyxys[i]]
            cw = max(x2 - x1, 1.0)
            ch = max(y2 - y1, 1.0)
            keypoints_crop_norm[i, :, 0] = (kxy[i, :, 0] - x1) / cw
            keypoints_crop_norm[i, :, 1] = (kxy[i, :, 1] - y1) / ch
            keypoints_crop_norm[i, :, 2] = kconf[i] if kconf.ndim == 2 else 0.0

    rel, rel_pose_mask, rel_present = extract_relational_features(
        kxy,
        boxes,
        present,
        pose_mask,
        body_len,
        prev_keypoints_raw=prev_kxy,
        prev_bboxes=prev_boxes,
        fps=fps,
        pose_conf_threshold=float(pose_conf_threshold),
        keypoints_conf_raw=kconf if kconf.size else None,
    )
    det = NCropDetection(
        frame_idx=frame_idx,
        group_rgb=group_rgb,
        animal_rgbs=animal_rgbs,
        animal_mask=present.astype(bool),
        pose_mask=pose_mask.astype(bool),
        pose_conf=pose_conf,
        crop_conf=crop_conf,
        track_conf=track_conf,
        track_age=track_age,
        relation_features=rel,
        relation_pose_mask=rel_pose_mask,
        relation_present=rel_present,
        bbox_xyxy_animals=boxes.astype(np.int32),
        bbox_xyxy_group=tuple(int(v) for v in group_xyxy),
        crop_xyxy_animals=animal_xyxys.astype(np.int32),
        crop_xyxy_group=tuple(int(v) for v in group_xyxy),
        track_ids=track_ids.astype(np.int32),
        keypoints_xy_raw=kxy.astype(np.float32),
        keypoints_conf_raw=kconf.astype(np.float32),
        keypoints_crop_norm=keypoints_crop_norm.astype(np.float32),
        yolo_status="mobilenetv3_detected" if bool(present.any()) else "mobilenetv3_missed",
    )
    return det, boxes.copy(), kxy.copy()


def iterate_mobilenetv3_n_crops(
    runtime: MobileNetV3PoseRuntime,
    video_path: Path | str,
    *,
    n_animals: int = 2,
    crop_size: int = 224,
    group_crop_size: int | None = None,
    animal_scale_factor: float = 4.0,
    group_scale_factor: float = 8.0,
    body_length_px: float | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = "cpu",
    keep_last_box: bool = True,
    pose_conf_threshold: float = 0.3,
    fps: float = 30.0,
    batch_size: int = 8,
    pre_nms_topk: int = 1000,
) -> Generator[NCropDetection, None, None]:
    dev = next(runtime.model.parameters()).device
    if str(dev) != str(torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")):
        # The loaded runtime owns placement. This branch avoids silently moving
        # tensors for every frame if a caller passes a stale device string.
        device_obj = dev
    else:
        device_obj = dev
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    group_crop_size = int(group_crop_size or crop_size)
    n_animals = int(n_animals)
    tracker = _SlotTracker(num_slots=n_animals)
    frame_idx = 0
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    last = None
    track_first_seen: dict[int, int] = {}
    prev_boxes = None
    prev_kxy = None

    def flush_batch() -> list[NCropDetection]:
        nonlocal batch_frames, batch_indices, last, prev_boxes, prev_kxy
        if not batch_frames:
            return []
        x = _preprocess_batch(batch_frames, runtime.image_size, runtime.normalization, device_obj)
        with torch.no_grad():
            decoded = decode_pose_outputs(
                runtime.model(x),
                num_keypoints=runtime.num_keypoints,
                num_classes=runtime.num_classes,
                conf_threshold=float(conf_threshold),
                nms_iou=float(iou_threshold),
                max_detections=max(int(n_animals) * 4, int(n_animals)),
                pre_nms_topk=int(pre_nms_topk),
                parameterization=runtime.parameterization,
            )
        out: list[NCropDetection] = []
        for idx, frame_bgr, pred in zip(batch_indices, batch_frames, decoded):
            tracked = tracker.update(pred)
            boxes, conf, track_ids, kxy, kconf, present = _prediction_to_arrays(tracked, n_animals, frame_bgr.shape)
            if not present.any() and keep_last_box and last is not None:
                boxes, conf, track_ids, kxy, kconf, crop_present = [v.copy() for v in last]
                present = np.zeros_like(crop_present, dtype=bool)
            elif present.any():
                last = (boxes.copy(), conf.copy(), track_ids.copy(), kxy.copy(), kconf.copy(), present.copy())
            det, prev_boxes, prev_kxy = _build_detection(
                int(idx),
                frame_bgr,
                boxes,
                conf,
                track_ids,
                kxy,
                kconf,
                present,
                n_animals=n_animals,
                crop_size=int(crop_size),
                group_crop_size=group_crop_size,
                animal_scale_factor=float(animal_scale_factor),
                group_scale_factor=float(group_scale_factor),
                body_length_px=body_length_px,
                pose_conf_threshold=float(pose_conf_threshold),
                fps=float(fps),
                track_first_seen=track_first_seen,
                prev_boxes=prev_boxes,
                prev_kxy=prev_kxy,
            )
            out.append(det)
        batch_frames = []
        batch_indices = []
        return out

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            frame_idx += 1
            if len(batch_frames) >= int(batch_size):
                yield from flush_batch()
        yield from flush_batch()
    finally:
        cap.release()


class _SlotTracker:
    def __init__(self, num_slots: int, max_cost: float = 1.25, max_missed: int = 8):
        self.num_slots = int(num_slots)
        self.max_cost = float(max_cost)
        self.max_missed = int(max_missed)
        self.slots: list[dict[str, torch.Tensor | int] | None] = [None for _ in range(self.num_slots)]

    def update(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scores = pred["scores"].detach().float().cpu()
        if scores.numel() == 0:
            for slot in self.slots:
                if slot is not None:
                    slot["missed"] = int(slot.get("missed", 0)) + 1
            self.slots = [slot if slot is not None and int(slot.get("missed", 0)) <= self.max_missed else None for slot in self.slots]
            return pred
        order = scores.argsort(descending=True).tolist()
        used: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for slot_id, slot in enumerate(self.slots):
            if slot is None:
                continue
            for det_idx in order:
                cost = self._cost(slot, pred, det_idx)
                if cost <= self.max_cost:
                    pairs.append((cost, slot_id, det_idx))
        pairs.sort(key=lambda x: x[0])
        updated: set[int] = set()
        for _, slot_id, det_idx in pairs:
            if slot_id in updated or det_idx in used:
                continue
            self.slots[slot_id] = self._state(pred, det_idx, self.slots[slot_id])
            updated.add(slot_id)
            used.add(det_idx)
        for slot_id, slot in enumerate(self.slots):
            if slot is None or slot_id in updated:
                continue
            slot["missed"] = int(slot.get("missed", 0)) + 1
            if int(slot["missed"]) > self.max_missed:
                self.slots[slot_id] = None
        for det_idx in order:
            if det_idx in used:
                continue
            empty = [i for i, slot in enumerate(self.slots) if slot is None]
            if not empty:
                break
            self.slots[empty[0]] = self._state(pred, det_idx, None)
        active = [(i, slot) for i, slot in enumerate(self.slots) if slot is not None]
        if not active:
            return pred
        return {
            "boxes_xyxy": torch.stack([slot["box"] for _, slot in active], dim=0),
            "boxes_xywh": torch.stack([_xyxy_to_xywh(slot["box"]) for _, slot in active], dim=0),
            "keypoints": torch.stack([slot["keypoints"] for _, slot in active], dim=0),
            "keypoint_scores": torch.stack([slot["keypoint_scores"] for _, slot in active], dim=0),
            "scores": torch.stack([slot["score"] for _, slot in active], dim=0),
            "classes": torch.stack([slot["cls"] for _, slot in active], dim=0),
            "slot_ids": torch.tensor([slot_id for slot_id, _ in active], dtype=torch.long),
        }

    def _state(self, pred: dict[str, torch.Tensor], det_idx: int, previous: Optional[dict]) -> dict:
        return {
            "box": pred["boxes_xyxy"][det_idx].detach().float().cpu(),
            "keypoints": pred["keypoints"][det_idx].detach().float().cpu(),
            "keypoint_scores": pred["keypoint_scores"][det_idx].detach().float().cpu(),
            "score": pred["scores"][det_idx].detach().float().cpu(),
            "cls": pred["classes"][det_idx].detach().long().cpu(),
            "missed": 0,
            "age": 1 if previous is None else int(previous.get("age", 0)) + 1,
        }

    def _cost(self, slot: dict, pred: dict[str, torch.Tensor], det_idx: int) -> float:
        box = pred["boxes_xyxy"][det_idx].detach().float().cpu()
        old = slot["box"]
        iou_cost = 1.0 - float(_single_iou(old, box))
        center_cost = float(torch.linalg.vector_norm(_center(old) - _center(box)))
        old_scores = slot["keypoint_scores"]
        new_scores = pred["keypoint_scores"][det_idx].detach().float().cpu()
        valid = (old_scores > 0.5) & (new_scores > 0.5)
        if bool(valid.any()):
            kpt_cost = float(torch.linalg.vector_norm(slot["keypoints"][valid] - pred["keypoints"][det_idx].detach().float().cpu()[valid], dim=1).mean())
        else:
            kpt_cost = 0.0
        return iou_cost + center_cost + 0.4 * kpt_cost


def _center(box_xyxy: torch.Tensor) -> torch.Tensor:
    return torch.stack([(box_xyxy[0] + box_xyxy[2]) * 0.5, (box_xyxy[1] + box_xyxy[3]) * 0.5])


def _xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            (box_xyxy[0] + box_xyxy[2]) * 0.5,
            (box_xyxy[1] + box_xyxy[3]) * 0.5,
            (box_xyxy[2] - box_xyxy[0]).clamp(min=0.0),
            (box_xyxy[3] - box_xyxy[1]).clamp(min=0.0),
        ]
    )


def _single_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(a[:2], b[:2])
    rb = torch.minimum(a[2:], b[2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[0] * wh[1]
    area_a = (a[2] - a[0]).clamp(min=0.0) * (a[3] - a[1]).clamp(min=0.0)
    area_b = (b[2] - b[0]).clamp(min=0.0) * (b[3] - b[1]).clamp(min=0.0)
    return inter / (area_a + area_b - inter + 1e-7)


