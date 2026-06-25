"""
BehaviorScope-Y model: MultiAnimalBehaviorSequenceClassifier with YOLO visual backbone.

Differences from BehaviorScope-N (model_n.py):
  * Visual backbone is a YOLO-pose model's CSP-Darknet trunk (layers 0-9
    through SPPF), NOT MobileNet/EfficientNet/ViT/SlowFast.
  * The YOLO backbone is MARS-domain-pretrained (trained on MARS keypoint
    annotations), giving features tuned to mouse appearance from day one.
  * Architecturally identical otherwise: same four streams (group RGB,
    per-animal RGB, pose-self, relations), same fusion, same temporal head.

Hard-mask vs soft-gate semantics (locked in proposal Â§6, Â§8):
  * `animal_mask` is the HARD MASK for cross-animal attention pooling.
  * `pose_mask` is the SOFT GATE inside per-animal hybrid fusion only.
  * Reliability features (`pose_conf`, `crop_conf`, `track_conf`,
    `track_age`) are concatenated to tokens as INPUT features, not used
    as masks.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

try:
    from model import (
        PoseVisualFusion,
        TemporalAttentionPooling,
        TemporalPositionalEncoding,
    )
    from config import MODEL_DEFAULTS
except Exception:  # pragma: no cover
    from .model import (
        PoseVisualFusion,
        TemporalAttentionPooling,
        TemporalPositionalEncoding,
    )
    from .config import MODEL_DEFAULTS


# ----------------------------------------------------------------------------
# YOLO visual backbone
# ----------------------------------------------------------------------------

class YOLOFrameEncoder(nn.Module):
    """
    Per-frame visual encoder backed by a YOLO-pose model's CSP-Darknet trunk.

    Loads a YOLOv8/v11-pose checkpoint, extracts layers 0-9 (the backbone
    through SPPF), applies global average pooling, and returns a fixed-size
    feature vector per crop. The C2PSA block (layer 10), FPN neck
    (layers 11-22), and pose head (layer 23) are discarded because the
    behavior model uses the SPPF appearance descriptor only.

    Input :  [B, 3, H, W]   (e.g. 224x224 RGB crop)
    Output:  [B, feature_dim]   (typically 256 for YOLOv11n-pose)
    """

    DEFAULT_BACKBONE_END_LAYER = 10  # exclusive: keeps layers [0, 1, ..., 9]

    def __init__(
        self,
        weights_path: str,
        backbone_end_layer: int = DEFAULT_BACKBONE_END_LAYER,
        trainable: bool = False,
    ):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "BehaviorScope-Y requires ultralytics. Install with `pip install ultralytics`."
            ) from exc

        yolo = YOLO(weights_path)
        full_model = yolo.model.model  # nn.ModuleList of layers

        if backbone_end_layer < 1 or backbone_end_layer > len(full_model):
            raise ValueError(
                f"backbone_end_layer={backbone_end_layer} out of range "
                f"[1, {len(full_model)}]"
            )

        # Take a copy of layers [0, backbone_end_layer). These layers are
        # purely sequential in YOLO's backbone (no cross-layer skip
        # connections within layers 0-9), so wrapping in nn.Sequential
        # is safe.
        self.backbone = nn.Sequential(*[full_model[i] for i in range(backbone_end_layer)])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Probe output dim with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.backbone(dummy)
        self.feature_dim: int = int(feat.shape[1])

        # Ultralytics loads YOLO weights with requires_grad=False by default
        # (inference mode). Explicitly set the requires_grad flag here so
        # --train_backbone genuinely fine-tunes when requested, and frozen
        # mode is unambiguous when not.
        for p in self.backbone.parameters():
            p.requires_grad = bool(trainable)
        if not trainable:
            # Lock BatchNorm / Dropout statistics so the frozen YOLO behaves
            # identically to its detection-time configuration.
            self.backbone.eval()

        self._weights_path = weights_path
        self._backbone_end_layer = backbone_end_layer
        self._trainable = bool(trainable)

    def train(self, mode: bool = True):
        # When the backbone is frozen, keep it in eval() regardless of the
        # parent model's training mode. Prevents BN running-stat updates and
        # any dropout activations inside the YOLO trunk.
        super().train(mode)
        if not self._trainable:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._trainable:
            feat = self.backbone(x)
        else:
            # Frozen backbone: skip autograd-graph construction entirely.
            # Drops activation memory (~8-10 GB at batch=32 Ã— 32 frames Ã—
            # 3 crop pathways) and avoids unnecessary backward computation
            # through layers that have no learnable parameters in this run.
            # The output is detached so downstream gradients flow only into
            # the trainable head, not back through the YOLO trunk.
            with torch.no_grad():
                feat = self.backbone(x)
            feat = feat.detach()
        feat = self.pool(feat).flatten(1)  # [B, C]
        return feat

    def extra_repr(self) -> str:
        return (
            f"weights={self._weights_path!r}, "
            f"end_layer={self._backbone_end_layer}, "
            f"feature_dim={self.feature_dim}, "
            f"trainable={self._trainable}"
        )


def build_yolo_encoder(
    weights_path: str,
    trainable: bool = False,
    backbone_end_layer: int = YOLOFrameEncoder.DEFAULT_BACKBONE_END_LAYER,
) -> Tuple[nn.Module, int]:
    """Drop-in replacement for build_frame_encoder, returning (encoder, feature_dim)."""
    enc = YOLOFrameEncoder(
        weights_path=weights_path,
        backbone_end_layer=backbone_end_layer,
        trainable=trainable,
    )
    return enc, enc.feature_dim


def inspect_yolo_keypoint_count(weights_path: str) -> Optional[int]:
    """Return the number of keypoints the given YOLO-pose checkpoint emits, or
    None if the checkpoint is not a pose model. Used to validate that the
    BehaviorScope-Y `num_keypoints` setting matches the YOLO model that will
    serve as both the visual backbone and the inference-time pose detector.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    try:
        yolo = YOLO(weights_path)
    except Exception:
        return None
    if getattr(yolo, "task", None) != "pose":
        return None
    # Ultralytics stores pose head config under model.model.kpt_shape = (nkpt, ndim)
    kpt_shape = getattr(yolo.model, "kpt_shape", None)
    if kpt_shape is None:
        # Older ultralytics versions: walk to the head module
        head = yolo.model.model[-1]
        kpt_shape = getattr(head, "kpt_shape", None)
    if kpt_shape is None:
        return None
    try:
        return int(kpt_shape[0])
    except (TypeError, IndexError):
        return None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean over `dim` weighted by float(mask). Falls back to 0 where mask
    is empty along the reduction axis."""
    m = mask.to(x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(-1)
    summed = (x * m).sum(dim=dim)
    denom = m.sum(dim=dim).clamp(min=1.0)
    return summed / denom


# ----------------------------------------------------------------------------
# Submodules
# ----------------------------------------------------------------------------

class PoseSelfEncoder(nn.Module):
    """Project per-animal pose features (incl. reliability) to a fixed
    hidden dim. Used as the `pose` input to BehaviorScope's existing
    `PoseVisualFusion` per-animal."""

    def __init__(self, pose_input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        # x: [B, T, N, D_in] or [B*N, T, D_in]
        return self.net(x)


class RelationEncoder(nn.Module):
    """Encode each pair's [REL_FEATURE_DIM + 2] relation+reliability vector
    into a hidden representation. Applied to every (i, j) pair; pooling
    happens elsewhere."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        # x: [..., D_in]
        return self.net(x)


class CausalTCNBlock(nn.Module):
    """Residual causal temporal-convolution block for sequence classification."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("TCN kernel_size must be >= 1")
        self.left_pad = int((kernel_size - 1) * dilation)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        return conv(F.pad(x, (self.left_pad, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._causal_conv(self.conv1, x)
        y = self.dropout(F.relu(self.norm1(y), inplace=True))
        y = self._causal_conv(self.conv2, y)
        y = self.dropout(F.relu(self.norm2(y), inplace=True))
        return F.relu(y + self.residual(x), inplace=True)


class TemporalConvNet(nn.Module):
    """Dilated causal TCN over `[B, T, D]` features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("TCN num_layers must be >= 1")
        layers = []
        in_ch = int(input_dim)
        for layer_idx in range(int(num_layers)):
            dilation = 2 ** layer_idx
            layers.append(
                CausalTCNBlock(
                    in_ch,
                    int(hidden_dim),
                    kernel_size=int(kernel_size),
                    dilation=dilation,
                    dropout=float(dropout),
                )
            )
            in_ch = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.out_dim = int(hidden_dim)
        self.receptive_field = 1 + (int(kernel_size) - 1) * ((2 ** int(num_layers)) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, D, T] -> [B, T, H]
        y = self.net(x.transpose(1, 2).contiguous())
        return y.transpose(1, 2).contiguous()


# ----------------------------------------------------------------------------
# Main classifier
# ----------------------------------------------------------------------------

class MultiAnimalBehaviorSequenceClassifier(nn.Module):
    """
    Architecture (proposal Â§5):

        group RGB        -> visual_encoder -> + group_view_emb -> group_token
        per-animal RGB   -> visual_encoder -> + animal_view_emb -> per-animal token
        per-animal pose  -> pose encoder + reliability concat
                         -> PoseVisualFusion(visual_token, pose_token)
                         -> per-animal hybrid token h_i
        cross-animal     -> masked_mean over present animals (hard mask)
                         -> g_t
        relation         -> per-pair encoder
                         -> masked_mean over present pairs (hard mask)
                         -> r_t
        temporal head    -> LSTM or attention over [B, T, group + g + r]
                         -> classifier
    """

    def __init__(
        self,
        num_classes: int,
        yolo_weights_path: str,
        train_backbone: bool = False,
        yolo_backbone_end_layer: int = YOLOFrameEncoder.DEFAULT_BACKBONE_END_LAYER,
        n_animals: int = 2,
        num_keypoints: int = 7,
        rel_feature_dim: int = 11,
        hidden_dim: int = MODEL_DEFAULTS["hidden_dim"],
        dropout: float = MODEL_DEFAULTS["dropout"],
        sequence_model: str = "lstm",
        num_lstm_layers: int = MODEL_DEFAULTS["num_layers"],
        tcn_layers: int = 5,
        tcn_kernel_size: int = 3,
        bidirectional_lstm: bool = False,
        use_attention_pool: bool = False,
        attention_heads: int = MODEL_DEFAULTS["attention_heads"],
        attention_proj_dim: int = 0,
        positional_encoding: str = "none",
        positional_encoding_max_len: int = MODEL_DEFAULTS["positional_encoding_max_len"],
        pose_fusion_dim: int = MODEL_DEFAULTS["pose_fusion_dim"],
        pose_fusion_strategy: str = "gated_attention",
        # Ablation flags
        disable_visual_streams: bool = False,
        disable_group_rgb: bool = False,
        disable_per_animal_rgb: bool = False,
        disable_pose_self: bool = False,
        disable_relations: bool = False,
        relations_pose_only: bool = False,
        precomputed_visual_dim: int = 0,
    ):
        super().__init__()
        self.n_animals = int(n_animals)
        self.num_keypoints = int(num_keypoints)
        self.rel_feature_dim = int(rel_feature_dim)

        # Ablation flags
        self.disable_visual_streams = bool(disable_visual_streams)
        self.disable_group_rgb = bool(disable_group_rgb) or self.disable_visual_streams
        self.disable_per_animal_rgb = bool(disable_per_animal_rgb) or self.disable_visual_streams
        self.disable_pose_self = bool(disable_pose_self)
        self.disable_relations = bool(disable_relations)
        self.relations_pose_only = bool(relations_pose_only)

        # ---- Visual backbone (shared) ----
        # Skip the build entirely when no visual stream is alive (R4 pose-only,
        # R4b geometry-only). This avoids:
        #   1. Loading Kinetics/ImageNet pretrained weights for an encoder
        #      that will never be called in forward().
        #   2. Allocating ~150â€“200 MB of GPU memory for SlowFast weights
        #      whose gradients are never computed.
        #   3. Forcing a Kinetics weight download on a fresh machine for a
        #      pose-only run that doesn't need it.
        self._visual_alive = not (self.disable_group_rgb and self.disable_per_animal_rgb)
        self.precomputed_visual_dim = int(precomputed_visual_dim or 0)
        if self._visual_alive and self.precomputed_visual_dim > 0:
            self.frame_encoder = None
            feature_dim = int(self.precomputed_visual_dim)
        elif self._visual_alive:
            self.frame_encoder, feature_dim = build_yolo_encoder(
                weights_path=yolo_weights_path,
                trainable=train_backbone,
                backbone_end_layer=yolo_backbone_end_layer,
            )
            # Validate keypoint count alignment between the YOLO pose head
            # (used at inference for live keypoint detection) and the model's
            # configured num_keypoints (which defines the pose-feature
            # dimensionality the trained head expects). Mismatch here is a
            # silent footgun that would otherwise surface as a tensor-shape
            # error deep inside cropping_y at inference time.
            yolo_nkpt = inspect_yolo_keypoint_count(yolo_weights_path)
            if yolo_nkpt is not None and yolo_nkpt != self.num_keypoints:
                raise ValueError(
                    f"YOLO pose head emits {yolo_nkpt} keypoints, but model "
                    f"is configured for num_keypoints={self.num_keypoints}. "
                    f"These must match â€” either retrain the YOLO model with "
                    f"the right keypoint count, or set --num_keypoints={yolo_nkpt}."
                )
        else:
            self.frame_encoder = None
            feature_dim = 0
        self.feature_dim = feature_dim
        self.backbone_name = "precomputed_visual_cache" if self.precomputed_visual_dim > 0 else "yolo"
        self.backbone_type = "frame"  # YOLO backbone is always per-frame
        # YOLO backbone is per-frame; never a video backbone.
        self.is_video_backbone = False
        self.per_animal_rgb_alive = self._visual_alive and not self.disable_per_animal_rgb
        self.pose_self_alive = not self.disable_pose_self
        self.per_animal_stream_alive = self.per_animal_rgb_alive or self.pose_self_alive

        # View embeddings (small, learned post-pool tokens). Only allocated
        # when the visual encoder produces tokens to add them to.
        if self._visual_alive:
            self.group_view_embedding = nn.Parameter(torch.randn(feature_dim) * 0.02)
            self.animal_view_embedding = nn.Parameter(torch.randn(feature_dim) * 0.02)
        else:
            self.group_view_embedding = None
            self.animal_view_embedding = None

        # ---- Per-animal pose encoder ----
        # Pose feature dim from extract_rich_pose_features(K) = K*4 + K*(K-1)/2
        D_pose_self = num_keypoints * 4 + num_keypoints * (num_keypoints - 1) // 2
        # + 4 reliability features (pose_conf, crop_conf, track_conf, track_age)
        D_pose_input = D_pose_self + 4
        self.pose_self_encoder = PoseSelfEncoder(D_pose_input, hidden_dim, dropout)

        # ---- Per-animal hybrid fusion (PoseVisualFusion, reused) ----
        # When per-animal RGB is alive, build PoseVisualFusion. The fusion
        # module accepts pose features that may be zeroed (--disable_pose_self
        # zeros pose_input in forward), so building it for pose-disabled
        # configs is safe and preserves checkpoint compatibility for R2/R3
        # (per-animal RGB only, pose-self disabled). Skipping it when
        # pose_self is disabled would shrink fused_token_dim by pose_fusion_dim
        # and break .pt files saved before that change.
        # Only when per-animal RGB is dead (R4 / R4b) do we skip pose_fusion
        # entirely; the per-animal token is then either the pose-self encoder
        # output (R4 / R4b) or absent (degenerate "all disabled" config).
        if self.per_animal_rgb_alive:
            # cross_modal_transformer is not currently supported in the
            # multi-animal model: it expects a rank-1 pose mask but the
            # N-animal pipeline produces a rank-3 [B, T, N] mask that gets
            # collapsed to rank-2 before fusion. PoseVisualFusion's
            # transformer branch rejects rank-2 masks with a clear error,
            # so fail fast at construction time instead.
            if str(pose_fusion_strategy).strip().lower() == "cross_modal_transformer":
                raise ValueError(
                    "pose_fusion_strategy='cross_modal_transformer' is not "
                    "supported for the N-animal model. Use 'gated_attention' "
                    "(default) or 'concat'."
                )
            self.pose_fusion = PoseVisualFusion(
                visual_dim=feature_dim,
                pose_dim=hidden_dim,
                fusion_dim=pose_fusion_dim,
                strategy=pose_fusion_strategy,
                dropout=dropout,
            )
            self.fused_token_dim = self.pose_fusion.output_dim
        elif self.pose_self_alive:
            self.pose_fusion = None
            self.fused_token_dim = hidden_dim
        else:
            self.pose_fusion = None
            self.fused_token_dim = 0

        # ---- Relation encoder ----
        # Each pair's input = REL_FEATURE_DIM + 2 reliability scalars
        D_rel_input = rel_feature_dim + 2  # rel + (pose_pair_conf, bbox_pair_valid)
        self.relation_encoder = RelationEncoder(D_rel_input, hidden_dim, dropout)
        self.rel_token_dim = self.relation_encoder.out_dim

        # ---- Combined per-frame dim into temporal head ----
        # group_token (feature_dim) + per-animal-pooled (fused_token_dim) + relation-pooled (rel_token_dim)
        D_combined = 0
        if not self.disable_group_rgb:
            D_combined += feature_dim
        if self.per_animal_stream_alive:
            D_combined += self.fused_token_dim
        if not self.disable_relations:
            D_combined += self.rel_token_dim
        if D_combined == 0:
            raise ValueError("All streams disabled â€” nothing to classify.")
        self.combined_dim = D_combined

        # Optional positional encoding before the temporal head
        self.positional_encoding = (
            TemporalPositionalEncoding(
                self.combined_dim,
                kind=str(positional_encoding or "none").strip().lower(),
                max_len=int(positional_encoding_max_len),
            )
            if positional_encoding and positional_encoding != "none"
            else None
        )

        # ---- Temporal head ----
        self.sequence_model_type = sequence_model.lower()
        self.use_attention_pool = bool(use_attention_pool)
        if self.is_video_backbone:
            # SlowFast already does temporal modeling; collapse over T via mean.
            self.temporal_head = None
            head_in_dim = self.combined_dim
        elif self.sequence_model_type == "lstm":
            lstm = nn.LSTM(
                input_size=self.combined_dim,
                hidden_size=hidden_dim,
                num_layers=num_lstm_layers,
                batch_first=True,
                bidirectional=bidirectional_lstm,
                dropout=dropout if num_lstm_layers > 1 else 0.0,
            )
            self.temporal_head = lstm
            seq_out_dim = hidden_dim * (2 if bidirectional_lstm else 1)
            if self.use_attention_pool:
                self.attn_pool = TemporalAttentionPooling(
                    dim=seq_out_dim,
                    num_heads=int(attention_heads),
                    dropout=dropout,
                )
            else:
                self.attn_pool = None
            head_in_dim = seq_out_dim
        elif self.sequence_model_type == "attention":
            self.temporal_head = None
            _proj = int(attention_proj_dim)
            if _proj > 0 and _proj != self.combined_dim:
                self.attn_proj = nn.Linear(self.combined_dim, _proj)
                attn_dim = _proj
                # Rebuild PE at projected dim so sinusoidal freqs match
                if positional_encoding and positional_encoding != "none":
                    self.positional_encoding = TemporalPositionalEncoding(
                        attn_dim,
                        kind=str(positional_encoding or "none").strip().lower(),
                        max_len=int(positional_encoding_max_len),
                    )
            else:
                self.attn_proj = None
                attn_dim = self.combined_dim
            self.attn_pool = TemporalAttentionPooling(
                dim=attn_dim,
                num_heads=int(attention_heads),
                dropout=dropout,
            )
            head_in_dim = attn_dim
        elif self.sequence_model_type == "tcn":
            self.temporal_head = TemporalConvNet(
                input_dim=self.combined_dim,
                hidden_dim=hidden_dim,
                num_layers=int(tcn_layers),
                kernel_size=int(tcn_kernel_size),
                dropout=dropout,
            )
            self.attn_pool = None
            head_in_dim = self.temporal_head.out_dim
        else:
            raise ValueError(f"Unsupported sequence_model: {sequence_model}")

        # ---- Classifier head ----
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Dropout(dropout),
            nn.Linear(head_in_dim, num_classes),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 3, H, W]  ->  features [B, T, D]   (frame backbone)
                  for video backbone:  [B, 3, T, H, W] -> [B, D] (collapsed time)
        We always treat the per-frame encoder as a frame backbone here.
        For a video backbone (SlowFast), we pass the whole window in once and
        get a single per-clip vector, which we then broadcast across T.
        """
        B, T = x.shape[:2]
        if self.is_video_backbone:
            # Permute to [B, C, T, H, W] which is what SlowFast wants.
            v = x.permute(0, 2, 1, 3, 4).contiguous()
            feat = self.frame_encoder(v)  # [B, D]
            # Broadcast to T to keep downstream code uniform
            return feat.unsqueeze(1).expand(B, T, -1).contiguous()
        # Frame backbone: encode each frame independently, batched
        if self.frame_encoder is None:
            raise RuntimeError(
                "This model was constructed for precomputed visual features "
                f"(feature_dim={self.feature_dim}) and cannot encode raw RGB frames."
            )
        flat = x.reshape(B * T, *x.shape[2:])
        feat = self.frame_encoder(flat)
        return feat.reshape(B, T, -1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device

        # Pull batch items. non_blocking=True overlaps the CPU->GPU transfer
        # with subsequent ops because the DataLoader uses pin_memory=True;
        # without it the transfer is synchronous and stalls the GPU.
        #
        # Inference may provide precomputed visual features (`group_feat`,
        # `animal_feat`) so overlapping windows do not re-run the frozen YOLO
        # trunk on the same frames. Training still passes raw crops only.
        nb = (device.type == "cuda")
        animal_mask = batch["animal_mask"].to(device, non_blocking=nb)   # [B, T, N]  bool
        pose_mask = batch["pose_mask"].to(device, non_blocking=nb)       # [B, T, N]  bool
        pose_conf = batch["pose_conf"].to(device, non_blocking=nb)       # [B, T, N]
        crop_conf = batch["crop_conf"].to(device, non_blocking=nb)
        track_conf = batch["track_conf"].to(device, non_blocking=nb)
        track_age = batch["track_age"].to(device, non_blocking=nb)
        pose_self = batch["pose_self"].to(device, non_blocking=nb)       # [B, T, N, D_pose_self]
        rel_features = batch["relation_features"].to(device, non_blocking=nb)         # [B, T, N, N, R]
        rel_pose_mask = batch["relation_pose_mask"].to(device, non_blocking=nb)       # [B, T, N, N]
        rel_pose_conf = batch["relation_pose_conf"].to(device, non_blocking=nb)       # [B, T, N, N]
        rel_present = batch["relation_present"].to(device, non_blocking=nb)           # [B, T, N, N]

        B, T, N = animal_mask.shape

        # ============ Visual streams ============
        # When all visual streams are disabled, the visual backbone is
        # `None` and `_encode_frames` would fail. Skip the visual path
        # entirely; `animal_feat` stays a zero placeholder for shape
        # compatibility with the rest of the pipeline.
        group_token = None
        if self._visual_alive and not self.disable_group_rgb:
            if "group_feat" in batch:
                group_feat = batch["group_feat"].to(device, non_blocking=nb)  # [B, T, D]
            else:
                group = batch["group"].to(device, non_blocking=nb)            # [B, T, 3, H, W]
                group_feat = self._encode_frames(group)                       # [B, T, D]
            group_token = group_feat + self.group_view_embedding     # [B, T, D]

        # Per-animal RGB: encode each animal separately
        # animal: [B, N, T, 3, H, W] -> [B*N, T, 3, H, W]
        animal_feat = None
        if self.per_animal_rgb_alive:
            if "animal_feat" in batch:
                animal_feat = batch["animal_feat"].to(device, non_blocking=nb)  # [B, T, N, D]
                animal_feat = animal_feat + self.animal_view_embedding.squeeze(0)
            else:
                animal = batch["animal"].to(device, non_blocking=nb)            # [B, N, T, 3, H, W]
                an = animal.reshape(B * N, T, *animal.shape[-3:])
                animal_feat = self._encode_frames(an).reshape(B, N, T, -1)      # [B, N, T, D]
                animal_feat = animal_feat + self.animal_view_embedding
                animal_feat = animal_feat.permute(0, 2, 1, 3)                   # [B, T, N, D]

        # ============ Per-animal pose-self ============
        # Concatenate reliability features
        reliability = torch.stack(
            [pose_conf, crop_conf, track_conf, track_age], dim=-1
        )                                                              # [B, T, N, 4]
        pose_input = torch.cat([pose_self, reliability], dim=-1)        # [B, T, N, D_pose_input]
        if self.disable_pose_self:
            pose_input = torch.zeros_like(pose_input)
        pose_feat = self.pose_self_encoder(pose_input)                  # [B, T, N, hidden]

        # ============ Per-animal hybrid fusion ============
        animal_pool = None
        if self.pose_fusion is not None:
            # Visual is alive â€” fuse visual+pose per animal.
            # Reshape so PoseVisualFusion can process [B*N, T, D]
            assert animal_feat is not None
            v = animal_feat.permute(0, 2, 1, 3).reshape(B * N, T, self.feature_dim)
            p = pose_feat.permute(0, 2, 1, 3).reshape(B * N, T, pose_feat.shape[-1])
            # PoseVisualFusion's pose_present mask: [B*N, T] in our reshape order
            pp = pose_mask.permute(0, 2, 1).reshape(B * N, T)
            fused = self.pose_fusion(v, p, pose_present=pp)             # [B*N, T, D_fused]
            fused = fused.reshape(B, N, T, self.fused_token_dim).permute(0, 2, 1, 3)
            animal_pool = masked_mean(fused, animal_mask, dim=2)        # [B, T, D_fused]
        else:
            # Visual is dead (R4 / R4b) â€” the per-animal token *is* the
            # pose-self encoder output. Keep tensor layout consistent so
            # masked_mean over N still works downstream.
            fused = pose_feat                                            # [B, T, N, hidden_dim]
        # fused: [B, T, N, D_fused]

        # ============ Cross-animal pool (masked mean over N) ============
        # Hard mask: animal_mask  [B, T, N]
        if animal_pool is None:
            if self.per_animal_rgb_alive:
                assert animal_feat is not None
                animal_pool = masked_mean(animal_feat, animal_mask, dim=2)  # [B, T, D]
            elif self.pose_self_alive:
                animal_pool = masked_mean(pose_feat, animal_mask, dim=2)    # [B, T, hidden_dim]

        # ============ Relations ============
        relation_pool = None
        if not self.disable_relations:
            # Build relation token input: rel_features + [rel_pose_conf, animal_mask_pair]
            # Use rel_pose_conf and a separate "bbox pair valid" derived from rel_present
            rel_pose_conf_unsq = rel_pose_conf.unsqueeze(-1)             # [B, T, N, N, 1]
            rel_present_unsq = rel_present.float().unsqueeze(-1)         # [B, T, N, N, 1]
            rel_input = torch.cat([rel_features, rel_pose_conf_unsq, rel_present_unsq], dim=-1)
            # [B, T, N, N, R + 2]
            if self.relations_pose_only:
                # Zero the bbox half (indices 0..5) when ablation requires it
                bbox_half = rel_input[..., :6].zero_()
                _ = bbox_half  # for clarity
            rel_tokens = self.relation_encoder(rel_input)                # [B, T, N, N, D_rel]

            # Mask to present pairs (excludes diagonal automatically since
            # rel_present[..., i, i] is always False from preprocessing)
            pair_mask = rel_present                                      # [B, T, N, N]
            # Flatten N x N pair axes for masked_mean over pairs
            rel_flat = rel_tokens.reshape(B, T, N * N, self.rel_token_dim)
            mask_flat = pair_mask.reshape(B, T, N * N)
            relation_pool = masked_mean(rel_flat, mask_flat, dim=2)      # [B, T, D_rel]

        # ============ Combine streams ============
        parts = []
        if group_token is not None:
            parts.append(group_token)                                    # [B, T, D]
        if animal_pool is not None:
            parts.append(animal_pool)                                    # [B, T, D_fused]
        if relation_pool is not None:
            parts.append(relation_pool)                                  # [B, T, D_rel]
        combined = torch.cat(parts, dim=-1)                              # [B, T, D_combined]

        # Attention bottleneck projection (before PE so positions match dim)
        if getattr(self, "attn_proj", None) is not None:
            combined = self.attn_proj(combined)

        # Positional encoding (optional)
        if self.positional_encoding is not None:
            combined = self.positional_encoding(combined)

        # ============ Temporal head ============
        if self.is_video_backbone:
            # SlowFast already collapsed time; combined is repeated across T.
            # Take a mean over T as a passthrough.
            pooled = combined.mean(dim=1)
        elif self.sequence_model_type == "lstm":
            seq_out, _ = self.temporal_head(combined)                     # [B, T, hidden*]
            if self.attn_pool is not None:
                pooled = self.attn_pool(seq_out)
            else:
                pooled = seq_out[:, -1]
        elif self.sequence_model_type == "tcn":
            seq_out = self.temporal_head(combined)                         # [B, T, hidden]
            pooled = seq_out[:, -1]
        else:
            pooled = self.attn_pool(combined)

        # ============ Classifier ============
        logits = self.classifier(pooled)
        return logits




