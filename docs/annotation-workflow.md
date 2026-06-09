# Annotation Workflow

The annotation workflow is designed to produce full-video behavior labels with explicit split assignment and quality-control state. The key principle is to keep manual bout boundaries traceable from the GUI to the exported `.annot` files and manifest.

## Recommended Workflow

1. **Import videos.** Use `File > Import videos...` for selected files or `File > Import folder...` for a directory.
2. **Define behaviors.** Use `Project > Behaviors...` to set behavior names, definitions, colors, and hotkeys.
3. **Assign splits.** Select videos in the left panel, then use `Project > Assign selected videos to...`.
4. **Label spans.** Choose a behavior and mark the start and end of each bout.
5. **Check boundaries.** Step frame-by-frame around ambiguous transitions with `[` and `]`.
6. **Approve accepted spans.** Use Draft, Ready, Approve, and Reject to separate working notes from accepted labels.
7. **Export full-video annotations.** Use `Project > Export full-video annotations...`.

## Annotation States

- **Draft**: a working label that should not yet be treated as final.
- **Ready**: a span whose behavior and boundaries are ready for quality control.
- **Approved**: a span accepted for training or evaluation export.
- **Rejected**: a span intentionally excluded after checking.

Approved spans are the expected source for training and held-out evaluation exports.

## Boundary Guidance

BehaviorScope-X is most useful when bout boundaries are treated as measurements, not only class labels. For short social behaviors, small boundary errors can affect bout F1 even when frame accuracy looks strong. Therefore:

- label the first frame where the behavior is visibly present,
- label the last frame before the behavior visibly ends,
- use the notes field for occlusion, contact ambiguity, or partial visibility,
- lock spans after the boundary is correct,
- keep behavior definitions stable across annotators and videos.

## Exported Files

The full-video export writes:

- a `source_manifest.csv` linking videos, splits, and annotation files,
- per-video `.annot` files,
- class-name metadata when available.

These files are the bridge from the GUI to cache construction for YOLO-pose, MobileNetV3, and DeepLabCut-HRNet workflows.



