# Expected DLC Outputs

The DLC analysis produces separate artifacts for pose training, top-down cache
construction, HRNet feature extraction, classifier training, and held-out
evaluation.

## Training

Expected pose/detector state files are written under the DLC project root:

```text
pipeline_state/dlc_training/
dlc-models-pytorch/iteration-0/.../train/snapshot-best-027.pt
dlc-models-pytorch/iteration-0/.../train/snapshot-detector-best-004.pt
```

The exact epoch numbers may differ if the full training is rerun, but the
selection criteria should remain metric-driven and recorded in the training
state JSON files.

## Train/Validation Caches

```text
BehaviorScope_X_minimal/outputs/npz_cache/mars_full_video_dlc_topdown
BehaviorScope_X_minimal/outputs/npz_cache/mars_dlc_topdown_hrnet_feature_cache_trainval
```

## Held-Out Caches

```text
outputs/npz_cache/mars_full_video_dlc_topdown_heldout
outputs/npz_cache/mars_dlc_topdown_hrnet_feature_cache_heldout
```

## Classifier and Evaluation

```text
behaviorscope_outputs/dlc_topdown_attention256_training/mars_dlc_topdown_hrnet_attention256_seed42
behaviorscope_outputs/dlc_topdown_heldout_eval/mars_dlc_topdown_hrnet_attention256_seed42
```


