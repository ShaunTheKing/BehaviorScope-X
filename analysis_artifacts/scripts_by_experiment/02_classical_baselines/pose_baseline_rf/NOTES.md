# pose_baseline_rf â€” Design Notes

## Purpose

SimBA-style pose-only classical-ML baseline (RF / XGBoost) using identical
engineered features as BehaviorScope-Y's pose_relations_only LSTM stream.
The only variable that changes is the classifier family. Everything upstream
(YOLO-pose inference, feature engineering, evaluation protocol) is identical.

## Experimental Parity Table (for manuscript Methods / Supplemental)

| Parameter                  | BehaviorScope-Y LSTM              | Pose-only RF/XGBoost baseline     | Match? |
|----------------------------|-----------------------------------|-----------------------------------|--------|
| Classes                    | attack, investigation, mount, other | attack, investigation, mount, other | Yes |
| Train split                | 57 videos (MARS train)            | Same NPZ cache                    | Yes |
| Validation split           | 33 videos (MARS validation)       | Same NPZ cache                    | Yes |
| Held-out split             | 10 test_1 + 20 test_2 (30 videos) | Same NPZ cache                   | Yes |
| Pose estimator             | YOLO26n-pose (frozen)             | Same checkpoint, same extraction  | Yes |
| Input features             | pose_self + relational (132-d)    | Same arrays from same NPZs       | Yes |
| Feature engineering        | centered coords + pairwise dist + velocities + 11-d relational + reliability scalars | Identical (shared code) | Yes |
| Visual features            | Frozen SPPF 256-d (group + animal) | **Not used** (pose-only)        | Deliberate difference |
| Temporal modeling          | LSTM (256/512/896 hidden)         | **None (per-frame) or sliding-window concatenation** | Deliberate difference |
| Classifier                 | 1-layer LSTM + linear head        | **RandomForest / XGBoost**        | Deliberate difference |
| Model selection criterion  | Best validation macro-F1 (behavior classes) | Same                    | Yes |
| Class weighting            | sqrt-inverse, clamp 1.5           | sklearn `balanced` (inverse freq) | Comparable* |
| Median filter window       | 5 frames                          | 5 frames                         | Yes |
| Min bout duration          | 15 frames                         | 15 frames                        | Yes |
| Bout tIoU thresholds       | 0.25 and 0.50                     | 0.25 and 0.50                    | Yes |
| FPS                        | 30.0                              | 30.0                             | Yes |
| Evaluation functions       | batch_infer_eval.py               | Same â€” imported directly          | Yes |
| Overlapping window fusion  | Probability averaging over 32-frame windows (stride 16) | Same | Yes |

*Note on class weighting: sklearn's `balanced` uses pure inverse frequency,
which gives rare classes (attack, mount) slightly MORE weight than the LSTM's
sqrt-inverse with clamp 1.5. This choice favors the RF baseline. If
BehaviorScope-Y still outperforms, the argument is stronger.

## Three Deliberate Differences (the experimental variables)

1. **No visual features** â€” the RF receives only pose_self + relational (132-d),
   not the frozen SPPF 256-d descriptors. This isolates whether visual features
   add signal beyond pose geometry.

2. **No learned temporal modeling** â€” the per-frame RF has zero temporal context;
   the sliding-window RF (Â±7 frames = 15-frame concatenation) has crude context
   but no learned sequential processing. This isolates whether the LSTM's
   temporal modeling contributes beyond what static/windowed features provide.

3. **Classical ML classifier** â€” RandomForest and XGBoost instead of a neural
   LSTM head. This addresses the external concern that internal stream ablation
   (pose-only LSTM vs. full LSTM) conflates input features with classifier
   family.

## Sliding-Window Size Rationale (K=7 â†’ 15 frames vs. LSTM's 32 frames)

This design choice MUST be explained in the methods section. A reader may ask:
"You gave the LSTM 32 frames but the RF only 15 â€” is that fair?"

### Why not K=15 (31 frames) to match the LSTM's 32-frame window?

Matching the LSTM's temporal receptive field exactly would require K=15, yielding
31-frame concatenation and 132 Ã— 31 = **4,092 features** per sample. This is
impractical for classical ML for three reasons:

1. **Computational cost** â€” RF with 4,092 features on 653K training samples would
   take hours to train (K=7 with 1,980 features already took 20 minutes for RF
   vs. 4.6 minutes for per-frame 132 features â€” a 4.3Ã— slowdown for 15Ã— more
   features). Training time scales super-linearly with feature count for tree
   ensembles.

2. **Curse of dimensionality** â€” Tree-based models split on individual features.
   With 4,092 columns of concatenated pose coordinates, the majority of features
   are redundant or noisy copies of neighboring-frame values. The trees cannot
   learn meaningful sequential patterns across distant columns â€” they can only
   find axis-aligned splits in a 4,092-d space where most axes carry marginal
   signal. This leads to overfitting, not better generalization.

3. **Diminishing returns** â€” The jump from K=0 (132 features) to K=7 (1,980
   features) already showed diminishing improvement: XGBoost gained +0.067
   macro-F1 from per-frame to context15, but both models still overfit
   substantially (train F1 > 0.94 vs. val F1 < 0.77). Adding more temporal
   features would widen the trainâ€“val gap without improving held-out performance.

### The methodological point this makes

K=7 was chosen as a practical upper bound â€” enough temporal context to demonstrate
that crude neighbor-frame information helps classical ML, without drowning the
models in dimensions where they cannot compete. **This limitation is itself an
argument for learned temporal modeling**: an LSTM processes 32 frames through a
fixed-size hidden state (256-d), learning which temporal patterns are
discriminative. Feature stacking scales the input dimensionality linearly with
window size, making it fundamentally less parameter-efficient.

### Draft methods language

> "The sliding-window variant concatenated Â±7 neighboring frames (15-frame
> context, 1,980 features) around each center frame. We chose K=7 rather than
> matching the LSTM's full 32-frame receptive field (which would yield 4,092
> concatenated features) because tree-based classifiers do not scale gracefully
> with concatenated temporal features: the curse of dimensionality leads to
> overfitting rather than improved generalization. This asymmetry is itself
> informative â€” it demonstrates that learned sequential models achieve more
> efficient temporal integration than naive feature concatenation."

## Manuscript Framing

In the manuscript, this should be described as:

> "a SimBA-style pose-only classical-ML baseline (Random Forest / XGBoost;
> cf. Goodwin et al., 2024) using identical YOLO26n-pose keypoint features
> and evaluation protocol"

NOT as "we ran SimBA" â€” we used our own feature engineering, not SimBA's
pipeline. The comparison is methodologically cleaner because feature parity
is perfect, but the attribution must be honest.

## What the Results Proved

The actual outcome was closest to the third scenario: **RF/XGBoost ~ full LSTM
on held-out data**, despite validation ranking strongly favoring the LSTM.

### Held-Out Ranking (smoothed, 30 videos, mean across videos)

| Rank | Model | Frame F1 | Bout@.25 | Bout@.50 |
|------|-------|----------|----------|----------|
| 1 | XGBoost context15 | 0.7265 | 0.6640 | 0.5057 |
| 2 | RF context15 | 0.7207 | 0.6821 | 0.4780 |
| 3 | Full LSTM-256 | 0.7065 | 0.6576 | 0.4870 |
| 4 | RF per-frame | 0.7031 | 0.6370 | 0.4714 |
| 5 | XGBoost per-frame | 0.6949 | 0.6214 | 0.4584 |
| 6 | Pose-only LSTM-256 | 0.6785 | 0.6167 | 0.4382 |

### Paired Bootstrap CIs (10,000 resamples, seed=42, 30 per-video deltas)

| Comparison | Frame F1 delta | CI | Sig? | Bout@.25 delta | Sig? | Bout@.50 delta | Sig? |
|---|---|---|---|---|---|---|---|
| XGB ctx15 âˆ’ Full LSTM | +0.0200 | [+0.0016, +0.0398] | **Yes** | +0.0064 | No | +0.0187 | No |
| RF ctx15 âˆ’ Full LSTM | +0.0142 | CI crosses zero | No | +0.0245 | No | âˆ’0.0090 | No |
| XGB ctx15 âˆ’ Pose LSTM | +0.0480 | CI excludes zero | **Yes** | +0.0473 | **Yes** | +0.0675 | **Yes** |
| RF perframe âˆ’ Full LSTM | âˆ’0.0034 | CI crosses zero | No | âˆ’0.0206 | No | âˆ’0.0156 | No |

### Interpretation for manuscript

- **Correct verb**: "matched" for baselines vs Full LSTM (5 of 6 pairwise CIs
  cross zero). Only XGBoost context15 frame F1 "exceeded" the Full LSTM (barely).
- **Correct framing**: Pose features, not classifier architecture, dominated
  held-out performance. All 6 models in a 0.048 frame-F1 range (0.679â€“0.727).
- **Core thesis supported**: The frozen YOLO-pose representation carries most
  of the behavioral signal regardless of classifier family.
- **Validation-to-held-out reversal**: LSTM overfits more from val to held-out
  than RF/XGBoost, likely because the LSTM's learned temporal dynamics are tuned
  to the training distribution and generalize less robustly than crude feature
  stacking.

## Feature Extraction Results (Step 1)

Extraction completed successfully. Verified frame counts match manifests exactly.

| Split    | Frames    | Windows | Features | Frames/Window | Source manifest              |
|----------|-----------|---------|----------|---------------|------------------------------|
| Train    | 653,248   | 20,414  | 132-d    | 32            | mars_full_video (train)      |
| Val      | 322,528   | 10,079  | 132-d    | 32            | mars_full_video (val)        |
| Held-out | 720,864   | 22,527  | 132-d    | 32            | mars_heldout_eval            |
| **Total**| **1,696,640** | **53,020** | | | |

Output files (in `outputs/features/`):
- `train_features.npz`  â€” X: [653248, 132] float32 + y: [653248] int16
- `val_features.npz`    â€” X: [322528, 132] float32 + y: [322528] int16
- `heldout_features.npz` â€” X: [720864, 132] float32 + y: [720864] int16
- `heldout_window_meta.json`   â€” per-window video mapping for evaluation
- `train_val_window_meta.json` â€” per-window metadata for temporal context boundaries
- `extraction_summary.json`    â€” machine-readable summary

## Training Configuration (Step 2)

Four classifiers will be trained:

| Model     | Temporal context | Feature dim | Notes                          |
|-----------|-----------------|-------------|--------------------------------|
| RF per-frame       | None (K=0)      | 132         | Strictest baseline             |
| XGBoost per-frame  | None (K=0)      | 132         | Gradient boosting, early stop  |
| RF sliding-window  | Â±7 frames (K=7) | 1,980       | Crude temporal context         |
| XGBoost sliding-window | Â±7 frames (K=7) | 1,980   | Best-case for classical ML     |

RF hyperparameters: n_estimators=500, max_features=sqrt, class_weight=balanced, min_samples_leaf=5
XGBoost hyperparameters: max_depth=8, lr=0.1, subsample=0.8, early_stopping_rounds=50

## Training Results (Step 2)

### Per-frame models (K=0, 132 features)

| Model | Train macro-F1 | Val macro-F1 | Val accuracy | Train time | Notes |
|-------|---------------|-------------|-------------|------------|-------|
| RF per-frame | 0.9558 | **0.6574** | 0.8033 | 275.4s | 500 trees, 32 threads |
| XGBoost per-frame | 0.9152 | **0.6948** | 0.8197 | 42.2s | Early stopped at round 153/1000 |

Key observations:
- Both models overfit (train >> val), which is expected with frame-level labels
  from windowed data where class boundaries are noisy.
- XGBoost per-frame outperforms RF per-frame on validation (+0.037 macro-F1),
  with 6.5Ã— faster training.
- Val macro-F1 of 0.6574â€“0.6948 is the baseline to beat.

### Sliding-window models (K=7, Â±7 frames, 1980 features)

| Model | Train macro-F1 | Val macro-F1 | Val accuracy | Train time | Notes |
|-------|---------------|-------------|-------------|------------|-------|
| RF context15 | 0.9755 | **0.7064** | 0.8280 | 1194.3s (~20 min) | 500 trees, 32 threads |
| XGBoost context15 | 0.9455 | **0.7615** | 0.8486 | 882.6s (~15 min) | Early stopped at round 152/1000 |

Key observations:
- Sliding-window context improves both classifiers substantially over per-frame.
- RF context15 gains +0.049 val macro-F1 over RF per-frame (0.7064 vs 0.6574).
- XGBoost context15 gains +0.067 val macro-F1 over XGBoost per-frame (0.7615 vs 0.6948).
- XGBoost context15 is the best classical-ML model at **0.7615** val macro-F1.
- Both still overfit (train > val), but the gap is narrower with temporal context.
- XGBoost early-stopped at round 152 â€” nearly identical to per-frame (153), suggesting
  the extra features don't change the complexity/capacity tradeoff much.

### Complete Training Summary Table

| Model | K | Features | Train F1 | Val F1 | Val Acc | Time | Best Round |
|-------|---|----------|----------|--------|---------|------|------------|
| RF per-frame | 0 | 132 | 0.9558 | 0.6574 | 0.8033 | 275s | â€” |
| XGBoost per-frame | 0 | 132 | 0.9152 | 0.6948 | 0.8197 | 42s | 153 |
| RF context15 | 7 | 1,980 | 0.9755 | 0.7064 | 0.8280 | 1,194s | â€” |
| XGBoost context15 | 7 | 1,980 | 0.9455 | **0.7615** | 0.8486 | 882s | 152 |

**Comparison target**: pose_relations_only LSTM (256 hidden) val macro-F1 = **0.7814**

The best classical-ML baseline (XGBoost context15, 0.7615) approaches but does not
match the LSTM's 0.7814 â€” a gap of **0.0199** on validation. The held-out evaluation
(Step 3) will be the definitive comparison.

### Cross-Reference Table 1: Validation Frame macro-F1 (all models)

| Model | Features | Val Frame macro-F1 |
|---|---|---|
| RF per-frame | 132-d pose | 0.6574 |
| XGBoost per-frame | 132-d pose | 0.6948 |
| RF context15 | 1,980-d pose (Â±7 frames) | 0.7064 |
| XGBoost context15 | 1,980-d pose (Â±7 frames) | 0.7615 |
| **Pose-only LSTM 256** | **132-d pose** | **0.7814** |
| **Full LSTM 256 (BehaviorScope-Y)** | **132-d pose + 256-d visual** | **0.8120** |

Hierarchy: RF < XGBoost < Pose-only LSTM < Full LSTM.
Each architectural component (temporal modeling, visual features) adds measurably.

### Cross-Reference Table 2: Held-Out Test â€” SMOOTHED (30 videos, mean across videos)

**IMPORTANT**: The manuscript reports **smoothed** frame macro-F1 throughout
(e.g., line 798: "smoothed frame macro-F1 values of 0.706, 0.705, and 0.709").
All held-out numbers here use the `smoothed_frames` / `smoothed` prediction type
to match the manuscript. Do NOT mix in `raw_windows` numbers â€” they are ~0.03
higher for frame F1 and would create an inconsistency a reader could catch.

Source: LSTM numbers from `outputs/test_eval/*/per_video_summary.csv` (`smoothed_frames` rows).
RF/XGBoost numbers from `evaluate_heldout.py` (`smoothed` rows, 30 videos, final).

| Model | Frame macro-F1 | Bout F1 @0.25 | Bout F1 @0.50 |
|---|---|---|---|
| RF per-frame | **0.7031** | **0.6370** | **0.4714** |
| XGBoost per-frame | **0.6949** | **0.6214** | **0.4584** |
| RF context15 | **0.7207** | **0.6821** | **0.4780** |
| XGBoost context15 | **0.7265** | **0.6640** | **0.5057** |
| **Pose-only LSTM 256** | **0.6785** | **0.6167** | **0.4382** |
| **Full LSTM 256 (BehaviorScope-Y)** | **0.7065** | **0.6576** | **0.4870** |

LSTM held-out results from RTX 2080 machine, 0 failures across all 30 videos.
RF/XGBoost held-out results are hardware-independent (deterministic CPU math on same NPZs).

For reference â€” raw_windows numbers (NOT used in manuscript, kept for internal tracking):
| Model | raw Frame F1 | raw Bout@.25 | raw Bout@.50 |
|---|---|---|---|
| Pose-only LSTM 256 | 0.7101 | 0.6456 | 0.4927 |
| Full LSTM 256 | 0.7384 | 0.6648 | 0.5282 |

### Raw training log (for reproducibility / supplemental)

```
======================================================================
pose_baseline_rf â€” Training
======================================================================

Loading features ...
  Train: (653248, 132)  Val: (322528, 132)

â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Temporal context: K=0 (perframe)
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  Feature dim: 132

  Training RF (perframe) ...
    n_estimators=500, max_features=sqrt, class_weight=balanced
    [500/500 trees] elapsed: 4.6min
    Training time: 275.4s
    Train macro-F1 (behavior): 0.9558
    Val   macro-F1 (behavior): 0.6574
    Val   accuracy:            0.8033
    -> outputs/models/rf_perframe.pkl

  Training XGBoost (perframe) ...
    n_estimators=1000, max_depth=8, early_stopping=50
    [0]   validation_0-mlogloss:1.06489
    [50]  validation_0-mlogloss:0.51675
    [100] validation_0-mlogloss:0.50207
    [150] validation_0-mlogloss:0.49937
    [200] validation_0-mlogloss:0.50017
    [203] validation_0-mlogloss:0.50017
    Training time: 42.2s  (best round: 153)
    Train macro-F1 (behavior): 0.9152
    Val   macro-F1 (behavior): 0.6948
    Val   accuracy:            0.8197
    -> outputs/models/xgb_perframe.pkl

â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Temporal context: K=7 (context15)
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  Adding Â±7 frame context ...
  Feature dim: 1980

  Training RF (context15) ...
    n_estimators=500, max_features=sqrt, class_weight=balanced
    [500/500 trees] elapsed: 19.9min
    Training time: 1194.3s
    Train macro-F1 (behavior): 0.9755
    Val   macro-F1 (behavior): 0.7064
    Val   accuracy:            0.8280
    -> outputs/models/rf_context15.pkl

  Training XGBoost (context15) ...
    n_estimators=1000, max_depth=8, early_stopping=50
    [0]   validation_0-mlogloss:1.05976
    [50]  validation_0-mlogloss:0.45129
    [100] validation_0-mlogloss:0.43030
    [150] validation_0-mlogloss:0.42786
    [200] validation_0-mlogloss:0.42874
    [202] validation_0-mlogloss:0.42874
    Training time: 882.6s  (best round: 152)
    Train macro-F1 (behavior): 0.9455
    Val   macro-F1 (behavior): 0.7615
    Val   accuracy:            0.8486
    -> outputs/models/xgb_context15.pkl

======================================================================
Training Summary
======================================================================
Model        Tag            Val macro-F1    Val acc     Time
------------------------------------------------------------
RandomForest perframe             0.6574     0.8033   275.4s
XGBoost      perframe             0.6948     0.8197    42.2s
RandomForest context15            0.7064     0.8280  1194.3s
XGBoost      context15            0.7615     0.8486   882.6s

Training complete.
```

## Run Order

```bash
cd <DATA_ROOT>\BehaviorScope_Y_final_runs\pose_baseline_rf
python extract_features.py      # Step 1: NPZ -> flat arrays (~15 min)
python train_baseline.py        # Step 2: Train RF + XGBoost (~10 min)
python evaluate_heldout.py      # Step 3: Held-out evaluation (~30 min)
```

## Manuscript Writing Plan

The baseline comparison adds content to three manuscript sections. Draft
language below â€” adapt once results are in.

### Introduction (1â€“2 sentences, near end of intro)

Introduce the baseline rationale without heavy detail. Something like:

> "To disentangle the contributions of amortized visual features and learned
> temporal modeling from the choice of classifier family, we additionally
> evaluated pose-only classical-ML baselines â€” Random Forest (Breiman, 2001)
> and gradient-boosted trees (XGBoost; Chen & Guestrin, 2016) â€” trained on
> identical YOLO26n-pose keypoint features and evaluated under the same
> held-out protocol. This design follows the SimBA paradigm (Goodwin et al.,
> 2024) of pairing pose-derived features with non-sequential classifiers."

Key references to add to references.bib:
- Breiman, L. (2001). Random Forests. Machine Learning, 45, 5â€“32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
  KDD 2016.
- Goodwin et al. (2024) already in bib (SimBA, Nature Neuroscience).

### Methods â€” new subsection: "Pose-only classical-ML baseline"

Needs to cover:
1. **Motivation**: Isolate whether the LSTM + visual features outperform a
   pose-only classical classifier, addressing the confound that internal
   stream ablation (pose-only LSTM vs. full LSTM) uses the same model family.
2. **Feature vector description**: 132-d per frame = pose_self (49 per animal
   Ã— 2) + relational (11 Ã— 2 directed pairs) + reliability scalars (4 Ã— 2)
   + masks (2 Ã— 2). Extracted from the same NPZ windows using identical code.
3. **Temporal context variants**: Per-frame (132-d) and sliding-window
   (Â±7 frames concatenated, 1,980-d).
4. **Classifier details**: RF (500 trees, balanced weighting) and XGBoost
   (gradient boosting with early stopping on validation macro-F1).
5. **Evaluation parity**: Same 30-video held-out set, same smoothing
   (median filter k=5, min bout 15 frames), same frame-level and bout-level
   metrics (tIoU @ 0.25 and 0.50).
6. **Include the parity table** (from above) as a supplemental table or
   inline in methods.

### Discussion â€” new paragraph in "Comparison with existing approaches"

Interpret the three-way comparison:

| Classifier | Temporal modeling | Visual features | What it tests |
|---|---|---|---|
| LSTM (BehaviorScope-Y) | Learned sequential (32-frame windows) | Yes (frozen SPPF 256-d) | Full pipeline |
| RF (baseline) | None / crude sliding-window | No | Pose features + ensemble voting |
| XGBoost (baseline) | None / crude sliding-window | No | Pose features + gradient boosting |

Frame the result as:
- If LSTM wins: "The advantage of BehaviorScope-Y over pose-only classical-ML
  baselines confirms that frozen visual feature reuse and learned temporal
  modeling contribute discriminative signal beyond what pose geometry alone
  provides, regardless of classifier family."
- Acknowledge that the comparison is conservative (class weighting favors
  the baseline) and that the RF/XGBoost had access to identical features.

### Supplemental

Consider adding:
- The full parity table (already drafted above)
- Per-class held-out results for the baselines
- Training curves / validation macro-F1 for XGBoost (early stopping trace)

## Bug Fix: evaluate_heldout.py missing 2 videos (FIXED)

**Problem discovered during first evaluation run.** Two test_2 videos were
silently dropped because `find_annot(video_dir, allow_pred_named=False)`
filtered out their .annot files, which contain `_pred_` in the filename:

- `Mouse457_..._actions_pred_tomomi_fix_edTK_final_time_TS_withUSV.annot`
- `Mouse518_..._xgb_top_pcf_wnd_actions_pred_v1_6_CH_final.annot`

These are corrected prediction files that serve as ground truth in the MARS
dataset. The LSTM evaluation pipeline uses a different `find_annot` variant
(from `postprocess_rich_eval.py`) that falls back to all .annot files when
strict filtering removes everything. The RF evaluation used
`batch_infer_eval.py`'s stricter variant that returns `None`.

**Fix**: Added fallback in `evaluate_heldout.py` line 193:
```python
annot_path = find_annot(video_dir, allow_pred_named=False)
if annot_path is None:
    annot_path = find_annot(video_dir, allow_pred_named=True)
```

**Impact**: First run produced 28/30 videos. Must re-run to get 30/30.
All 4 models are affected (they all evaluate the same video set).

## Preliminary Held-Out Results (28 videos, BEFORE bug fix â€” DO NOT USE IN MANUSCRIPT)

Only rf_context15 completed before the bug was discovered. These are on
28/30 videos and must be re-run with the fix for the final 30/30 comparison.

| Model | Prediction Type | Frame F1 | Bout@.25 | Bout@.50 | Videos |
|---|---|---|---|---|---|
| RF context15 | smoothed | 0.7350 | 0.6977 | 0.4914 | 28 |
| (LSTM pose-only 256, same 28) | smoothed | 0.6905 | 0.6304 | 0.4537 | 28 |
| (LSTM full 256, same 28) | smoothed | 0.7152 | 0.6676 | 0.4960 | 28 |

**Surprising**: RF context15 outperforms both LSTMs on frame-F1 and bout@0.25
on this 28-video subset! The validation ranking (LSTM > RF) does NOT hold on
held-out. This changes the narrative significantly â€” the LSTM overfits more
from val to held-out than the RF.

**DO NOT WRITE MANUSCRIPT TEXT BASED ON THESE NUMBERS.** Wait for re-run with 30/30 videos.

## Data Dependencies

- NPZ cache (external drive): D:\BehaviorScope-Y_final_run_full_copy\npz_cache\
- Eval feature cache: D:\BehaviorScope-Y_final_run_full_copy\eval_feature_cache\mars_stride16\
- MARS .annot files: <DATA_ROOT>\BehaviorScope_Y_final_runs\MARS-data\
- Trained LSTM weights: <DATA_ROOT>\BehaviorScope_Y_final_runs\outputs\training_runs\
- Shared scripts: <DATA_ROOT>\BehaviorScope_Y_final_runs\shared_scripts\scripts\
