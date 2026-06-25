# BehaviorScope-Y End-to-End Hardware Benchmark

This folder is portable. Move or copy the whole directory to another machine,
then run the benchmark from inside the copied folder. Scripts resolve paths
relative to this folder and do not require the original assembly-machine
project path.

## Included

- Three MARS held-out MP4 videos from `test_1`
- Matching `.annot` files for traceability
- MARS YOLO-pose checkpoint
- Two BehaviorScope-Y temporal classifiers:
  - `full_lstm256`, seed 42
  - `full_attention256`, seed 42
- Minimal inference scripts and utilities needed by `scripts/infer_y.py`
- Portable benchmark runner

## Requirements

Use a Python environment with the BehaviorScope-Y runtime dependencies:

- `torch`
- `ultralytics`
- `opencv-python`
- `numpy`
- `psutil` recommended for CPU/RAM telemetry
- `pynvml` or `nvidia-smi` recommended for NVIDIA GPU telemetry
- `ffmpeg` optional for the predecode/cache comparison mode

On Apple Silicon, install a PyTorch build with MPS support.

## Windows Examples

CUDA:

```powershell
cd <this-folder>
powershell -ExecutionPolicy Bypass -File .\run_benchmark.ps1 -Device cuda:0 -Amp auto
```

CPU:

```powershell
cd <this-folder>
powershell -ExecutionPolicy Bypass -File .\run_benchmark.ps1 -Device cpu -Amp off
```

Dry run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_benchmark.ps1 -DryRun
```

Optional FFmpeg predecode/cache mode on NVIDIA:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_benchmark.ps1 -Device cuda:0 -Amp auto -DecodeBackend ffmpeg-predecode -FfmpegHwaccel cuda -FfmpegEncoder h264_nvenc -FfmpegPreset p1
```

## macOS / Linux Examples

Apple Silicon MPS:

```bash
cd <this-folder>
bash run_benchmark.sh --device mps --amp off
```

CPU:

```bash
cd <this-folder>
bash run_benchmark.sh --device cpu --amp off
```

NVIDIA CUDA:

```bash
cd <this-folder>
bash run_benchmark.sh --device cuda:0 --amp auto
```

Optional FFmpeg predecode/cache mode on Apple Silicon:

```bash
bash run_benchmark.sh --device mps --amp off --decode-backend ffmpeg-predecode --ffmpeg-hwaccel videotoolbox --ffmpeg-encoder h264_videotoolbox
```

## Decode Modes

The default `opencv` mode is the manuscript-style baseline: the Python runner
passes each MP4 to the local `scripts/infer_y.py` path, which loads the
YOLO-pose checkpoint and temporal sequence classifier and runs full-video
inference.

The optional `ffmpeg-predecode` mode first creates a local cached MP4 under
`outputs/video_cache/ffmpeg_predecode/`, then runs the same Python/PyTorch
inference path on that cached video. The cache build time is reported as
`predecode_wall_time_s` and is separate from `wall_time_s` and
`end_to_end_fps`. This mode is useful for testing whether video preparation is
contributing to low throughput on a particular machine, but it is not a
zero-copy NVDEC or VideoToolbox frame provider.

## Outputs

Results are written inside the bundle:

```text
outputs/benchmarks/benchmark_plan.json
outputs/benchmarks/benchmark_summary.csv
outputs/benchmarks/<model>/rep01/test_1/<video_id>/
```

Each video/model run writes:

- behavior prediction CSV
- sidecar runtime metrics CSV from `infer_y.py`
- stdout/stderr logs
- `command.json` recording the exact local command

Key summary fields:

- `wall_time_s`
- `end_to_end_fps`
- `mean_instant_fps`
- `median_instant_fps`
- `mean_cpu_percent_normalized`
- `max_cpu_rss_mb`
- `mean_gpu_util_percent`
- `max_gpu_mem_used_mb`
- `decode_backend`
- `source_video_path`
- `inference_video_path`
- `predecode_wall_time_s`
- `mean_pose_crop_next_ms`
- `mean_feature_encode_ms`
- `mean_window_assembly_ms`
- `mean_classify_ms`
- `mean_csv_write_ms`

The `pose_crop_next` stage is the upstream frame path: video decode,
Ultralytics pose/tracking, GPU-to-CPU result transfer, and crop construction.
If this field dominates on a discrete GPU workstation while MPS is faster, the
main bottleneck is likely host/device transfer or host-side preprocessing
rather than the temporal sequence classifier.

For MPS, GPU utilization/memory columns may be blank because PyTorch and macOS
do not expose the same portable counters as NVIDIA/NVML. CPU/RAM and end-to-end
FPS are still recorded.
