"""Create symlinks from the R7 test_eval output tree to the pre-converted MP4 cache."""
import os
from pathlib import Path

CACHE = Path(r".\data\data\manuscript_v5\test_eval\_source_mp4_cache")
OUTPUT = Path(r".\data\manuscript_v5\test_eval\R7_yolo_attn_v5_full_attention_head_mars_test_eval")

for split in ["test_1", "test_2"]:
    split_cache = CACHE / split
    if not split_cache.exists():
        print(f"[skip] {split_cache} not found")
        continue
    for mp4 in sorted(split_cache.glob("*.mp4")):
        video_id = mp4.stem
        target_dir = OUTPUT / split / video_id / "source_mp4"
        target_dir.mkdir(parents=True, exist_ok=True)
        link_path = target_dir / mp4.name
        if link_path.exists():
            print(f"[exists] {link_path}")
            continue
        os.symlink(str(mp4), str(link_path))
        print(f"[linked] {link_path} -> {mp4}")

print("\nDone. Re-run the test eval command — conversion will be skipped.")
