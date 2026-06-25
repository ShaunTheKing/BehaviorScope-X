from pathlib import Path
import numpy as np


PATHS = {
    "yolo_sppf": Path("outputs/npz_cache/mars_full_video/yolo_feature_cache"),
    "mobilenetv3": Path("outputs/controlled_comparison_runs/run_20260528_205727/feature_caches/mobilenetv3_native/heldout"),
    "dlc_hrnet": Path("outputs/npz_cache/mars_dlc_topdown_hrnet_feature_cache_heldout"),
}


def main():
    for route, root in PATHS.items():
        files = sorted(root.glob("*.npz"))
        print(f"\n## {route}")
        print(f"root={root}")
        print(f"n_files={len(files)}")
        if not files:
            continue
        print(f"sample={files[0].name}")
        with np.load(files[0], allow_pickle=False) as z:
            for key in z.files:
                arr = z[key]
                print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
