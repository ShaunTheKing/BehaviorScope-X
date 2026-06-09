from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from make_ethogram_summary import summarize_csv  # noqa: E402


class EthogramSummaryTests(unittest.TestCase):
    def test_frame_csv_builds_behavior_summary_and_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video_a.smoothed_frames.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["frame_idx", "frame_start", "frame_end", "predicted_class_id", "predicted_class"],
                )
                writer.writeheader()
                for frame, label in enumerate(["other", "attack", "attack", "other", "mount"]):
                    writer.writerow(
                        {
                            "frame_idx": frame,
                            "frame_start": frame,
                            "frame_end": frame,
                            "predicted_class_id": {"other": 0, "attack": 1, "mount": 2}[label],
                            "predicted_class": label,
                        }
                    )

            segments, summary, metadata = summarize_csv(
                path,
                fps=10.0,
                include_background=False,
                background_label="other",
            )

        self.assertEqual(metadata["total_frames"], 5)
        self.assertEqual([(row["behavior"], row["start_frame"], row["end_frame"]) for row in segments], [
            ("attack", 1, 2),
            ("mount", 4, 4),
        ])
        by_behavior = {row["behavior"]: row for row in summary}
        self.assertEqual(by_behavior["attack"]["frames"], 2)
        self.assertEqual(by_behavior["attack"]["bouts"], 1)
        self.assertEqual(by_behavior["mount"]["seconds"], 0.1)


if __name__ == "__main__":
    unittest.main()
