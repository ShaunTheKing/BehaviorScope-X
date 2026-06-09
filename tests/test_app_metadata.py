from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app_metadata  # noqa: E402


class AppMetadataTests(unittest.TestCase):
    def test_public_name_metadata_is_consistent(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(app_metadata.APP_NAME, "BehaviorScope-X")
        self.assertEqual(app_metadata.APP_SLUG, "behaviorscope_x")
        self.assertEqual(app_metadata.LOCAL_STATE_DIR, ".behaviorscope_x")
        self.assertEqual(pyproject["project"]["name"], "behaviorscope-x")
        self.assertEqual(pyproject["project"]["version"], app_metadata.APP_VERSION)
        self.assertEqual(pyproject["project"]["scripts"]["behaviorscope-x"], "behaviorscope_x_qt:main")
        self.assertEqual(pyproject["project"]["scripts"]["behaviorscope-y"], "behaviorscope_x_qt:main")

    def test_docs_use_public_name(self) -> None:
        for relative in [
            "README.md",
            "docs/index.md",
            "docs/gui-overview.md",
        ]:
            with self.subTest(relative=relative):
                text = (ROOT / relative).read_text(encoding="utf-8-sig")
                self.assertIn("BehaviorScope-X", text)


if __name__ == "__main__":
    unittest.main()
