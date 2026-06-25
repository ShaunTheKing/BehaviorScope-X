#!/usr/bin/env python3
"""Release launcher for the BehaviorScope-Y Qt application."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = Path(__file__).resolve().parents[1]


def resolve_app() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "behaviorscope_y_qt.py",
        PROJECT_ROOT / "shared_scripts" / "scripts" / "behaviorscope_y_qt.py",
        RELEASE_ROOT / "supporting_scripts" / "shared_scripts" / "scripts" / "behaviorscope_y_qt.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find behaviorscope_y_qt.py in project or release supporting scripts.")


def main() -> int:
    app = resolve_app()
    cmd = [sys.executable, str(app), *sys.argv[1:]]
    if app.parent.name == "gui_application":
        cwd = RELEASE_ROOT
    else:
        cwd = app.parents[2]
    return subprocess.run(cmd, cwd=str(cwd)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
