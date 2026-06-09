from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal


class TutorialDownloadWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, *, repo_id: str, local_dir: Path):
        super().__init__()
        self.repo_id = repo_id
        self.local_dir = local_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            try:
                from huggingface_hub import HfApi, hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "The Hugging Face download helper is not installed. "
                    "Install project dependencies with: pip install -r requirements.txt"
                ) from exc

            self.local_dir.mkdir(parents=True, exist_ok=True)
            api = HfApi()
            files = [
                path
                for path in api.list_repo_files(self.repo_id, repo_type="dataset")
                if path and not path.endswith("/")
            ]
            if not files:
                raise RuntimeError(f"No files found in Hugging Face dataset: {self.repo_id}")
            total = len(files)
            for index, filename in enumerate(files, start=1):
                if self._cancelled:
                    raise RuntimeError("Tutorial download was cancelled.")
                self.progress.emit(index - 1, total, filename)
                kwargs = {
                    "repo_id": self.repo_id,
                    "repo_type": "dataset",
                    "filename": filename,
                    "local_dir": str(self.local_dir),
                }
                try:
                    hf_hub_download(local_dir_use_symlinks=False, **kwargs)
                except TypeError:
                    hf_hub_download(**kwargs)
                self.progress.emit(index, total, filename)
            self.finished.emit(str(self.local_dir))
        except Exception as exc:
            self.failed.emit(str(exc))
