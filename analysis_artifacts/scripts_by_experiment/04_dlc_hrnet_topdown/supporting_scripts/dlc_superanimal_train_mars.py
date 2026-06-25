"""Fine-tune DLC SuperAnimal TopViewMouse on the converted MARS DLC project."""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path


warnings.filterwarnings(
    "ignore",
    message=".*NaturalNameWarning.*",
)


def touch_marker(state_dir: Path, name: str, payload: dict | None = None) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{name}.json").write_text(
        json.dumps(payload or {"status": "complete"}, indent=2),
        encoding="utf-8",
    )


def marker_exists(state_dir: Path, name: str) -> bool:
    return (state_dir / f"{name}.json").is_file()


def read_marker(state_dir: Path, name: str) -> dict:
    path = state_dir / f"{name}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def should_skip_marker(state_dir: Path, name: str) -> bool:
    payload = read_marker(state_dir, name)
    return bool(payload) and payload.get("status", "complete") != "invalid"


def expected_pytorch_config(config_path: str, shuffle: int, auxiliaryfunctions, deeplabcut) -> Path:
    cfg = auxiliaryfunctions.read_config(config_path)
    train_fraction = float(cfg["TrainingFraction"][0])
    model_folder = auxiliaryfunctions.get_model_folder(
        train_fraction,
        int(shuffle),
        cfg,
        engine=deeplabcut.Engine.PYTORCH,
    )
    return Path(cfg["project_path"]) / model_folder / "train" / "pytorch_config.yaml"


def model_config_ready(config_path: str, shuffle: int, auxiliaryfunctions, deeplabcut) -> bool:
    return expected_pytorch_config(config_path, shuffle, auxiliaryfunctions, deeplabcut).is_file()


def parse_snapshot_epoch(snapshot_path: Path) -> int:
    match = re.search(r"-(\d+)\.pt$", snapshot_path.name)
    if not match:
        raise ValueError(f"Could not parse epoch from snapshot name: {snapshot_path}")
    return int(match.group(1))


def list_pose_snapshots(model_train_dir: Path) -> list[Path]:
    snapshots = [
        path
        for path in model_train_dir.glob("snapshot*.pt")
        if re.match(r"^snapshot(-best)?-\d+\.pt$", path.name)
    ]
    return sorted(snapshots, key=parse_snapshot_epoch)


def latest_pose_snapshot(model_train_dir: Path) -> Path | None:
    snapshots = list_pose_snapshots(model_train_dir)
    if not snapshots:
        return None
    return snapshots[-1]


def best_pose_snapshot(model_train_dir: Path) -> Path | None:
    snapshots = [
        path
        for path in list_pose_snapshots(model_train_dir)
        if "-best-" in path.name
    ]
    if not snapshots:
        return None
    return snapshots[-1]


def pose_snapshot_path(model_train_dir: Path, epoch: int, *, best: bool) -> Path:
    label = f"best-{epoch:03}" if best else f"{epoch:03}"
    return model_train_dir / f"snapshot-{label}.pt"


def read_snapshot_metric(snapshot_path: Path, metric_key: str) -> float | None:
    import torch

    try:
        state = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(snapshot_path, map_location="cpu")
    metadata = state.get("metadata", {})
    metrics = metadata.get("metrics", {})
    value = metrics.get(metric_key)
    if value is None:
        return None
    return float(value)


def best_metric_pose_snapshot(model_train_dir: Path, metric_key: str) -> tuple[Path | None, float | None, int | None]:
    rows: list[tuple[float, int, Path]] = []
    for snapshot in list_pose_snapshots(model_train_dir):
        metric = read_snapshot_metric(snapshot, metric_key)
        if metric is not None:
            rows.append((metric, parse_snapshot_epoch(snapshot), snapshot))
    if not rows:
        return None, None, None
    metric, epoch, path = max(rows, key=lambda item: (item[0], item[1]))
    return path, metric, epoch


def relabel_true_best_pose_snapshot(model_train_dir: Path, metric_key: str) -> tuple[Path | None, float | None, int | None]:
    _, best_metric, best_epoch = best_metric_pose_snapshot(model_train_dir, metric_key)
    if best_epoch is None:
        return best_pose_snapshot(model_train_dir), None, None

    for snapshot in list_pose_snapshots(model_train_dir):
        epoch = parse_snapshot_epoch(snapshot)
        if "-best-" in snapshot.name and epoch != best_epoch:
            regular_target = pose_snapshot_path(model_train_dir, epoch, best=False)
            if not regular_target.exists():
                snapshot.rename(regular_target)

    regular_best = pose_snapshot_path(model_train_dir, best_epoch, best=False)
    best_target = pose_snapshot_path(model_train_dir, best_epoch, best=True)
    if regular_best.exists() and not best_target.exists():
        regular_best.rename(best_target)

    return best_target if best_target.exists() else None, best_metric, best_epoch


def detector_snapshots(model_train_dir: Path) -> list[Path]:
    snapshots = [
        path
        for path in model_train_dir.glob("snapshot-detector*.pt")
        if re.match(r"^snapshot-detector(-best)?-\d+\.pt$", path.name)
    ]
    return sorted(snapshots, key=parse_snapshot_epoch)


def best_detector_snapshot(model_train_dir: Path) -> Path | None:
    snapshots = [path for path in detector_snapshots(model_train_dir) if "-best-" in path.name]
    if snapshots:
        return snapshots[-1]
    snapshots = detector_snapshots(model_train_dir)
    return snapshots[-1] if snapshots else None


def train_detector(
    *,
    deeplabcut,
    config_path: str,
    train_config: Path,
    args: argparse.Namespace,
) -> dict:
    if int(args.detector_epochs) <= 0:
        return {"status": "skipped", "reason": "detector_epochs <= 0"}

    model_train_dir = train_config.parent
    best_pose_before, best_pose_metric_before, best_pose_epoch_before = relabel_true_best_pose_snapshot(
        model_train_dir,
        str(args.early_stop_metric),
    )
    pose_before = {
        path.name: {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in list_pose_snapshots(model_train_dir)
    }
    print("[dlc] train detector", flush=True)
    deeplabcut.train_network(
        config_path,
        detector_epochs=int(args.detector_epochs),
        detector_batch_size=int(args.detector_batch_size),
        detector_save_epochs=int(args.detector_save_epochs),
        epochs=0,
        save_epochs=int(args.save_epochs),
        batch_size=int(args.batch_size),
        displayiters=int(args.displayiters),
        shuffle=int(args.finetune_shuffle),
        device=args.device,
        max_snapshots_to_keep=int(args.max_snapshots_to_keep),
        pytorch_cfg_updates={
            "detector.runner.eval_interval": int(args.detector_eval_interval),
            "detector.runner.key_metric": "test.mAP@50:95",
            "detector.runner.key_metric_asc": True,
            "detector.runner.snapshots.save_optimizer_state": True,
            "detector.train_settings.dataloader_workers": int(args.dataloader_workers),
            "detector.train_settings.dataloader_pin_memory": bool(args.dataloader_pin_memory),
        },
    )
    best_pose_after, best_pose_metric_after, best_pose_epoch_after = relabel_true_best_pose_snapshot(
        model_train_dir,
        str(args.early_stop_metric),
    )
    pose_after = {
        path.name: {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in list_pose_snapshots(model_train_dir)
    }
    if pose_before != pose_after:
        raise RuntimeError(
            "Detector training changed the pose snapshot set. Refusing to continue "
            "because pose checkpoint preservation was required."
        )
    if best_pose_before != best_pose_after or best_pose_metric_before != best_pose_metric_after:
        raise RuntimeError(
            "Detector training changed the selected best pose checkpoint. Refusing to continue."
        )
    best = best_detector_snapshot(model_train_dir)
    return {
        "status": "complete",
        "detector_epochs": int(args.detector_epochs),
        "detector_batch_size": int(args.detector_batch_size),
        "detector_eval_interval": int(args.detector_eval_interval),
        "detector_save_epochs": int(args.detector_save_epochs),
        "best_detector_snapshot": str(best) if best else None,
        "detector_snapshots": [str(path) for path in detector_snapshots(model_train_dir)],
        "pose_checkpoint_guard": {
            "best_pose_snapshot": str(best_pose_after) if best_pose_after else None,
            "best_pose_epoch": best_pose_epoch_after,
            "best_pose_metric": best_pose_metric_after,
            "unchanged": True,
        },
    }


def train_with_optional_early_stopping(
    *,
    deeplabcut,
    config_path: str,
    train_config: Path,
    args: argparse.Namespace,
) -> dict:
    model_train_dir = train_config.parent
    metric_key = str(args.early_stop_metric)
    patience = int(args.early_stop_patience)
    max_epochs = int(args.epochs)
    eval_interval = int(args.eval_interval)
    min_delta = float(args.early_stop_min_delta)
    completed_epoch = 0
    pytorch_cfg_updates = {
        "runner.eval_interval": eval_interval,
        "runner.key_metric": "test.mAP",
        "runner.key_metric_asc": True,
        "train_settings.dataloader_workers": int(args.dataloader_workers),
        "train_settings.dataloader_pin_memory": bool(args.dataloader_pin_memory),
        "detector.train_settings.dataloader_workers": int(args.dataloader_workers),
        "detector.train_settings.dataloader_pin_memory": bool(args.dataloader_pin_memory),
    }
    initial_latest = latest_pose_snapshot(model_train_dir)
    if initial_latest is not None:
        completed_epoch = parse_snapshot_epoch(initial_latest)

    if patience <= 0:
        print("[dlc] train_network", flush=True)
        deeplabcut.train_network(
            config_path,
            detector_epochs=int(args.detector_epochs),
            epochs=max_epochs,
            save_epochs=int(args.save_epochs),
            batch_size=int(args.batch_size),
            displayiters=int(args.displayiters),
            shuffle=int(args.finetune_shuffle),
            device=args.device,
            max_snapshots_to_keep=int(args.max_snapshots_to_keep),
            pytorch_cfg_updates=pytorch_cfg_updates,
        )
        latest = latest_pose_snapshot(model_train_dir)
        best, best_metric, best_epoch = relabel_true_best_pose_snapshot(model_train_dir, metric_key)
        return {
            "status": "complete",
            "stopped_early": False,
            "epochs_requested": max_epochs,
            "epochs_completed": parse_snapshot_epoch(latest) if latest else None,
            "best_snapshot": str(best) if best else None,
            "best_epoch": best_epoch,
            "best_metric_key": metric_key,
            "best_metric": best_metric,
        }

    best = best_pose_snapshot(model_train_dir)
    best_metric = read_snapshot_metric(best, metric_key) if best else None
    best_epoch = parse_snapshot_epoch(best) if best else None
    stale_evaluations = 0

    print(
        "[dlc] train_network with early stopping "
        f"(metric={metric_key}, patience={patience}, eval_interval={eval_interval})",
        flush=True,
    )
    while completed_epoch < max_epochs:
        snapshot = latest_pose_snapshot(model_train_dir)
        snapshot_arg = str(snapshot) if snapshot is not None else None
        kwargs = {
            "detector_epochs": int(args.detector_epochs),
            "epochs": 1,
            "save_epochs": int(args.save_epochs),
            "batch_size": int(args.batch_size),
            "displayiters": int(args.displayiters),
            "shuffle": int(args.finetune_shuffle),
            "device": args.device,
            "max_snapshots_to_keep": int(args.max_snapshots_to_keep),
            "pytorch_cfg_updates": {
                **pytorch_cfg_updates,
                "runner.snapshots.save_optimizer_state": True,
            },
        }
        if snapshot_arg is not None:
            kwargs["snapshot_path"] = snapshot_arg

        deeplabcut.train_network(config_path, **kwargs)

        latest = latest_pose_snapshot(model_train_dir)
        if latest is None:
            raise FileNotFoundError(f"DLC did not create a pose snapshot in {model_train_dir}")
        completed_epoch = parse_snapshot_epoch(latest)

        current_best, current_metric, current_best_epoch = relabel_true_best_pose_snapshot(
            model_train_dir,
            metric_key,
        )
        if current_metric is not None and (
            best_metric is None or current_metric > best_metric + min_delta
        ):
            best_metric = current_metric
            best_epoch = current_best_epoch
            stale_evaluations = 0
            print(
                f"[dlc] new best {metric_key}={best_metric:.4f} at epoch {best_epoch}",
                flush=True,
            )
        elif completed_epoch % eval_interval == 0 and current_metric is not None:
            stale_evaluations += 1
            print(
                f"[dlc] no {metric_key} improvement for "
                f"{stale_evaluations}/{patience} validation epochs "
                f"(best={best_metric:.4f} at epoch {best_epoch})",
                flush=True,
            )

        touch_marker(
            Path(args.project_root) / "pipeline_state" / "dlc_training",
            "train_progress",
            {
                "status": "running",
                "epochs_requested": max_epochs,
                "epochs_completed": completed_epoch,
                "early_stop_metric": metric_key,
                "early_stop_patience": patience,
                "dataloader_workers": int(args.dataloader_workers),
                "dataloader_pin_memory": bool(args.dataloader_pin_memory),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "best_snapshot": str(current_best) if current_best else None,
                "stale_validation_epochs": stale_evaluations,
            },
        )
        if stale_evaluations >= patience:
            print(
                f"[dlc] early stopping at epoch {completed_epoch}; "
                f"best {metric_key}={best_metric:.4f} at epoch {best_epoch}",
                flush=True,
            )
            break

    final_best, final_metric, final_epoch = relabel_true_best_pose_snapshot(model_train_dir, metric_key)
    return {
        "status": "complete",
        "stopped_early": stale_evaluations >= patience,
        "epochs_requested": max_epochs,
        "epochs_completed": completed_epoch,
        "early_stop_metric": metric_key,
        "early_stop_patience": patience,
        "best_metric": final_metric,
        "best_epoch": final_epoch,
        "best_snapshot": str(final_best) if final_best else None,
        "max_snapshots_to_keep": int(args.max_snapshots_to_keep),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--project_root", required=True)
    p.add_argument("--conversion_table_csv", default="")
    p.add_argument("--scorer", default="BehaviorScopeY")
    p.add_argument("--superanimal_name", default="superanimal_topviewmouse")
    p.add_argument("--model_name", default="hrnet_w32")
    p.add_argument("--detector_name", default="fasterrcnn_resnet50_fpn_v2")
    p.add_argument("--base_shuffle", type=int, default=0)
    p.add_argument("--finetune_shuffle", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--detector_epochs", type=int, default=0)
    p.add_argument("--detector_batch_size", type=int, default=2)
    p.add_argument("--detector_save_epochs", type=int, default=1)
    p.add_argument("--detector_eval_interval", type=int, default=1)
    p.add_argument("--save_epochs", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--early_stop_metric", default="metrics/test.mAP")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--max_snapshots_to_keep", type=int, default=5)
    p.add_argument("--dataloader_workers", type=int, default=4)
    p.add_argument("--dataloader_pin_memory", action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--displayiters", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--stages",
        nargs="+",
        default=["convert", "check", "base_dataset", "finetune_dataset", "train", "evaluate"],
        choices=[
            "convert",
            "check",
            "base_dataset",
            "finetune_dataset",
            "detector_train",
            "train",
            "evaluate",
        ],
    )
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root)
    state_dir = project_root / "pipeline_state" / "dlc_training"
    config_path = str(Path(args.config_path))

    import deeplabcut
    import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
    from deeplabcut.modelzoo import build_weight_init
    from deeplabcut.modelzoo.utils import create_conversion_table, read_conversion_table_from_csv

    if "convert" in args.stages and not (args.resume and should_skip_marker(state_dir, "convert")):
        print("[dlc] convertcsv2h5", flush=True)
        deeplabcut.convertcsv2h5(config_path, userfeedback=False, scorer=args.scorer)
        touch_marker(state_dir, "convert", {"config_path": config_path, "scorer": args.scorer})

    if args.conversion_table_csv:
        conversion_table = Path(args.conversion_table_csv)
        if conversion_table.is_file():
            print(f"[dlc] applying conversion table: {conversion_table}", flush=True)
            create_conversion_table(
                config=config_path,
                super_animal=args.superanimal_name,
                project_to_super_animal=read_conversion_table_from_csv(conversion_table),
            )
        else:
            raise FileNotFoundError(f"conversion table not found: {conversion_table}")

    if "check" in args.stages and not (args.resume and should_skip_marker(state_dir, "check")):
        print("[dlc] check_labels", flush=True)
        deeplabcut.check_labels(config_path, visualizeindividuals=True)
        touch_marker(state_dir, "check", {"config_path": config_path})

    base_ready = model_config_ready(config_path, int(args.base_shuffle), auxiliaryfunctions, deeplabcut)
    if "base_dataset" in args.stages and not (args.resume and should_skip_marker(state_dir, "base_dataset") and base_ready):
        if args.resume and marker_exists(state_dir, "base_dataset") and not base_ready:
            print(
                "[dlc] base_dataset marker exists, but expected PyTorch config is missing; rebuilding.",
                flush=True,
            )
        print("[dlc] create base PyTorch training dataset", flush=True)
        deeplabcut.create_training_dataset(
            config_path,
            Shuffles=[int(args.base_shuffle)],
            net_type=f"top_down_{args.model_name}",
            detector_type=args.detector_name,
            engine=deeplabcut.Engine.PYTORCH,
            userfeedback=False,
        )
        base_config = expected_pytorch_config(config_path, int(args.base_shuffle), auxiliaryfunctions, deeplabcut)
        if not base_config.is_file():
            raise FileNotFoundError(f"Base dataset did not create expected PyTorch config: {base_config}")
        touch_marker(state_dir, "base_dataset", {"shuffle": int(args.base_shuffle), "pytorch_config": str(base_config)})

    finetune_ready = model_config_ready(config_path, int(args.finetune_shuffle), auxiliaryfunctions, deeplabcut)
    if "finetune_dataset" in args.stages and not (args.resume and should_skip_marker(state_dir, "finetune_dataset") and finetune_ready):
        if args.resume and marker_exists(state_dir, "finetune_dataset") and not finetune_ready:
            print(
                "[dlc] finetune_dataset marker exists, but expected PyTorch config is missing; rebuilding.",
                flush=True,
            )
        print("[dlc] build SuperAnimal weight init", flush=True)
        weight_init = build_weight_init(
            cfg=auxiliaryfunctions.read_config(config_path),
            super_animal=args.superanimal_name,
            model_name=args.model_name,
            detector_name=args.detector_name,
            with_decoder=True,
        )
        print("[dlc] create SuperAnimal fine-tune dataset from existing split", flush=True)
        deeplabcut.create_training_dataset_from_existing_split(
            config_path,
            from_shuffle=int(args.base_shuffle),
            shuffles=[int(args.finetune_shuffle)],
            engine=deeplabcut.Engine.PYTORCH,
            net_type=f"top_down_{args.model_name}",
            detector_type=args.detector_name,
            weight_init=weight_init,
            userfeedback=False,
        )
        finetune_config = expected_pytorch_config(config_path, int(args.finetune_shuffle), auxiliaryfunctions, deeplabcut)
        if not finetune_config.is_file():
            raise FileNotFoundError(f"Fine-tune dataset did not create expected PyTorch config: {finetune_config}")
        touch_marker(
            state_dir,
            "finetune_dataset",
            {
                "from_shuffle": int(args.base_shuffle),
                "shuffle": int(args.finetune_shuffle),
                "superanimal_name": args.superanimal_name,
                "model_name": args.model_name,
                "detector_name": args.detector_name,
                "pytorch_config": str(finetune_config),
            },
        )

    if "train" in args.stages and not (args.resume and should_skip_marker(state_dir, "train")):
        train_config = expected_pytorch_config(config_path, int(args.finetune_shuffle), auxiliaryfunctions, deeplabcut)
        if not train_config.is_file():
            raise FileNotFoundError(
                f"Cannot train because fine-tune PyTorch config is missing: {train_config}. "
                "Run with --stages base_dataset finetune_dataset train, or remove the stale "
                "finetune_dataset marker."
            )
        print("[dlc] train_network", flush=True)
        train_payload = train_with_optional_early_stopping(
            deeplabcut=deeplabcut,
            config_path=config_path,
            train_config=train_config,
            args=args,
        )
        train_payload["shuffle"] = int(args.finetune_shuffle)
        touch_marker(state_dir, "train", train_payload)

    if "detector_train" in args.stages and not (args.resume and should_skip_marker(state_dir, "detector_train")):
        train_config = expected_pytorch_config(config_path, int(args.finetune_shuffle), auxiliaryfunctions, deeplabcut)
        if not train_config.is_file():
            raise FileNotFoundError(
                f"Cannot train detector because fine-tune PyTorch config is missing: {train_config}. "
                "Run with --stages base_dataset finetune_dataset detector_train."
            )
        detector_payload = train_detector(
            deeplabcut=deeplabcut,
            config_path=config_path,
            train_config=train_config,
            args=args,
        )
        detector_payload["shuffle"] = int(args.finetune_shuffle)
        touch_marker(state_dir, "detector_train", detector_payload)

    if "evaluate" in args.stages and not (args.resume and should_skip_marker(state_dir, "evaluate")):
        print("[dlc] evaluate_network", flush=True)
        deeplabcut.evaluate_network(
            config_path,
            Shuffles=[int(args.finetune_shuffle)],
            snapshotindex="best",
            detector_snapshot_index="best",
            pcutoff=0.6,
        )
        touch_marker(state_dir, "evaluate", {"shuffle": int(args.finetune_shuffle)})

    print("[dlc] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
