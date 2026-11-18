"""Run a PixelEC training stage with explicit data, initialization and output."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelec.paths import CONFIG_DIR, ROOT, asset_path, data_path, prepare_environment

TRAINING_VENDOR = Path(__file__).resolve().parent / "engine/vendor"
TRAIN_SPLITS = {"train", "training"}
HELD_OUT_SPLITS = {"test", "testing", "val", "validation"}


def configuration(dataset, stage, path=None):
    path = Path(path).expanduser().resolve() if path else CONFIG_DIR / "train" / dataset / f"stage{stage}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("pretrained", "dinov2_checkpoint"):
        value = Path(config[key]).expanduser()
        config[key] = str(asset_path(value) if value.parts[:1] == ("weights",) else (ROOT / value).resolve())
    for key in ("dinov2_repository", "output_dir"):
        config[key] = str((ROOT / Path(config[key]).expanduser()).resolve())
    config["manifests"] = [str(data_path(value)) for value in config["manifests"]]
    return config


def bind_training_rows(config, folder):
    bound = []
    for index, manifest in enumerate(config["manifests"]):
        rows = []
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            keys = ("lq", "gt", "mask") if row.get("mask") else ("lq", "gt")
            for key in keys:
                path = data_path(row[key]).resolve()
                parts = {part.lower() for part in path.parts}
                if not parts & TRAIN_SPLITS or parts & HELD_OUT_SPLITS:
                    raise ValueError(f"Training paths must belong to a train partition: {path}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                row[key] = str(path)
            rows.append(row)
        if not rows:
            raise ValueError(f"Empty training manifest: {manifest}")
        target = folder / f"train_{index}.jsonl"
        target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        bound.append(str(target))
    config["manifests"] = bound


def install_training_imports():
    sys.path.insert(0, str(TRAINING_VENDOR))


def stage1(config_file):
    from safetensors.torch import load_file
    from train.engine.dino_memory import install_all_layer_dino_memory
    from train.engine import official_stage1 as official

    original_build = official.build_model
    official.build_model = lambda config: install_all_layer_dino_memory(original_build(config))

    def strict_load(model, ema, path):
        state = load_file(str(path))
        if "local_gate_logit_scale" in state and "local_gate_logit_scale" not in model.state_dict():
            state.pop("local_gate_logit_scale")
        model.load_state_dict(state, strict=True)
        ema.load_state_dict(model.state_dict(), strict=True)

    official.load_pretrained = strict_load
    sys.argv = [str(official.__file__), "--config", str(config_file)]
    official.main()


def check_runtime(stage):
    if stage == 1:
        from train.engine import official_stage1
    else:
        from accelerate import DataLoaderConfiguration
        from torchdata.stateful_dataloader import StatefulDataLoader
        from train.engine import stage23
        from train.engine import stage4
        from train.engine import train_base
        from train.engine import view_data

        if "use_stateful_dataloader" not in inspect.signature(DataLoaderConfiguration).parameters:
            raise RuntimeError("Accelerate must support stateful data loading")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sice", "msec", "lcdp"), required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--config", help="Override the dataset's portable JSON config")
    parser.add_argument("--output", help="Training output directory")
    parser.add_argument("--pretrained", help="EMA safetensors used to initialize the stage")
    parser.add_argument("--resume", help="Complete trusted checkpoint directory")
    parser.add_argument("--num-workers", type=int, help="Override the training config's worker count")
    parser.add_argument("--check-only", action="store_true", help="Check config and imports without loading images")
    args = parser.parse_args()

    config = configuration(args.dataset, args.stage, args.config)
    if args.output:
        config["output_dir"] = str(Path(args.output).expanduser().resolve())
    if args.pretrained:
        config["pretrained"] = str(Path(args.pretrained).expanduser().resolve())
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error("--num-workers cannot be negative")
        config["num_workers"] = args.num_workers
    if any(config.get(key) for key in ("test_lq_dir", "test_gt_dir", "eval_steps")):
        raise ValueError("Run test-set inference and scoring with the separate evaluation entry points")

    config["resume"] = str(Path(args.resume).expanduser().resolve()) if args.resume else None
    if config["resume"]:
        checkpoint = Path(config["resume"])
        required = {"model.safetensors", "ema_model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"}
        if args.stage >= 2:
            required.update(("custom_checkpoint_0.pkl", "custom_checkpoint_1.pkl"))
        missing = sorted(name for name in required if not (checkpoint / name).is_file())
        if missing:
            parser.error("Missing files in --resume checkpoint: " + ", ".join(missing))
        os.environ["PIXELEC_TRUSTED_CHECKPOINT"] = str(checkpoint)

    source = Path(config["pretrained"])
    required_paths = [source, Path(config["dinov2_checkpoint"]), Path(config["dinov2_repository"]) / "hubconf.py"]
    required_paths.extend(Path(manifest) for manifest in config["manifests"])
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}. See the README for dataset and weight setup.")

    prepare_environment()
    install_training_imports()
    if args.check_only:
        check_runtime(args.stage)
        print(json.dumps({
            "passed": True, "dataset": args.dataset, "stage": args.stage,
            "updates": config["max_steps"], "training_started": False, "dataset_images_read": False,
        }, indent=2))
        return

    if os.name == "nt" and args.stage >= 2 and config["num_workers"] > 0:
        parser.error("Use --num-workers 0 for Stages II–IV on Windows, or use the original worker count on Linux")

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=bool(config["resume"]))
    active = output / "runtime" / f"stage{args.stage}.yaml"
    active.parent.mkdir(exist_ok=True)
    bind_training_rows(config, active.parent)

    import yaml

    active.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if args.stage == 1:
        stage1(active)
        return

    from train.engine import train_base
    from train.engine.condition_audit import install_condition_audit
    from train.engine.loader_state import install as install_loader

    install_loader(train_base)
    install_condition_audit()
    if args.stage == 4:
        from train.engine.stage4 import run

        run(config, source, active)
    else:
        from train.engine.stage23 import run

        run(config, source, active, args.stage)


if __name__ == "__main__":
    main()
