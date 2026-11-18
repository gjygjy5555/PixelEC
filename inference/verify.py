"""Verify PixelEC checkpoints on CPU with optional synthetic forward passes."""
import argparse
import gc
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from inference.runtime import DATASETS, build_model, load_config
from pixelec.paths import CODE_DIR, DINO_CHECKPOINT, DINO_REPOSITORY, asset_path, prepare_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/verification.json"))
    args = parser.parse_args()

    prepare_environment()
    import torch
    from vendor import pixrestore
    from vendor.pixrestore.vision import load_dinov2

    torch.set_num_threads(2)
    if not Path(pixrestore.__file__).resolve().is_relative_to(CODE_DIR / "vendor/pixrestore"):
        raise RuntimeError("Unexpected PixRestore import")
    encoder = load_dinov2(
        "dinov2s", torch.device("cpu"), repository=str(DINO_REPOSITORY),
        checkpoint=str(asset_path(DINO_CHECKPOINT)),
    )
    dino_parameters = sum(p.numel() for p in encoder.parameters())
    del encoder

    results = []
    for dataset in DATASETS:
        name = f"{dataset}_main"
        config = load_config(dataset)
        model, _ = build_model(config, asset_path(config["weight"]), verify_hash=config["sha256"])
        verification = {
            "case": name,
            "strict_load": True,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        if args.synthetic:
            torch.manual_seed(123)
            lq = torch.randn(1, 3, 16, 24)
            model.set_observation(lq, None)
            inputs = torch.cat((lq, torch.randn_like(lq)), dim=1)
            features = [torch.randn(1, 4, 384) for _ in range(6)]
            with torch.no_grad():
                rgb, _ = model(inputs, torch.ones(1), z=features)
            if rgb.shape != (1, 3, 16, 24) or not torch.isfinite(rgb).all():
                raise RuntimeError(f"Synthetic forward failed for {name}")
            verification.update(synthetic_forward_shape=list(rgb.shape), synthetic_forward_finite=True)
        results.append(verification)
        print("VERIFIED", name, flush=True)
        del model
        gc.collect()

    report = {
        "passed": True,
        "cases": results,
        "dinov2_parameters": dino_parameters,
        "dataset_images_read": False,
        "training_started": False,
        "cuda_used": False,
    }
    output = args.out.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ALL_RUNTIME_CHECKS_PASS", len(results), flush=True)


if __name__ == "__main__":
    main()
