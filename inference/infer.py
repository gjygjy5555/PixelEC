"""Run single-step PixelEC inference at native resolution."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from inference.runtime import DATASETS, build_model, load_config, run_inference
from pixelec.paths import ROOT, asset_path, prepare_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True, help="Select the final EMA for a dataset")
    parser.add_argument("--input", help="Ordered JSONL, defaults to the bundled test manifest")
    parser.add_argument("--output", help="Output directory for restored images")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", type=int, help="CUDA-visible device index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true", help="Strict-load weights on CPU without opening images")
    args = parser.parse_args()

    if not args.verify_only and not args.output:
        parser.error("--output is required for inference")

    config = load_config(args.dataset)
    checkpoint = asset_path(config["weight"])
    if args.verify_only:
        prepare_environment()
        model, _ = build_model(config, checkpoint, verify_hash=config["sha256"])
        print("STRICT_LOAD_PASS", args.dataset, sum(p.numel() for p in model.parameters()))
        return

    run_inference(
        config, checkpoint, args.input or ROOT / config["test_manifest"], args.output,
        device=args.device, gpu=args.gpu, seed=args.seed, start_index=args.start_index,
        verify_hash=config["sha256"],
    )


if __name__ == "__main__":
    main()
