"""Repository, dataset and weight locations shared by both programs."""
from pathlib import Path
import os

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
CONFIG_DIR = ROOT / "config"
DINO_REPOSITORY = CODE_DIR / "vendor/dinov2_repo"
DINO_CHECKPOINT = "weights/dinov2_vits14_pretrain.pth"


def asset_path(relative):
    assets = Path(os.environ.get("PIXELEC_ASSETS_ROOT", ROOT)).expanduser()
    relative = Path(relative)
    path = assets / relative
    if path.is_file() or relative.parts[:1] != ("weights",):
        return path.resolve()

    # Also accept weights from the original delivery without moving the files.
    legacy = assets / "main_project" / relative
    return legacy.resolve() if legacy.is_file() else path.resolve()


def data_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    data_root = os.environ.get("PIXELEC_DATA_ROOT")
    if data_root and path.parts[:1] == ("datasets",):
        return Path(data_root).expanduser().joinpath(*path.parts[1:]).resolve()
    return (ROOT / path).resolve()


def prepare_environment():
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    cache = os.environ.get("PIXELEC_CACHE_DIR")
    if cache:
        root = Path(cache).expanduser().resolve()
        for key in ("TMPDIR", "TMP", "TEMP", "TORCH_HOME", "HF_HOME",
                    "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR",
                    "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR"):
            if key not in os.environ or key in ("TMPDIR", "TMP", "TEMP"):
                directory = root / key.lower()
                directory.mkdir(parents=True, exist_ok=True)
                os.environ[key] = str(directory)
