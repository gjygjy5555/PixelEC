# PixelEC

| Directory | Contents                                                                |
| --- |-------------------------------------------------------------------------|
| `code/` | Model, flow matching, and vendored dependencies                         |
| `config/` | Training and inference configurations and dataset manifests |
| `train/` | Four-stage training program                                             |
| `inference/` | Inference, evaluation, and checkpoint verification                      |

## Installation

Run the commands from the repository root.

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r inference/requirements.txt
```

## Data and weights

Dataset downloads are available from the original project pages:

- [SICE-V2](https://github.com/KevinJ-Huang/ExposureNorm-Compensation#dataset)
- [MSEC](https://github.com/mahmoudnafifi/Exposure_Correction#dataset)
- [LCDP](https://github.com/onpix/LCDPNet#readme)

In the [JSONL manifests](config/manifests) for your dataset, replace each `lq` and `gt` value with the absolute path to the corresponding input and ground-truth image on your machine. 
Use the training manifests for training and `test.jsonl` for inference and scoring.


## Single-step inference and evaluation

Single-step inference and score:

```bash
python inference/infer.py --dataset sice --output runs/sice
python inference/score.py --dataset sice --path runs/sice
python inference/score_boundary.py --path runs/sice --out runs/sice/boundary.json
```

Our default seed offset is `42`.

## Training

```bash
python -m pip install -r train/requirements.txt
python train/train.py --dataset sice --stage 3 --check-only
python train/train.py --dataset sice --stage 3 --output runs/sice-stage3
```

## Acknowledgments

The model and training code build on PixRestore, and we use the DINOv2 encoder and adapts the BasicSR RGB metrics. [PixRestore](https://github.com/csslc/PixRestore) · [DINOv2](https://github.com/facebookresearch/dinov2) · [BasicSR](https://github.com/XPixelGroup/BasicSR)
