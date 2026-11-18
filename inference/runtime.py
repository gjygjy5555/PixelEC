"""Model selection and native-resolution inference."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from pixelec.paths import (
    CONFIG_DIR, DINO_CHECKPOINT, DINO_REPOSITORY, asset_path, data_path,
    prepare_environment,
)

DATASETS = ("sice", "msec", "lcdp")


def load_config(dataset):
    return json.loads((CONFIG_DIR / "inference" / f"{dataset}.json").read_text(encoding="utf-8"))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def resolve_config(config):
    cfg=dict(config)
    cfg.setdefault('fusion', 'all_layers')
    cfg.setdefault('head', 'nearest_rgb3')
    cfg['dinov2_repository'] = str(DINO_REPOSITORY)
    cfg['dinov2_checkpoint'] = str(asset_path(DINO_CHECKPOINT))
    return SimpleNamespace(**cfg)


def build_model(config, checkpoint, *, verify_hash=None):
    from safetensors.torch import load_file
    from pixelec.config import build_model as construct
    cfg = resolve_config(config)
    path=Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {path}. See the README for weight archives and PIXELEC_ASSETS_ROOT.")
    if verify_hash and digest(path) != verify_hash:
        raise RuntimeError(f"EMA SHA256 mismatch: {path}")
    state=load_file(str(path),device='cpu')
    model = construct(cfg, state=state)
    model.load_state_dict(state,strict=True)
    del state
    model.eval().requires_grad_(False)
    if sum(x.numel() for x in model.final_layer.parameters())!=1215619:
        raise RuntimeError('Unexpected RGB readout shape')
    return model, cfg


def pad_native(image):
    import numpy as np
    from PIL import Image

    w, h = image.size
    right, bottom = (-w) % 8, (-h) % 8
    if right or bottom:
        image = Image.fromarray(np.pad(np.asarray(image), ((0,bottom),(0,right),(0,0)), mode="reflect"))
    return image, right, bottom


def run_inference(config, checkpoint, input_manifest, output, *, device='cuda',
                  gpu=None, seed=42, start_index=0, verify_hash=None):
    out=Path(output).expanduser().resolve()
    if out.exists():raise FileExistsError(out)
    if gpu is not None and (device == "cpu" or gpu < 0):
        raise ValueError("--gpu requires CUDA and a nonnegative visible device index")
    if start_index < 0:
        raise ValueError("--start-index must be nonnegative")

    prepare_environment()
    import torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor,to_pil_image
    from accelerate.utils import set_seed
    from pixelec.objective import build_flow
    from vendor.pixrestore.vision import load_dinov2,extract_layers
    model, cfg = build_model(config, checkpoint, verify_hash=verify_hash)
    dev = torch.device(device, gpu) if gpu is not None else torch.device(device)
    model.to(dev)
    enc=load_dinov2(cfg.encoder_type,dev,repository=cfg.dinov2_repository,checkpoint=cfg.dinov2_checkpoint)
    flow=build_flow(cfg,type('Device',(),{'device':dev})())
    src_manifest=Path(input_manifest).expanduser()
    rows=[json.loads(s) for s in src_manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows or len({data_path(r['lq']).resolve() for r in rows})!=len(rows):raise ValueError('Empty or duplicated LQ manifest')
    for row in rows:
        src=data_path(row['lq'])
        if {'train','training'}&{x.lower() for x in src.resolve().parts}:raise ValueError('Inference refuses training images')
        if not src.is_file():raise FileNotFoundError(src)
    out.mkdir(parents=True);saved=[]
    for i,row in enumerate(rows):
        src=data_path(row['lq'])
        with Image.open(src) as im:im=im.convert('RGB')
        w,h=im.size;pad,right,bottom=pad_native(im)
        # This device-side normalization and PNG conversion match the locked runtime.
        x=pil_to_tensor(pad)[None].to(dev).float().div_(127.5).sub_(1)
        model.set_observation(x,None);set_seed(seed+start_index+i)
        with torch.no_grad(),torch.autocast(dev.type,dtype=torch.bfloat16,enabled=dev.type=='cuda'):
            z=extract_layers(enc,x,cfg.encoder_layers,cfg.encoder_input_size)
            y=flow.sample_multistep_fm(model,x,venc_fea=z,n_steps=1,schedule='linear')
        image=to_pil_image(y[0].float().cpu().add(1).mul(.5).clamp(0,1)).crop((0,0,w,h))
        pred=out/f'{start_index+i:05d}_{src.stem}.png';image.save(pred)
        gt=row.get('gt');gt=str(data_path(gt).resolve()) if gt else gt
        saved.append(dict(lq=str(src),gt=gt,pred=str(pred),type=row.get('type','unknown'),seed=seed+start_index+i,size=[w,h],pad_rb=[right,bottom]))
        print('INFERRED', i + 1, len(rows), dev, flush=True)
        model.last_rgb=None
    (out/'manifest.json').write_text(json.dumps(saved,indent=2))
    protocol=dict(checkpoint=str(checkpoint),sha256=verify_hash or digest(checkpoint),n=len(rows),FM_offset=seed,
                  start_index=start_index,seed_rule='offset + global image index',strict_EMA=True,
                  one_step=True,native_pad_only=True,GT_reads=False,device=str(dev),
                  inference_rope='dynamic')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    return protocol
