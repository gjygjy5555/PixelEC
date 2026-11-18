"""Paper residual-gradient boundary ratio: RGB01, crop16, patch8, per-image mean."""
from pathlib import Path
import argparse
import json
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def boundary(prediction,gt):
    import numpy as np
    pred=np.asarray(prediction,dtype=np.float64)/255
    target=np.asarray(gt,dtype=np.float64)/255
    if pred.shape!=target.shape or min(pred.shape[:2])<=48:raise ValueError('Unaligned or too-small images')
    residual=(pred-target)[16:-16,16:-16];bnd=[];interior=[]
    for axis in [1,0]:
        difference=np.diff(residual,axis=axis);mask=np.arange(difference.shape[axis])%8==7
        bnd.append(float(np.square(np.take(difference,np.flatnonzero(mask),axis=axis)).mean()))
        interior.append(float(np.square(np.take(difference,np.flatnonzero(~mask),axis=axis)).mean()))
    eb=float(np.mean(bnd));ei=float(np.mean(interior))
    return dict(boundary_gradient_mse=eb,interior_gradient_mse=ei,boundary_to_interior_mse=eb/(ei+1e-12))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.path);out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    rows=json.loads((root/'manifest.json').read_text())
    if not rows or len({r['pred'] for r in rows})!=len(rows):raise ValueError('Empty/duplicated manifest')
    from PIL import Image
    import numpy as np
    out.parent.mkdir(parents=True,exist_ok=True);results=[]
    progress=out.with_name(out.stem+'.per_image.jsonl')
    with progress.open('x') as stream:
        for row in rows:
            if not row.get('gt'):
                raise ValueError('Scoring requires a GT path for each prediction')
            if {'train', 'training'} & {part.lower() for part in Path(row['gt']).resolve().parts}:
                raise ValueError('Scoring requires held-out ground truth')
            with Image.open(row['pred']) as im:pred=np.asarray(im.convert('RGB'))
            with Image.open(row['gt']) as im:gt=np.asarray(im.convert('RGB'))
            r=dict(lq=row['lq'],type=row.get('type','unknown'),**boundary(pred,gt));results.append(r)
            stream.write(json.dumps(r)+'\n');stream.flush()
    result=dict(n=len(results),Rbnd=float(np.mean([r['boundary_to_interior_mse'] for r in results])),
                boundary_gradient_mse=float(np.mean([r['boundary_gradient_mse'] for r in results])),
                interior_gradient_mse=float(np.mean([r['interior_gradient_mse'] for r in results])),
                protocol='Native saved RGB01, crop16 preserving phase, forward squared gradients, starts7mod8, equal H/V, arithmetic mean of per-image ratios',
                per_image=results)
    out.write_text(json.dumps(result,indent=2));print('BOUNDARY_COMPLETE',result['n'],result['Rbnd'])


if __name__=='__main__':main()
