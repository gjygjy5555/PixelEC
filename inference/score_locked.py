import sys
import pathlib,json,concurrent.futures,multiprocessing,argparse
if not __package__:
 sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1] / 'code'))
import numpy as np
from PIL import Image
def init():
 import cv2
 cv2.setNumThreads(1)
 global calculate_psnr,calculate_ssim
 from pixelec.basicsr_metrics import calculate_psnr,calculate_ssim
def one(r):
 if not r.get('gt'):
  raise ValueError('Scoring requires a GT path for each prediction')
 if {'train', 'training'} & {part.lower() for part in pathlib.Path(r['gt']).resolve().parts}:
  raise ValueError('Scoring requires held-out ground truth')
 a=np.asarray(Image.open(r['pred']).convert('RGB'),dtype=np.float64);b=np.asarray(Image.open(r['gt']).convert('RGB'),dtype=np.float64);assert a.shape==b.shape
 return {'stem':pathlib.Path(r['lq']).stem,'type':r.get('type','unknown'),'size':[int(b.shape[1]),int(b.shape[0])],'psnr':float(calculate_psnr(a,b,0,input_order='HWC')),'ssim':float(calculate_ssim(a,b,0,input_order='HWC'))}
def write(p,d):
 t=p.with_suffix('.writing');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main():
 parser=argparse.ArgumentParser(description='RGB PSNR/SSIM: per-image arithmetic mean, crop0. Input directory must contain manifest.json.')
 parser.add_argument('legacy_path',nargs='?',help='Legacy positional results directory')
 parser.add_argument('legacy_out',nargs='?',help='Legacy positional metrics JSON')
 parser.add_argument('--path',help='Inference output directory containing manifest.json')
 parser.add_argument('--out','--output',dest='out',help='Output JSON; default: PATH/metrics.json')
 args=parser.parse_args()
 if args.path and args.legacy_path:parser.error('Use --path or positional path, not both')
 value=args.path or args.legacy_path
 if not value:parser.error('--path is required (or supply a legacy positional path)')
 if args.out and args.legacy_out:parser.error('Supply the output path only once')
 source=pathlib.Path(value).expanduser().resolve()
 if not (source/'manifest.json').is_file():parser.error('Missing manifest.json in '+str(source))
 out=pathlib.Path(args.out or args.legacy_out or source/'metrics.json').expanduser().resolve()
 out.parent.mkdir(parents=True,exist_ok=True)
 rows=json.loads((source/'manifest.json').read_text());per=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=init,mp_context=multiprocessing.get_context('spawn')) as pool:
  for x in pool.map(one,rows,chunksize=1):
   per.append(x)
   if len(per)%10==0:print('SCORED',len(per),'/',len(rows),flush=True)
 def agg(v):return {'n':len(v),'psnr':float(np.mean([x['psnr'] for x in v])),'ssim':float(np.mean([x['ssim'] for x in v]))}
 d={'n_unique_lq':len({x['lq'] for x in rows}),'all':agg(per),'per_image':per,'metric':'BasicSR RGB HWC crop0 float64 per-image then mean','protocol':'native pad-only one-step EMA','restored_dir':str(source)}
 for t in sorted({x['type'] for x in per}):d[t]=agg([x for x in per if x['type']==t])
 write(out,d);print('FULL_COMPLETE',d['all'],flush=True)

if __name__ == '__main__':
 main()
