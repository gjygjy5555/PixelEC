"""RGB PSNR/SSIM entry, with explicit pair-count and duplicate checks."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'code'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path',required=True)
    p.add_argument('--out')
    p.add_argument('--dataset',choices=['sice','msec','lcdp'])
    p.add_argument('--expected-count',type=int)
    a=p.parse_args();source=Path(a.path).expanduser().resolve()
    rows=json.loads((source/'manifest.json').read_text())
    if not rows:raise ValueError('Empty prediction manifest')
    for key in ['lq','pred']:
        if len({str(Path(r[key]).resolve()) for r in rows})!=len(rows):raise ValueError('Duplicated '+key)
    count=a.expected_count or ({'sice':60,'msec':5905,'lcdp':218}[a.dataset] if a.dataset else None)
    if count and len(rows)!=count:raise ValueError(f'Expected {count} predictions, got {len(rows)}')
    out=Path(a.out).expanduser().resolve() if a.out else source/'metrics.json'
    if out.exists():raise FileExistsError(out)
    from inference.score_locked import main as original_main
    sys.argv=['score_locked.py','--path',str(source),'--out',str(out)]
    original_main()


if __name__=='__main__':main()
