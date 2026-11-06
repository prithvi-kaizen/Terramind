"""Phase 3 research tooling. Every extra model pass is opt-in."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='terramind_methane_best.pt')
    commands=parser.add_subparsers(dest='command',required=True)
    train=commands.add_parser('train')
    train.add_argument('--manifest',required=True)
    train.add_argument('--directory',required=True)
    train.add_argument('--epochs',type=int,default=1)
    train.add_argument('--max-seconds',type=float,default=600)
    train.add_argument('--repeat-selected',action='store_true')
    train.add_argument('--without-negatives',action='store_true')
    temporal=commands.add_parser('temporal')
    temporal.add_argument('--manifest',required=True)
    temporal.add_argument('--cache',required=True)
    temporal.add_argument('--tolerance-m',type=float,default=200)
    temporal.add_argument('--output',required=True)
    consistency=commands.add_parser('consistency')
    consistency.add_argument('--tile',required=True)
    consistency.add_argument('--enable',action='store_true')
    consistency.add_argument('--output',required=True)
    batch=commands.add_parser('batch-consistency')
    batch.add_argument('--manifest',required=True)
    batch.add_argument('--mode',choices=['off','automatic'],default='off')
    batch.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='train':
        from methanepulse.training import train
        result=train(args.manifest,args.checkpoint,args.directory,args.epochs,args.max_seconds,
                     repeat_selected=args.repeat_selected,include_negatives=not args.without_negatives)
    elif args.command=='temporal':
        from methanepulse.reliability import temporal_review
        result=temporal_review(args.manifest,args.checkpoint,args.cache,tolerance_m=args.tolerance_m)
    elif args.command=='batch-consistency':
        from methanepulse.reliability import consistency_batch
        manifest_path=Path(args.manifest).resolve()
        rows=json.loads(manifest_path.read_text())['tiles']
        for row in rows: row['tile']=str((manifest_path.parent/row['tile']).resolve())
        result=consistency_batch(rows,args.checkpoint,args.mode)
    else:
        from methanepulse.reliability import consistency_check
        result=consistency_check(args.tile,args.checkpoint,enabled=args.enable)
    if args.command!='train':
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
