"""Validation-only pilot ablations; one model pass per scene, filters never promoted here."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
import torch
import torch.nn.functional as F
from methanepulse.candidates import CandidateConfig,extract_candidates,FILTER_FLAGS
from methanepulse.runtime import predict_tile,cache_info,sha256
from methanepulse.evaluation import counts,scores
from scripts.evaluate_public_data import HF,fetch,select_catalog_leaves,summary,BANDS


def infer_native(payload,checkpoint):
    with MemoryFile(payload) as mem,mem.open() as source:
        if source.count!=13 or source.shape!=(200,200) or not source.read_masks().all():
            raise ValueError('Expected fully valid 13-band 200x200 Sentinel-2 chip')
        data=source.read(out_shape=(13,512,512),resampling=Resampling.bilinear).astype('float32')
        if not np.isfinite(data).all():raise ValueError('Nonfinite source')
        crs,transform=source.crs,source.transform
        with MemoryFile() as adapted:
            with adapted.open(driver='GTiff',count=13,height=512,width=512,dtype='float32',crs=crs,
                transform=transform*rasterio.Affine.scale(200/512,200/512)) as dst:
                dst.write(data);dst.descriptions=BANDS
            prediction=predict_tile(adapted.name,checkpoint,.5,'cpu')
    probability=F.interpolate(torch.from_numpy(prediction.probability.copy())[None,None],size=(200,200),mode='bilinear',align_corners=False)[0,0].numpy()
    return probability,crs,transform


def paired_intervals(rows,baseline,variant,repetitions=1000):
    groups=sorted({(r['country'],r['site']) for r in rows})
    rng=np.random.default_rng(42)
    metrics=[]
    for _ in range(repetitions):
        sampled=rng.integers(0,len(groups),len(groups))
        selected=[r for index in sampled for r in rows if (r['country'],r['site'])==groups[index]]
        positive=[r for r in selected if r['label_positive']];negative=[r for r in selected if not r['label_positive']]
        if not positive or not negative:continue
        loss=np.mean([r['variants'][baseline]['detected']-r['variants'][variant]['detected'] for r in positive])
        delta=np.mean([r['variants'][variant]['detected']-r['variants'][baseline]['detected'] for r in negative])
        metrics.append([loss,delta])
    return {'site_clusters':len(groups),'method':'paired site-cluster bootstrap, 1000 resamples, seed 42; limited geographic/site support',
        'scene_recall_loss_95pct':np.quantile(np.array(metrics)[:,0],[.025,.975]).tolist() if metrics else None,
        'false_positive_rate_change_95pct':np.quantile(np.array(metrics)[:,1],[.025,.975]).tolist() if metrics else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--data',default='data/raw/phase2-validation')
    parser.add_argument('--output',default='docs/phase-2-validation.json')
    parser.add_argument('--checkpoint',default='terramind_methane_best.pt')
    args=parser.parse_args()
    if args.samples<1:parser.error("samples must be positive")
    torch.set_num_threads(2)
    root=Path(args.data);root.mkdir(parents=True,exist_ok=True)
    revision=json.loads(Path('docs/evaluation/real-results.json').read_text())['revision']
    base=HF.format(revision=revision)
    tables=[]
    for level in ('level0','level1'):
        path=root/(level+'.parquet')
        if not path.exists():fetch(base+f'.tacocat/{level}.parquet',path)
        tables.append(pd.read_parquet(path))
    parents,leaves=tables
    eligible=parents[(parents['split']=='validation') & parents['target:sensor'].isin(['S2A','S2B'])].copy()
    eligible['rank']=eligible.id.map(lambda value:hashlib.sha256(('terramind-phase2-validation-v1:'+value).encode()).hexdigest())
    selected=eligible.sort_values('rank').head(args.samples)
    test_ids={r['id'] for r in json.loads(Path('docs/evaluation/real-cohort.json').read_text())['selected']}
    cohort=[{'id':r['id'],'country':r['site:country'],'site':r['site:location_name'],'source_file':r['internal:source_file'],
        'parent_index':int(r['internal:current_id']),'split':'validation'} for _,r in selected.iterrows()]
    assert not test_ids & {r['id'] for r in cohort}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    cohort_path=output.with_name(output.stem+'-cohort.json')
    cohort_path.write_text(json.dumps({'dataset_revision':revision,'selection':'first N validation IDs ranked by sha256(terramind-phase2-validation-v1:+id), before predictions','selected':cohort},indent=2)+'\n')
    records=leaves[['id','internal:parent_id','internal:source_file','internal:offset','internal:size']].to_dict('records')
    jobs=[]
    for item in cohort:
        for leaf in select_catalog_leaves(records,item['parent_index'],item['source_file']):
            jobs.append((item['id'],leaf))
    def acquire(job):
        sample,leaf=job;path=root/(sample+'-'+leaf['id']+'.tiff')
        if not path.exists() or path.stat().st_size!=int(leaf['internal:size']):
            payload=fetch(base+urllib.parse.quote(leaf['internal:source_file']),byte_range=(int(leaf['internal:offset']),int(leaf['internal:size'])))
            path.write_bytes(payload)
        return sample,leaf['id'],path
    files={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for sample,kind,path in pool.map(acquire,jobs):files[(sample,kind)]=path
    config=CandidateConfig()
    rules={'raw':(),**{flag:(flag,) for flag in FILTER_FLAGS},'combined':FILTER_FLAGS}
    rows=[];failures=[]
    for index,item in enumerate(cohort):
        with rasterio.open(files[(item['id'],'plume')]) as label:
            values=label.read(1)
            if label.count!=1 or not np.isin(values,[0,1]).all() or not label.read_masks(1).all():raise ValueError('Ground truth must be a fully valid binary mask')
            truth=values>0;label_crs,label_transform=label.crs,label.transform
        for kind in ('target','bg0'):
            try:
                payload=files[(item['id'],kind)].read_bytes()
                probability,crs,transform=infer_native(payload,args.checkpoint)
                if crs!=label_crs or transform!=label_transform:raise ValueError('Image/label alignment mismatch')
                if kind=='target' and not truth.any():raise ValueError('Expected a positive expert mask')
                target=truth if kind=='target' else np.zeros_like(truth)
                review=extract_candidates(probability,transform,crs,config=config)
                variants={}
                for name,flags in rules.items():
                    keep=np.ones(len(review['candidates'])+1,dtype=bool);keep[0]=False
                    for candidate in review['candidates']:
                        if set(candidate['quality_flags']) & set(flags):keep[candidate['label_id']]=False
                    prediction=keep[review['labels']]
                    values=counts(prediction,target)
                    variants[name]={**values,**scores(values),'detected':int(prediction.any())}
                rows.append({**item,'kind':kind,'label_positive':bool(target.any()),'input_sha256':hashlib.sha256(payload).hexdigest(),
                    'label_sha256':sha256(files[(item['id'],'plume')]) if kind=='target' else None,
                    'candidate_count':len(review['candidates']),'variants':variants})
            except Exception as exc:failures.append({**item,'kind':kind,'error':str(exc)})
        print('validation',index+1,len(cohort),flush=True)
    results={}
    for name in rules:
        converted=[{**r,**r['variants'][name]} for r in rows]
        results[name]={'summary':summary(converted)}
        if name!='raw':results[name]['paired_comparison']=paired_intervals(rows,'raw',name)
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'dataset':'MethaneSET s2-finetune validation subset',
        'dataset_revision':revision,'checkpoint_sha256':sha256(args.checkpoint),'config':config.manifest(.5),
        'cohort':cohort_path.name,'threshold':.5,'adaptation':'same fixed 200→512→200 probability adapter as the previous external stress test; areas measured geodesically on native grid',
        'selected_pairs':len(cohort),'scored_inputs':len(rows),'failures':failures,'cache':cache_info(),
        'default_filter_flags':[],'promotion':'None. Pilot evidence on an adapted, limited-site validation cohort; keep all rules informational until deployment-matched validation supports promotion.',
        'recall_loss_tolerance_absolute':.02,'rules':{k:list(v) for k,v in rules.items()},'results':results,'inputs':rows}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v['summary'] for k,v in results.items()},indent=2))

if __name__=='__main__':main()
