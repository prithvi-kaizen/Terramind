"""Fixed-checkpoint public-data evaluation; never trains or tunes thresholds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import ssl
import tarfile
import time
import urllib.parse
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
import torch
import torch.nn.functional as F
from methanepulse.runtime import predict_tile,sha256,cache_info
from methanepulse.evaluation import counts,scores

HF='https://huggingface.co/datasets/tacofoundation/methaneset/resolve/{revision}/methaneset-s2-finetune/'
ARCHIVES={'train':('1c666b3a16cfc9177979043d726d4ae2',3995740160),'val':('c147c1ec2f308c19b544f443790b2871',643686400),'test':('26390d808dd6577ca537dc60d659fecc',1100000000)}
BANDS=('B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12')


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def fetch(url, destination=None, byte_range=None):
    import certifi
    context=ssl.create_default_context(cafile=certifi.where())
    for attempt in range(4):
        try:
            headers={'User-Agent':'Terramind-reproducible-evaluation/1.0'}
            if byte_range:
                start,size=byte_range
                headers['Range']=f'bytes={start}-{start+size-1}'
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90,context=context) as response:
                if byte_range:
                    if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                        raise ValueError('Server did not honor the requested byte range')
                    payload=response.read(size+1)
                    if len(payload)!=size:
                        raise ValueError('Incomplete range response')
                    return payload
                if destination:
                    with open(destination,'wb') as out:
                        while chunk:=response.read(1024*1024):
                            out.write(chunk)
                    return
                return response.read()
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)


def summary(rows):
    accepted=[r for r in rows if 'error' not in r]
    total={key:sum(r[key] for r in accepted) for key in ('tp','fp','fn')}
    positives=[r for r in accepted if r['label_positive']]
    negatives=[r for r in accepted if not r['label_positive']]
    tp=sum(r['detected'] for r in positives);fp=sum(r['detected'] for r in negatives)
    # Scene detection alone does not establish mask/plume overlap.
    return {'count':len(rows),'evaluated':len(accepted),'failed':len(rows)-len(accepted),
        'pixels':{**total,**scores(total)} if accepted else None,'mean_tile_iou':float(np.mean([r['iou'] for r in accepted])) if accepted else None,
        'scene_positive_count':len(positives),'scene_negative_count':len(negatives),
        'scene_recall':tp/len(positives) if positives else None,
        'scene_precision':tp/(tp+fp) if negatives and tp+fp else None,
        'false_positive_scene_rate':fp/len(negatives) if negatives else None,
        'positive_scenes_with_any_mask_overlap':sum(r['tp']>0 for r in positives)}


def metadata(checkpoint):
    return {'timestamp_utc':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':sha256(checkpoint),
        'threshold':0.5,'device':'cpu','threads':torch.get_num_threads(),'training':'none; frozen existing checkpoint',
        'training_run_association':'unknown','metric_conventions':{'pixels':'global TP/FP/FN; both-empty score=1',
        'scene_detection':'any foreground pixel; not source localization or plume matching',
        'scene_precision':'null when no negative scenes','failed_samples':'listed and excluded from metric denominator'}}


def synthetic(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    root=Path(args.data);root.mkdir(parents=True,exist_ok=True)
    allrows=[];archives=[]
    splits=args.splits.split(',')
    if any(split not in ARCHIVES for split in splits):raise ValueError('Unknown published split')
    for split in splits:
        path=root/f'{split}.tar'
        url=f'https://zenodo.org/records/16813369/files/{split}.tar?download=1'
        if not path.exists(): fetch(url,path)
        digest=hashlib.md5()
        with path.open('rb') as f:
            while chunk:=f.read(1024*1024):digest.update(chunk)
        if digest.hexdigest()!=ARCHIVES[split][0]:
            path.unlink() # this task's incomplete/mismatched public download only
            raise ValueError('Archive checksum mismatch: '+split)
        archives.append({'split':split,'url':url,'md5':digest.hexdigest(),'bytes':path.stat().st_size})
        rows=[]
        with tarfile.open(path,'r:') as archive:
            members={Path(m.name).name:m for m in archive.getmembers() if m.isfile()}
            images=sorted(n for n in members if n.endswith('_nonorm_bands20m.tiff'))
            labels={re.sub(r'_bin_labels?\.tiff$','',n):m for n,m in members.items() if re.search(r'_bin_labels?\.tiff$',n)}
            for index,name in enumerate(images):
                row={'id':name,'split':split,'location':name.split('___')[1].split('_')[0],'plume_id':name.split('___')[0]}
                try:
                    tilebytes=archive.extractfile(members[name]).read()
                    labelbytes=archive.extractfile(labels[name.replace('_nonorm_bands20m.tiff','')]).read()
                    with MemoryFile(tilebytes) as tile,MemoryFile(labelbytes) as label, tile.open() as src,label.open() as lab:
                        if src.shape!=lab.shape or src.crs!=lab.crs or src.transform!=lab.transform:
                            raise ValueError('Misaligned ground-truth label')
                        target=lab.read(1)
                        if not np.isfinite(target).all() or not lab.read_masks(1).all():
                            raise ValueError('Invalid label pixels')
                        target=target>0.5
                        prediction=predict_tile(tile.name,args.checkpoint,0.5,'cpu')
                        item=counts(prediction.mask,target)
                    row.update(**item,**scores(item),label_positive=bool(target.any()),detected=bool(prediction.mask.any()),
                        tile_sha256=hashlib.sha256(tilebytes).hexdigest(),label_sha256=hashlib.sha256(labelbytes).hexdigest())
                except Exception as exc: row['error']=f'{type(exc).__name__}: {exc}'
                rows.append(row)
                if index%25==0:print(split,index+1,len(images),flush=True)
            if not images:raise ValueError('No images found in '+split)
        allrows.extend(rows)
        save(output/f'geoch4-{split}.json',{**metadata(args.checkpoint),'dataset':'GeoCH4PlumeNet v1',
            'source':'https://zenodo.org/records/16813369','label_conversion':'mask > 0.5',
            'interpretation':'Published split; training membership of this checkpoint is unknown. Not an independent generalization estimate.',
            'archive':archives[-1],'summary':summary(rows),'tiles':rows})
        if args.remove_archives:path.unlink()
    # Audit rather than relabel the provider's published split convention.
    leakage={}
    for field in ('location','plume_id'):
        splitgroups={s:{r[field] for r in allrows if r['split']==s} for s in splits}
        leakage[field]={f'{a}/{b}':sorted(splitgroups[a]&splitgroups[b]) for i,a in enumerate(splits) for b in splits[i+1:]}
    save(output/'geoch4-complete.json',{**metadata(args.checkpoint),'dataset':'GeoCH4PlumeNet v1','archives':archives,
        'summary':summary(allrows),'split_overlap_audit':leakage,
        'by_split':{s:summary([r for r in allrows if r['split']==s]) for s in splits},
        'historical_location_holdout':summary([r for r in allrows if r['location'] in ('8','9')]),
        'interpretation':'Descriptive checkpoint audit over all downloaded published splits, including training data; never report the combined score as held-out test accuracy.'})


def resize_prediction(payload, checkpoint):
    """Explicit OOD adapter: preserve footprint, resize to 512, score at native grid."""
    with MemoryFile(payload) as source,source.open() as src:
        if src.count!=13 or src.shape!=(200,200) or not src.read_masks().all():
            raise ValueError('Expected fully valid 13-band 200x200 Sentinel-2 chip')
        if not np.isfinite(src.read()).all():raise ValueError('Nonfinite source values')
        shape=src.shape;transform=src.transform;crs=src.crs
        data=src.read(out_shape=(13,512,512),resampling=Resampling.bilinear).astype('float32')
        profile=dict(driver='GTiff',height=512,width=512,count=13,dtype='float32',crs=crs,
            transform=transform*rasterio.Affine.scale(shape[1]/512,shape[0]/512))
        with MemoryFile() as adapted:
            with adapted.open(**profile) as dst:
                dst.write(data);dst.descriptions=BANDS
            prediction=predict_tile(adapted.name,checkpoint,.5,'cpu')
    native=F.interpolate(torch.from_numpy(prediction.probability.copy())[None,None],size=shape,mode='bilinear',align_corners=False)[0,0].numpy()
    return native>.5,crs,transform


def real(args):
    import pandas as pd
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    root=Path(args.data);root.mkdir(parents=True,exist_ok=True)
    api=json.loads(fetch('https://huggingface.co/api/datasets/tacofoundation/methaneset'))
    revision=api['sha'];base=HF.format(revision=revision)
    tables=[]
    for name in ('level0','level1'):
        path=root/f'{name}.parquet';fetch(base+f'.tacocat/{name}.parquet',path)
        tables.append(pd.read_parquet(path))
    parents,leaves=tables
    eligible=parents[(parents['split']=='test')&parents['target:sensor'].isin(['S2A','S2B'])].copy()
    eligible['rank']=eligible['id'].map(lambda value:hashlib.sha256(('terramind-real-v1:'+value).encode()).hexdigest())
    selected=eligible.sort_values('rank').head(args.real_samples)
    cohort=[{'id':r['id'],'country':r['site:country'],'site':r['site:location_name'],'split':r['split'],
        'sensor':r['target:sensor'],'date':str(r['stac:time_start']),'parent_index':int(r['internal:current_id'])} for _,r in selected.iterrows()]
    save(output/'real-cohort.json',{'dataset_revision':revision,'selection':'first N test Sentinel-2 IDs sorted by sha256(terramind-real-v1:+id); chosen before predictions','selected':cohort})
    jobs=[]
    for item in cohort:
        records=leaves[(leaves['internal:parent_id']==item['parent_index'])&leaves['id'].isin(['target','bg0','plume'])]
        if set(records['id'])!={'target','bg0','plume'}:raise ValueError('Incomplete catalog entry')
        for _,leaf in records.iterrows():
            jobs.append((item['id'],leaf['id'],base+urllib.parse.quote(leaf['internal:source_file']),int(leaf['internal:offset']),int(leaf['internal:size'])))
    def download(job):
        sample,kind,url,start,size=job
        payload=fetch(url,byte_range=(start,size))
        if payload[:2] not in (b'II',b'MM'):raise ValueError('Range is not a TIFF payload')
        path=root/(sample+'-'+kind+'.tiff');path.write_bytes(payload)
        return sample,kind,path
    files={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for sample,kind,path in pool.map(download,jobs):files[(sample,kind)]=path
    rows=[]
    for i,item in enumerate(cohort):
        labelpath=files[(item['id'],'plume')]
        with rasterio.open(labelpath) as label:
            target=label.read(1)
            if label.count!=1 or not np.isin(target,[0,1]).all() or not label.read_masks(1).all():raise ValueError('Invalid real mask')
            labelcrs,labeltransform=label.crs,label.transform
        for kind in ('target','bg0'):
            row={**item,'kind':kind}
            try:
                payload=files[(item['id'],kind)].read_bytes()
                predicted,crs,transform=resize_prediction(payload,args.checkpoint)
                if crs!=labelcrs or transform!=labeltransform:raise ValueError('Image/label alignment mismatch')
                truth=target.astype(bool) if kind=='target' else np.zeros_like(target,dtype=bool)
                if kind=='target' and not truth.any():raise ValueError('Expert mask is empty')
                value=counts(predicted,truth)
                row.update(**value,**scores(value),label_positive=bool(truth.any()),detected=bool(predicted.any()),
                    input_sha256=hashlib.sha256(payload).hexdigest(),label_sha256=sha256(labelpath) if kind=='target' else None)
            except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
            rows.append(row)
        print('real',i+1,len(cohort),flush=True)
    save(output/'real-results.json',{**metadata(args.checkpoint),'dataset':'MethaneSET s2-finetune','revision':revision,
        'source':'https://huggingface.co/datasets/tacofoundation/methaneset','license':'CC BY-NC-SA 4.0',
        'cohort':'real-cohort.json','adaptation':'OOD fixed footprint resize: 200x200 10m to 512x512 bilinear; percentiles on resized bands; probabilities returned bilinearly to native 200x200 before threshold 0.5; no tuning',
        'negatives':'provider-curated bg0 clean references; not independently reviewed by this project',
        'warnings':['The subdataset README has copied Landsat text; actual catalog sensors and TIFFs checked as 13-band Sentinel-2.',
            'Subset stress test, not the full real-data benchmark or an operational validation.','Real clouds, scaling, footprint, and spatial resolution differ from synthetic training.'],
        'summary':summary(rows),'tiles':rows,'cache':cache_info()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['synthetic','real'])
    parser.add_argument('--data',default='data/raw/evaluation')
    parser.add_argument('--output',default='outputs/public-evaluation')
    parser.add_argument('--checkpoint',default='terramind_methane_best.pt')
    parser.add_argument('--splits',default='train,val,test')
    parser.add_argument('--real-samples',type=int,default=64)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--remove-archives',action='store_true')
    args=parser.parse_args()
    if args.real_samples<1:parser.error('real-samples must be positive')
    torch.set_num_threads(args.threads)
    (synthetic if args.dataset=='synthetic' else real)(args)

if __name__=='__main__':main()
