"""Bounded, CPU-first decoder experiments on reviewed, leak-checked manifests."""
import copy
import json
import math
from pathlib import Path
import platform
import random
import resource
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .evaluation import load_manifest, read_target, counts, scores
from .features import FeatureCache
from .runtime import _build_model, sha256, ARCHITECTURE_VERSION
from .preprocessing import load_and_preprocess, PREPROCESSING_VERSION

CONFIGURATIONS = (
    {'name':'bce-lr1e-4', 'lr':1e-4, 'dice_weight':0.},
    {'name':'bce-lr3e-4', 'lr':3e-4, 'dice_weight':0.},
    {'name':'bce-dice-lr1e-4', 'lr':1e-4, 'dice_weight':.5},
)


def reviewed_manifest(path):
    manifest = load_manifest(path)
    for row in manifest['tiles']:
        if not row.get('reviewer') or not row.get('review_evidence') or row.get('review_status')!='accepted':
            raise ValueError('Every label needs accepted review, reviewer and evidence')
        if row['split'] in ('train','validation'):
            target = read_target(row['mask'],row['tile'])
            if not target.any() and not row['verified_negative']:
                raise ValueError('Empty training/validation labels require verified negatives')
            if target.any() and row['verified_negative']:
                raise ValueError('Verified negative contains foreground')
    for split in ('train','validation'):
        rows = [r for r in manifest['tiles'] if r['split']==split]
        if not rows or not any(r['verified_negative'] for r in rows) or not any(not r['verified_negative'] for r in rows):
            raise ValueError('Training and validation each require reviewed positives and negatives')
    return manifest


def metrics(predictions, rows, targets):
    total = dict(tp=0,fp=0,fn=0)
    positives = detected = negatives = false_alarms = overlaps = 0
    tiles=[]
    for prediction,row in zip(predictions,rows):
        target = targets[row['id']]
        item = counts(prediction,target)
        tiles.append(dict(id=row['id'],location=row['location'],positive=bool(target.any()),
                          detected=bool(prediction.any()),**item,**scores(item)))
        for key in total:
            total[key] += item[key]
        if target.any():
            positives += 1
            detected += int(prediction.any())
            overlaps += int((prediction & target).any())
        else:
            negatives += 1
            false_alarms += int(prediction.any())
    return dict(pixels={**total,**scores(total)}, scene_recall=detected/positives if positives else None,
                overlap_recall=overlaps/positives if positives else None,
                false_positive_scene_rate=false_alarms/negatives if negatives else None,
                scene_precision=detected/(detected+false_alarms) if detected+false_alarms else None,
                positives=positives, verified_negatives=negatives, tiles=tiles)


def paired_intervals(baseline, candidate, replicates=500):
    sites=sorted({r['location'] for r in baseline['tiles']})
    rng=np.random.default_rng(42)
    deltas={'scene_recall':[], 'false_positive_scene_rate':[]}
    for _ in range(replicates):
        chosen=rng.choice(sites,len(sites),replace=True)
        rates=[]
        for report in (baseline,candidate):
            rows=[row for site in chosen for row in report['tiles'] if row['location']==site]
            positives=[r for r in rows if r['positive']]; negatives=[r for r in rows if not r['positive']]
            rates.append(dict(scene_recall=np.mean([r['detected'] for r in positives]) if positives else None,
                              false_positive_scene_rate=np.mean([r['detected'] for r in negatives]) if negatives else None))
        for key in deltas:
            if rates[0][key] is not None and rates[1][key] is not None:
                deltas[key].append(rates[1][key]-rates[0][key])
    return dict(method='paired site bootstrap; candidate minus baseline',replicates=replicates,seed=42,
                site_count=len(sites),intervals={key:dict(valid_replicates=len(values),
                    percentile_95=np.quantile(values,[.025,.975]).tolist() if values else None) for key,values in deltas.items()})


def promotion_gate(baseline, candidate):
    if any(baseline.get(k) is None or candidate.get(k) is None for k in ('scene_recall','false_positive_scene_rate')):
        return False
    return (candidate['false_positive_scene_rate'] < baseline['false_positive_scene_rate'] and
            candidate['scene_recall'] >= baseline['scene_recall']-.02)


def _train_experiment(manifest_path, checkpoint, directory, epochs=1, max_seconds=600, seed=42, repeat_selected=False, include_negatives=True):
    if type(epochs) is not int or not 1<=epochs<=10 or not math.isfinite(max_seconds) or not 1<=max_seconds<=7200:
        raise ValueError('Budget: 1–10 epochs and 1–7200 seconds')
    manifest = reviewed_manifest(manifest_path)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Experiment directory must be empty to preserve prior evidence')
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'resolved-manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    start = time.perf_counter()
    cpu_start = time.process_time()
    def budget():
        if time.perf_counter()-start > max_seconds:
            raise TimeoutError('Training wall-time budget exhausted; no checkpoint is promoted')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = _build_model(checkpoint, torch.device('cpu'))
    cache = FeatureCache(directory/'features',checkpoint)
    rows = [r for r in manifest['tiles'] if r['split'] in ('train','validation')]
    targets = {}
    # Batch-one on-disk features; tensors are not all retained in memory.
    for row in rows:
        budget()
        targets[row['id']] = read_target(row['mask'],row['tile'])
        cache.get(row['tile'],model)
    training = [r for r in rows if r['split']=='train' and (include_negatives or not r['verified_negative'])]
    validation = [r for r in rows if r['split']=='validation']
    initial_decoder = copy.deepcopy(model.seg_decoder.state_dict())
    def infer(rows):
        model.eval()
        predictions=[]
        for row in rows:
            budget()
            with torch.no_grad():
                logits,_ = model.decode(cache.get(row['tile'],model))
            predictions.append(torch.sigmoid(logits).squeeze().numpy()>.5)
        return metrics(predictions,rows,targets)
    baseline = infer(validation)
    swir=[]
    for row in validation:
        budget()
        x,*_=load_and_preprocess(row['tile'],'cpu')
        swir.append((torch.abs(x[0,0]-x[0,1])>.2).numpy())
    swir_baseline=metrics(swir,validation,targets)
    results=[]
    def run(configuration,run_seed):
        torch.manual_seed(run_seed)
        model.seg_decoder.load_state_dict(initial_decoder)
        for p in model.encoder.parameters(): p.requires_grad_(False)
        for p in model.severity_head.parameters(): p.requires_grad_(False)
        optimizer=torch.optim.Adam(model.seg_decoder.parameters(),lr=configuration['lr'])
        history=[]; best=-1.; best_state=None; best_metrics=None
        run_start=time.perf_counter()
        for epoch in range(epochs):
            model.train(); model.encoder.eval(); model.severity_head.eval()
            indices=list(range(len(training))); random.Random(run_seed+epoch).shuffle(indices)
            loss_sum=0.
            for index in indices:
                budget(); row=training[index]
                feature=cache.get(row['tile'],model)
                target=torch.tensor(targets[row['id']],dtype=torch.float32)[None,None]
                optimizer.zero_grad(set_to_none=True)
                logits,_=model.decode(feature)
                probability=torch.sigmoid(logits)
                dice=1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
                loss=F.binary_cross_entropy_with_logits(logits,target)+configuration['dice_weight']*dice
                if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
                loss.backward(); optimizer.step(); loss_sum+=float(loss.detach())
            measured=infer(validation)
            history.append(dict(epoch=epoch+1,loss=loss_sum/len(training),validation=measured))
            if measured['pixels']['iou']>best:
                best=measured['pixels']['iou']; best_state=copy.deepcopy(model.seg_decoder.state_dict()); best_metrics=measured
        model.seg_decoder.load_state_dict(best_state)
        name=f"{configuration['name']}-seed{run_seed}"
        target_path=directory/(name+'.pt')
        torch.save(model.state_dict(),target_path)
        result=dict(configuration=configuration,seed=run_seed,history=history,best_validation=best_metrics,
                    checkpoint=target_path.name,checkpoint_sha256=sha256(target_path),
                    validation_gate=promotion_gate(baseline,best_metrics),seconds=time.perf_counter()-run_start)
        result['paired_validation_intervals']=paired_intervals(baseline,best_metrics)
        results.append(result)
        # Preserve completed experiments even if a later configuration exhausts budget.
        (directory/'completed-runs.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        return result
    try:
        search=[run(config,seed) for config in CONFIGURATIONS]
        selected=max(search,key=lambda result:result['best_validation']['pixels']['iou'])
        if repeat_selected:
            for extra_seed in (seed+1,seed+2): run(selected['configuration'],extra_seed)
        status='completed'
    except TimeoutError:
        status='budget_exhausted'; selected=None
    elapsed=time.perf_counter()-start
    report=dict(schema_version=1,status=status,manifest_sha256=sha256(manifest_path),
                architecture=ARCHITECTURE_VERSION,preprocessing=PREPROCESSING_VERSION,
                environment=dict(platform=platform.platform(),python=platform.python_version(),
                                 torch=torch.__version__,numpy=np.__version__),
                baseline_checkpoint_sha256=sha256(checkpoint),threshold=.5,threshold_selection='fixed before experiments',
                baseline_validation=baseline,runs=results,selected=selected,
                swir_baseline=dict(method='absolute independently percentile-normalised B11/B12 difference > 0.2; fixed, not methane-specific',validation=swir_baseline),
                release_decision='retain_existing_checkpoint_pending_independent_test_and_uncertainty_review',
                includes_training_negatives=include_negatives,epochs=epochs,seed=seed,
                resources=dict(device='cpu',threads=2,wall_seconds=elapsed,cpu_wall_hours=elapsed/3600,
                               process_cpu_seconds=time.process_time()-cpu_start,
                               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),**cache.report()),
                test_labels_used=False,severity='unchanged; independent severity labels absent',
                limitations='Validation gate is a point estimate, not release approval. Evaluate a frozen selection on independent test data with site-level uncertainty. Negative ablation requires a separate fresh run using the same split/seed/configuration budgets.')
    (directory/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def train(manifest_path, checkpoint, directory, epochs=1, max_seconds=600, seed=42, repeat_selected=False, include_negatives=True):
    started=time.perf_counter()
    try:
        return _train_experiment(manifest_path,checkpoint,directory,epochs,max_seconds,seed,repeat_selected,include_negatives)
    except TimeoutError:
        directory=Path(directory)
        completed=directory/'completed-runs.json'
        report=dict(schema_version=1,status='budget_exhausted_during_preparation',
                    manifest_sha256=sha256(manifest_path),baseline_checkpoint_sha256=sha256(checkpoint),
                    runs=json.loads(completed.read_text()) if completed.exists() else [],
                    wall_seconds=time.perf_counter()-started,max_seconds=max_seconds,
                    release_decision='retain_existing_checkpoint',test_labels_used=False)
        (directory/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
