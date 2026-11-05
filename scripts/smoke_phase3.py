"""Real CPU feature/cache/gradient checks and an opt-in flip review benchmark."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from methanepulse.features import FeatureCache
from methanepulse.preprocessing import load_and_preprocess
from methanepulse import runtime
from methanepulse.reliability import consistency_check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/phase3-smoke.json')
    parser.add_argument('--benchmark-runs',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.benchmark_runs<=30: raise ValueError('Use at most 30 warm runs')
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    sample=next((root/'sample_input').glob('*.tiff')); checkpoint=root/'terramind_methane_best.pt'
    started=time.perf_counter()
    model=runtime._build_model(checkpoint,torch.device('cpu'))
    with tempfile.TemporaryDirectory(prefix='terramind-phase3-') as directory:
        cache=FeatureCache(directory,checkpoint)
        before=time.perf_counter(); cached=cache.get(sample,model); feature_cold=time.perf_counter()-before
        before=time.perf_counter(); hit=cache.get(sample,model); feature_hit=time.perf_counter()-before
        x,*_=load_and_preprocess(sample,'cpu')
        with torch.no_grad():
            direct=model.encode(x)
            a,_=model.decode(direct); b,_=model.decode(cached)
        maximum=float((direct-cached).abs().max())
        assert torch.equal(cached,hit) and maximum<=1e-6
        assert float((a-b).abs().max())<=1e-6
        assert not model.encoder.training
        model.train(); model.encoder.eval(); model.severity_head.eval()
        optimizer=torch.optim.Adam(model.seg_decoder.parameters(),lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        logits,_=model.decode(cached)
        # Deliberately artificial software fixture, never a label for evaluating quality.
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits))
        loss.backward()
        assert all(p.grad is None for p in model.encoder.parameters())
        assert any(p.grad is not None for p in model.seg_decoder.parameters())
        optimizer.step()
        feature_report=cache.report()
        feature_report.update(cold_seconds=feature_cold,hit_seconds=feature_hit,
                              cached_uncached_max_abs_difference=maximum,
                              disposable_decoder_step_loss=float(loss.detach()),
                              decoder_step_labels='artificial all-zero software fixture; no scientific evaluation')
    del model,optimizer,cached,hit,direct,a,b,logits,loss
    gc.collect()
    runtime.clear_model_cache()
    consistency=consistency_check(sample,checkpoint,enabled=True)
    assert runtime.cache_info()['forward_passes']==2
    benchmark={}
    for enabled in (False,True):
        times=[]
        for _ in range(args.benchmark_runs):
            before=time.perf_counter(); consistency_check(sample,checkpoint,enabled=enabled)
            times.append(time.perf_counter()-before)
        if times:
            benchmark['flip_enabled' if enabled else 'default_off']=dict(
                runs=len(times),median_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)))
    report=dict(checkpoint_sha256=runtime.sha256(checkpoint),input_sha256=runtime.sha256(sample),
                device='cpu',threads=2,python=platform.python_version(),torch=torch.__version__,
                platform=platform.platform(),features=feature_report,consistency=consistency,
                benchmark=benchmark,review_passes=runtime.cache_info(),
                total_wall_seconds=time.perf_counter()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                quality_claim=False,workload='single included sample, no temporal observations or research training')
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()
