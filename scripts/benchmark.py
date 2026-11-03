"""Fresh-process CPU benchmark, including input loading and artifact writing."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--checkpoint', default='terramind_methane_best.pt')
    parser.add_argument('--warm-requests', type=int, default=30)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--started', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warm_requests < 1 or args.threads < 1:
        parser.error('Request and thread counts must be positive')
    if not args.worker:
        env = {**os.environ, 'OMP_NUM_THREADS': str(args.threads), 'MKL_NUM_THREADS': str(args.threads), 'HF_HUB_OFFLINE': '1'}
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker', '--started', str(time.perf_counter())], env=env, check=True)
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from methanepulse.runtime import run_inference, cache_info, sha256
    torch.set_num_threads(args.threads)
    durations = []
    with tempfile.TemporaryDirectory(prefix='methanepulse-benchmark-') as directory:
        for index in range(args.warm_requests + 1):
            start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                packet = run_inference(args.input, args.checkpoint, directory, device='cpu')
            end = time.perf_counter()
            if index == 0:
                cold = end - args.started
                first_request = end - start
            else:
                durations.append(end - start)
        cache = cache_info()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {'hardware': {'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count()},
        'python': platform.python_version(), 'device': 'cpu', 'threads': args.threads,
        'workload': 'single supplied tile repeated; not a representative multi-scene benchmark',
        'tile_sha256': sha256(args.input), 'checkpoint_sha256': sha256(args.checkpoint),
        'checkpoint_bytes': Path(args.checkpoint).stat().st_size,
        'cold_process_to_first_output_seconds': cold, 'first_request_seconds': first_request,
        'warm_requests': args.warm_requests, 'warm_seconds': durations,
        'warm_median_seconds': float(np.median(durations)), 'warm_p95_seconds': float(np.percentile(durations,95)),
        'peak_process_rss_bytes': int(rss if sys.platform == 'darwin' else rss*1024),
        'memory_method': 'process lifetime high-water RSS; includes import and cold load, not per-request memory',
        'failed_requests': 0, 'cache': cache, 'plume_pixels': packet['detection']['plume_pixel_count']}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('cold_process_to_first_output_seconds','warm_median_seconds','warm_p95_seconds','peak_process_rss_bytes','cache')}))

if __name__ == '__main__':
    main()
