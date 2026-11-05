"""Fresh-process comparison against an explicit, immutable Phase 1 Git revision."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref',required=True)
    parser.add_argument('--warm-requests',type=int,default=30)
    parser.add_argument('--output',default='docs/phase-2-benchmark.json')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    baseline_ref=subprocess.check_output(['git','rev-parse',args.baseline_ref],cwd=root,text=True).strip()
    archive=subprocess.check_output(['git','archive',baseline_ref,'methanepulse','scripts/benchmark.py'],cwd=root)
    sample=next((root/'sample_input').glob('*.tiff'))
    with tempfile.TemporaryDirectory(prefix='terramind-phase2-benchmark-') as directory:
        directory=Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(directory,filter='data')
        reports={}
        for name,source in [('phase1',directory),('phase2',root)]:
            output=directory/(name+'.json')
            subprocess.run([sys.executable,str(source/'scripts/benchmark.py'),
                '--input',str(sample),'--checkpoint',str(root/'terramind_methane_best.pt'),
                '--threads','2','--warm-requests',str(args.warm_requests),'--output',str(output)],cwd=source,check=True)
            reports[name]=json.loads(output.read_text())
        ratio=reports['phase2']['warm_p95_seconds']/reports['phase1']['warm_p95_seconds']
        assert reports['phase1']['tile_sha256']==reports['phase2']['tile_sha256']
        assert reports['phase1']['checkpoint_sha256']==reports['phase2']['checkpoint_sha256']
        result={'baseline_ref':baseline_ref,'method':'Two sequential fresh processes on the same machine; Phase 1 then Phase 2; identical tile/checkpoint, 2 threads, 30 warm requests by default. Single-sample workload; host background activity and OS cache can affect timings.',
                'target_max_p95_ratio':1.2,'p95_ratio':ratio,'target_met':ratio<=1.2,**reports}
        path=root/args.output;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'p95_ratio':ratio,'target_met':ratio<=1.2}))

if __name__=='__main__':main()
