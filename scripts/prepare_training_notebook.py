"""Create an unexecuted corrected working copy without editing research artifacts."""
import argparse
import ast
import json
from pathlib import Path
import re


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError('Use a new output path; historical notebooks must remain unchanged')
    notebook = json.loads(source.read_text())
    scheduler_count = frozen_count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        text = ''.join(cell['source'])
        text = re.sub(r'(?m)^(\s*)model\.train\(\)\s*$', r'\1model.train()\n\1model.encoder.eval()  # Frozen features remain deterministic', text)
        frozen_count += text.count('model.encoder.eval()  # Frozen features remain deterministic')
        if 'optimizer = torch.optim.AdamW(' in text:
            text, removed = re.subn(r"scheduler = ReduceLROnPlateau\(optimizer, mode='max',\s*patience=5,\s*# was 3\s*factor=0\.5\)\n",'',text)
            if removed != 1:
                raise ValueError('Unexpected historical scheduler layout; inspect before preparing')
            scheduler_count += removed
            text = re.sub(r'(?m)^lr\s*= 5e-4[^\n]*\n','',text)
            text = re.sub(r'(?m)^patience\s*= 5[^\n]*\n','',text)
            text = text.replace('lr=1e-3,           # was 5e-4','lr=1e-3,           # Explicit effective learning rate')
        ast.parse(text.split("\n",1)[1] if text.startswith("%%writefile") else text)
        cell.update(source=text.splitlines(keepends=True),outputs=[],execution_count=None)
    if scheduler_count != 1 or frozen_count != 2:
        raise ValueError('Expected one joint optimizer cell and two training loops')
    notebook['metadata']['phase_1_preparation'] = {'executed':False,'optimizer_lr':0.001,'scheduler_patience':3,'encoder_training':False}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(notebook,indent=1)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='notebooks/TerraMind.ipynb')
    parser.add_argument('--output',default='outputs/notebooks/TerraMind-working.ipynb')
    args=parser.parse_args()
    prepare(args.source,args.output)

if __name__=='__main__':
    main()
