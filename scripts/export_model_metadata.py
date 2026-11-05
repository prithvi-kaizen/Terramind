"""Export checkpoint-specific architecture counts without retraining."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from methanepulse.runtime import _get_model,sha256,ARCHITECTURE_VERSION
from methanepulse.preprocessing import PREPROCESSING_VERSION


def main():
    root=Path(__file__).resolve().parents[1]
    checkpoint=root/'terramind_methane_best.pt'
    model=_get_model(checkpoint,torch.device('cpu'))
    notebook=root/'notebooks/TerraMind.ipynb'
    source='\n'.join(''.join(c['source']) for c in json.loads(notebook.read_text())['cells'])
    for required in ['batch_size=2','lr=1e-3',"patience=3, factor=0.5"]:
        if required not in source:raise ValueError('Historical configuration changed; inspect notebook before exporting metadata')
    data={'schema_version':1,'checkpoint_sha256':sha256(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
        'architecture_version':ARCHITECTURE_VERSION,'preprocessing_version':PREPROCESSING_VERSION,
        'encoder':{'parameters':sum(p.numel() for p in model.encoder.parameters()),'frozen':all(not p.requires_grad for p in model.encoder.parameters())},
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'decoder':{'transposed_convolutions':sum(isinstance(m,torch.nn.ConvTranspose2d) for m in model.seg_decoder),
            'output_convolutions':sum(isinstance(m,torch.nn.Conv2d) for m in model.seg_decoder)},
        'severity':{'pooling':'mean over patch embeddings','hidden_dimensions':[model.severity_head[0].in_features,model.severity_head[0].out_features,model.severity_head[-1].out_features]},
        'historical_notebook':{'path':'notebooks/TerraMind.ipynb','sha256':sha256(notebook),'optimizer':'AdamW','learning_rate':.001,
            'weight_decay':.0001,'scheduler_patience':3,'joint_epochs':20,'batch_size':2,'segmentation_weight':.7,'severity_weight':.3},
        'checkpoint_training_run':'unknown'}
    (root/'docs/model_configuration.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))

if __name__=='__main__':main()
