from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import rasterio
import torch
from methanepulse import runtime
from methanepulse.preprocessing import load_and_preprocess

SAMPLE = next((Path(__file__).resolve().parents[1]/'sample_input').glob('*.tiff'))


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    runtime.clear_model_cache()
    monkeypatch.setenv('METHANEPULSE_DEVICE','cpu')
    checkpoint=tmp_path/'model.pt'
    checkpoint.write_bytes(b'weights-1')
    models=[]
    class Model(torch.nn.Module):
        active=0
        max_active=0
        def forward(self,x):
            Model.active+=1
            Model.max_active=max(Model.max_active,Model.active)
            time.sleep(.02)
            Model.active-=1
            return torch.zeros((1,1,512,512)),torch.zeros((1,3))
    def build(*args):
        model=Model().eval()
        models.append(model)
        return model
    monkeypatch.setattr(runtime,'_build_model',build)
    yield checkpoint,models,Model
    runtime.clear_model_cache()


def test_cache_invalidates_on_contents_and_config(fake_model,tmp_path,monkeypatch):
    checkpoint,models,_=fake_model
    for _ in range(2):
        runtime.predict_tile(SAMPLE,checkpoint)
    assert len(models)==1
    checkpoint.write_bytes(b'weights-2')
    runtime.predict_tile(SAMPLE,checkpoint)
    assert len(models)==2
    monkeypatch.setattr(runtime,'ARCHITECTURE_VERSION','different-architecture')
    runtime.predict_tile(SAMPLE,checkpoint)
    assert len(models)==3
    assert runtime.cache_info()['entries']==1
    assert all(not model.training for model in models)


def test_concurrent_calls_load_once_and_keep_outputs_separate(fake_model,tmp_path):
    checkpoint,models,model_type=fake_model
    def request(_):
        return runtime.run_inference(SAMPLE,checkpoint,tmp_path/'outputs')
    with ThreadPoolExecutor(max_workers=3) as pool:
        packets=list(pool.map(request,range(3)))
    assert len(models)==1
    assert model_type.max_active==1
    assert runtime.cache_info()['forward_passes']==3
    paths=[Path(p['artifacts']['mask_path']) for p in packets]
    assert len(set(paths))==3
    assert all(path.exists() and (path.parent/'.completed').exists() for path in paths)
    assert packets[0]['detection']==packets[1]['detection']


@pytest.mark.parametrize('kind', ['nodata','constant','band-order','affine'])
def test_reject_invalid_metadata_before_loading_model(fake_model,tmp_path,kind):
    checkpoint,models,_=fake_model
    with rasterio.open(SAMPLE) as src:
        data=src.read()
        profile=src.profile.copy()
        descriptions=src.descriptions
    if kind=='nodata':
        profile['nodata']=65535
        data[:,0,0]=65535
    if kind=='constant':
        data[11]=1
    if kind=='affine':
        profile['transform']=rasterio.Affine(1,2,100,2,4,100)
    target=tmp_path/'bad.tiff'
    with rasterio.open(target,'w',**profile) as dst:
        dst.write(data)
        dst.descriptions=tuple(reversed(descriptions)) if kind=='band-order' else descriptions
    with pytest.raises(ValueError):
        runtime.predict_tile(target,checkpoint)
    assert not models
