import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from scripts import evaluate_public_data as public


def tiff(data):
    with MemoryFile() as file:
        with file.open(driver='GTiff',height=data.shape[1],width=data.shape[2],count=data.shape[0],dtype=str(data.dtype),crs='EPSG:32614',transform=rasterio.Affine(10,0,500000,0,-10,3000000)) as out:
            out.write(data)
        return file.read()


def test_summary_never_calls_all_failures_perfect():
    assert public.summary([{'error':'missing'}])['pixels'] is None
    rows=[dict(tp=2,fp=1,fn=1,iou=.5,label_positive=True,detected=True),dict(tp=0,fp=3,fn=0,iou=0,label_positive=False,detected=True)]
    result=public.summary(rows)
    assert result['pixels']['iou']==2/7
    assert result['scene_precision']==.5
    assert result['false_positive_scene_rate']==1
    assert public.summary(rows[:1])['scene_precision'] is None


def test_native_grid_adapter_is_explicit_and_preserves_footprint(monkeypatch):
    data=np.arange(13*200*200,dtype='float32').reshape(13,200,200)+1
    def predict(name,*args):
        with rasterio.open(name) as adapted:
            assert adapted.shape==(512,512)
            assert adapted.count==13
            assert adapted.descriptions[11:] == ('B11','B12')
            assert adapted.transform.a==10*200/512
        probability=np.full((512,512),.25,dtype='float32');probability[:256]=.75
        return SimpleNamespace(probability=probability)
    monkeypatch.setattr(public,'predict_tile',predict)
    mask,crs,transform=public.resize_prediction(tiff(data),'unused')
    assert mask.shape==(200,200) and mask[:100].all() and not mask[100:].any()
    assert transform.a==10 and crs.to_epsg()==32614


def test_synthetic_archive_matching_and_complete_report(tmp_path,monkeypatch):
    tile=tiff(np.ones((13,2,2),dtype='float32'))
    label=tiff(np.array([[[1,0],[0,0]]],dtype='uint8'))
    archive=tmp_path/'val.tar'
    with tarfile.open(archive,'w') as tar:
        for name,payload in [('0___2_S2A_DATE_R1000_nonorm_bands20m.tiff',tile),('label/0___2_S2A_DATE_R1000_bin_label.tiff',label)]:
            entry=tarfile.TarInfo(name);entry.size=len(payload);tar.addfile(entry,io.BytesIO(payload))
    monkeypatch.setitem(public.ARCHIVES,'val',(hashlib.md5(archive.read_bytes()).hexdigest(),0))
    monkeypatch.setattr(public,'predict_tile',lambda *args:SimpleNamespace(mask=np.array([[1,0],[0,0]],dtype='uint8')))
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'test')
    public.synthetic(argparse.Namespace(output=tmp_path/'reports',data=tmp_path,checkpoint=checkpoint,splits='val',remove_archives=False))
    report=json.loads((tmp_path/'reports/geoch4-val.json').read_text())
    assert report['summary']['pixels']['iou']==1
    assert report['summary']['evaluated']==1 and report['summary']['failed']==0
    assert report['tiles'][0]['location']=='2'


def test_taco_join_uses_source_archive_and_rejects_duplicates():
    rows=[{'id':kind,'internal:parent_id':0,'internal:source_file':country}
          for country in ['Algeria','Argentina'] for kind in ['target','bg0','plume']]
    selected=public.select_catalog_leaves(rows,0,'Argentina')
    assert len(selected)==3 and all(r['internal:source_file']=='Argentina' for r in selected)
    with pytest.raises(ValueError,match='exactly one'):
        public.select_catalog_leaves(rows+[rows[-1]],0,'Argentina')
