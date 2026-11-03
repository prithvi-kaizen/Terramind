import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from methanepulse.evaluation import counts,scores,load_manifest,read_target
from scripts import evaluate as evaluation_cli
from scripts.cleanup_outputs import expired_outputs


def write_pair(root,name,positive=False):
    tile=root/(name+'.tiff'); mask=root/(name+'-mask.tiff')
    profile=dict(driver='GTiff',height=2,width=2,count=1,dtype='uint8',crs='EPSG:32614',transform=from_origin(1,3,1,1))
    with rasterio.open(tile,'w',**profile) as dst:
        dst.write(np.full((1,2,2),sum(name.encode())%255,dtype='uint8'))
    with rasterio.open(mask,'w',**profile) as dst:
        dst.write(np.full((1,2,2),int(positive),dtype='uint8'))
    return dict(id=name,tile=tile.name,mask=mask.name,split='validation',location=name,acquisition=name,label_source='test fixture',verified_negative=not positive)


def test_hand_calculated_metrics():
    result=counts([1,1,0,0],[1,0,1,0])
    assert result==dict(tp=1,fp=1,fn=1)
    assert scores(result)==dict(iou=1/3,f1=.5)
    assert scores(dict(tp=0,fp=0,fn=0))==dict(iou=1.,f1=1.)


@pytest.mark.parametrize('leak',['location','acquisition','plume_id','tile'])
def test_manifest_leakage(tmp_path,leak):
    first=write_pair(tmp_path,'a'); second=write_pair(tmp_path,'bb')
    second['split']='test'
    if leak=='plume_id':
        first[leak]='same-plume'
    second[leak]=first[leak]
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(schema_version=1,tiles=[first,second])))
    with pytest.raises(ValueError,match='leakage|Duplicate'):
        load_manifest(path)


def test_evaluator_excludes_unverified_negatives(tmp_path,monkeypatch):
    positive=write_pair(tmp_path,'positive',True)
    negative=write_pair(tmp_path,'negative')
    unknown=write_pair(tmp_path,'unverified')
    unknown['verified_negative']=False
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(schema_version=1,tiles=[positive,negative,unknown])))
    checkpoint=tmp_path/'weights.pt'; checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(evaluation_cli,'predict_tile',lambda *a:SimpleNamespace(mask=np.ones((2,2),dtype='uint8')))
    report=evaluation_cli.evaluate(path,checkpoint)
    assert report['pixels']['tp']==4 and report['pixels']['fp']==8
    assert report['scenes']['precision']==.5
    assert report['scenes']['recall']==1
    assert report['scenes']['false_positive_scene_rate']==1
    assert report['scenes']['unverified_negative_count']==1


def test_label_alignment_and_binary_validation(tmp_path):
    row=write_pair(tmp_path,'sample')
    with rasterio.open(tmp_path/row['mask'],'r+') as label:
        label.write(np.full((1,2,2),2,dtype='uint8'))
    with pytest.raises(ValueError,match='binary'):
        read_target(tmp_path/row['mask'],tmp_path/row['tile'])


def test_retention_preserves_incomplete_and_other_files(tmp_path):
    complete=tmp_path/('a'*32); complete.mkdir(); (complete/'.completed').touch()
    incomplete=tmp_path/('b'*32); incomplete.mkdir()
    foreign=tmp_path/'foreign'; foreign.mkdir(); (foreign/'.completed').touch()
    assert expired_outputs(tmp_path,age_hours=0,keep=0)==[complete]
    assert expired_outputs(tmp_path,age_hours=0,keep=1)==[]


def test_prepared_training_notebook_preserves_original(tmp_path):
    from scripts.prepare_training_notebook import prepare
    from methanepulse.runtime import sha256
    source=Path(__file__).resolve().parents[1]/'notebooks'/'TerraMind.ipynb'
    before=sha256(source)
    output=tmp_path/'working.ipynb'
    prepare(source,output)
    prepared=json.loads(output.read_text())
    assert sha256(source)==before
    code='\n'.join(''.join(c['source']) for c in prepared['cells'] if c['cell_type']=='code')
    assert code.count("scheduler = ReduceLROnPlateau(")==1
    assert code.count('model.encoder.eval()  # Frozen features remain deterministic')==2
    assert 'lr         = 5e-4' not in code
    assert all(c['execution_count'] is None and not c['outputs'] for c in prepared['cells'] if c['cell_type']=='code')
