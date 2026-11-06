import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch

from methanepulse import reliability, runtime
from methanepulse.candidates import CandidateConfig
from methanepulse.features import FeatureCache
from methanepulse.model import TerraMindMethanePulse
from methanepulse.training import promotion_gate, reviewed_manifest, train

SAMPLE=next((Path(__file__).resolve().parents[1]/'sample_input').glob('*.tiff'))


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(1.))
    def forward(self,x):
        return [torch.ones(x.shape[0],196,384)*self.weight]


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=Encoder()
        self.seg_decoder=torch.nn.Conv2d(1,1,1)
        self.severity_head=torch.nn.Linear(384,3)
    def encode(self,x):
        self.encoder.eval()
        with torch.no_grad(): return self.encoder(x)[-1]
    def decode(self,features):
        x=features.mean((1,2)).reshape(-1,1,1,1).expand(-1,1,512,512)
        return self.seg_decoder(x), self.severity_head(features.mean(1))
    def forward(self,x): return self.decode(self.encode(x))


def test_encode_keeps_frozen_encoder_eval():
    model=TerraMindMethanePulse(Encoder())
    model.train()
    with torch.no_grad(): feat=model.encode(torch.zeros(1,2,512,512))
    assert not model.encoder.training
    assert not feat.requires_grad


def test_feature_cache_equivalence_augmentation_and_corruption(tmp_path):
    checkpoint=tmp_path/'checkpoint'; checkpoint.write_bytes(b'weights')
    model=SmallModel().eval()
    cache=FeatureCache(tmp_path/'features',checkpoint)
    first=cache.get(SAMPLE,model)
    assert torch.equal(first,cache.get(SAMPLE,model))
    assert cache.encoder_passes==1 and cache.hits==1
    cache.get(SAMPLE,model,'flip-x')
    assert cache.encoder_passes==2
    with pytest.raises(ValueError,match='augmentation'): cache.get(SAMPLE,model,'rotate')
    checkpoint.write_bytes(b'new')
    with pytest.raises(ValueError,match='changed'): cache.get(SAMPLE,model)


def test_feature_cache_rejects_invalid_tensor(tmp_path):
    checkpoint=tmp_path/'checkpoint'; checkpoint.write_bytes(b'weights')
    cache=FeatureCache(tmp_path/'cache',checkpoint)
    cache.get(SAMPLE,SmallModel())
    path=next((tmp_path/'cache').glob('*.npz'))
    with np.load(path) as saved: identity=saved['identity'].copy()
    np.savez(path,identity=identity,features=np.zeros((1,2)))
    with pytest.raises(ValueError,match='Invalid cached'): cache.get(SAMPLE,SmallModel())


@pytest.mark.parametrize('n',[0,1,9,10,19,20,101])
def test_routing_budget(n):
    result=reliability.route_consistency([True]*n,'automatic')
    assert result['extra_passes']==n//10
    assert result['maximum_total_passes']<=1.1*n
    assert reliability.route_consistency([True]*n)['extra_passes']==0


def test_consistency_optional_and_inverted(monkeypatch,tmp_path):
    runtime.clear_model_cache()
    checkpoint=tmp_path/'checkpoint'; checkpoint.write_bytes(b'weights')
    class Spatial(torch.nn.Module):
        def forward(self,x): return x[:,:1]*4-2, torch.zeros(1,3)
    monkeypatch.setattr(runtime,'_build_model',lambda *args:Spatial())
    result=reliability.consistency_check(SAMPLE,checkpoint)
    assert result['extra_passes']==0 and runtime.cache_info()['forward_passes']==1
    result=reliability.consistency_check(SAMPLE,checkpoint,enabled=True)
    assert result['signal']['mask_iou']==1
    assert result['signal']['mean_absolute_probability_difference']<1e-7
    assert runtime.cache_info()['forward_passes']==3
    runtime.clear_model_cache()


def test_prediction_cache_invalidates_settings(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoint'; checkpoint.write_bytes(b'weights')
    calls=[]
    monkeypatch.setattr(runtime,'predict_tile',lambda *args: calls.append(1) or SimpleNamespace(probability=np.ones((512,512),dtype=np.float32)))
    cache=reliability.PredictionCache(tmp_path/'cache')
    for _ in range(2): cache.get(SAMPLE,checkpoint,.5,CandidateConfig())
    assert len(calls)==1 and cache.hits==1
    cache.get(SAMPLE,checkpoint,.6,CandidateConfig())
    cache.get(SAMPLE,checkpoint,.6,CandidateConfig(min_area_m2=5))
    checkpoint.write_bytes(b'replaced')
    cache.get(SAMPLE,checkpoint,.6,CandidateConfig(min_area_m2=5))
    assert len(calls)==4


def test_spatial_matching_one_to_one():
    def c(identifier,lon): return dict(id=identifier,centroid_wgs84=[lon,30.])
    matches=reliability.match_candidates([c('a',0),c('b',.0001)],[c('c',0)],100)
    assert len(matches)==1 and matches[0]['displacement_m']==0
    assert reliability.match_candidates([c('a',0)],[c('b',1)],100)==[]
    with pytest.raises(ValueError): reliability.match_candidates([],[],float('nan'))


def temporal_manifest(tmp_path,n=2):
    with rasterio.open(SAMPLE) as src: pixels=src.read(); profile=src.profile.copy()
    tiles=[]
    for i in range(n):
        tile=tmp_path/f'date{i}.tiff'; values=pixels.copy(); values[0,0,0]+=i
        with rasterio.open(tile,'w',**profile) as dst: dst.write(values)
        tiles.append(tile)
    observations=[dict(date=f'2024-03-{20+i}',tile=str(tiles[i]),registration_residual_pixels=0,
                       registration_evidence='synthetic software fixture only',radiometry=dict(
                           processing_level='L2A',scale=.0001,offset=0,band_order_verified=True,evidence='fixture')) for i in range(n)]
    path=tmp_path/'temporal.json'; path.write_text(json.dumps(dict(location='fixture',observations=observations)))
    return path


def test_temporal_reuses_predictions_and_records_dates(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoint'; checkpoint.write_bytes(b'weights')
    probability=np.zeros((512,512),dtype=np.float32); probability[20:23,20:23]=.9
    monkeypatch.setattr(runtime,'predict_tile',lambda *args:SimpleNamespace(probability=probability))
    path=temporal_manifest(tmp_path)
    result=reliability.temporal_review(path,checkpoint,tmp_path/'cache')
    assert result['date_passes']==2 and result['cache_hits']==0
    assert result['observations'][0]['candidates'][0]['temporal_evidence']['observation_count']==2
    assert reliability.temporal_review(path,checkpoint,tmp_path/'cache')['date_passes']==0


@pytest.mark.parametrize('kind',['missing','too_many','duplicate','radiometry','registration','grid','nodata'])
def test_temporal_rejects_incompatible_dates(tmp_path,monkeypatch,kind):
    path=temporal_manifest(tmp_path,4 if kind=='too_many' else 2)
    data=json.loads(path.read_text())
    if kind=='missing': data['observations']=[]
    if kind=='duplicate': data['observations'][1]['date']=data['observations'][0]['date']
    if kind=='radiometry': data['observations'][1]['radiometry']['scale']=1
    if kind=='registration': data['observations'][1]['registration_residual_pixels']=.5
    if kind in ('grid','nodata'):
        with rasterio.open(SAMPLE) as src: pixels=src.read(); profile=src.profile.copy()
        if kind=='grid': profile['transform']=profile['transform']*rasterio.Affine.translation(1,0)
        else: profile['nodata']=65535; pixels[:,0,0]=65535
        other=tmp_path/'other.tiff'
        with rasterio.open(other,'w',**profile) as dst: dst.write(pixels)
        data['observations'][1]['tile']=str(other)
    path.write_text(json.dumps(data))
    monkeypatch.setattr(runtime,'predict_tile',lambda *args:pytest.fail('Must reject before prediction'))
    with pytest.raises(ValueError): reliability.temporal_review(path,'missing',tmp_path/'cache')


def test_empty_agreement():
    result=reliability.agreement(np.zeros((2,2)),np.zeros((2,2)),.5)
    assert result['both_empty'] and result['mask_iou']==1


def test_gate_rejects_tradeoff_and_missing_evidence():
    baseline=dict(scene_recall=.95,false_positive_scene_rate=.9)
    assert promotion_gate(baseline,dict(scene_recall=.94,false_positive_scene_rate=.8))
    assert not promotion_gate(baseline,dict(scene_recall=.90,false_positive_scene_rate=.8))
    assert not promotion_gate(baseline,dict(scene_recall=.95,false_positive_scene_rate=None))


def training_manifest(tmp_path):
    rows=[]
    with rasterio.open(SAMPLE) as src: pixels=src.read(); profile=src.profile.copy()
    for i,(split,positive) in enumerate([('train',True),('train',False),('validation',True),('validation',False),('test',True)]):
        tile=tmp_path/f'tile{i}.tiff'; data=pixels.copy(); data[0,0,0]+=i
        with rasterio.open(tile,'w',**profile) as dst: dst.write(data)
        mask=tmp_path/f'mask{i}.tiff'; mask_profile={**profile,'count':1,'dtype':'uint8','nodata':None}
        with rasterio.open(mask,'w',**mask_profile) as dst: dst.write(np.full((512,512),positive,dtype=np.uint8),1)
        rows.append(dict(id=str(i),split=split,tile=str(tile),mask=str(mask),location=f'site{i}',acquisition=f'date{i}',
                         label_source='synthetic test fixture',verified_negative=not positive,reviewer='test fixture',
                         review_status='accepted',review_evidence='synthetic, not scientific evidence'))
    path=tmp_path/'manifest.json'; path.write_text(json.dumps(dict(schema_version=1,tiles=rows)))
    return path


def test_bounded_training_three_configs_never_reads_test_labels(tmp_path,monkeypatch):
    path=training_manifest(tmp_path)
    checkpoint=tmp_path/'model.pt'; checkpoint.write_bytes(b'fixture')
    import methanepulse.training as module
    monkeypatch.setattr(module,'_build_model',lambda *args:SmallModel())
    original=module.read_target
    def guarded(mask,tile):
        assert 'mask4' not in str(mask)
        return original(mask,tile)
    monkeypatch.setattr(module,'read_target',guarded)
    report=train(path,checkpoint,tmp_path/'experiment',max_seconds=60)
    assert len(report['runs'])==3 and not report['test_labels_used']
    assert report['resources']['encoder_passes']==4
    assert 'retain_existing' in report['release_decision']
    assert len(list((tmp_path/'experiment').glob('*.pt')))==3
    with pytest.raises(ValueError,match='empty'): train(path,checkpoint,tmp_path/'experiment')


def test_training_rejects_unreviewed_and_leaking_splits(tmp_path):
    path=training_manifest(tmp_path)
    data=json.loads(path.read_text()); data['tiles'][0]['review_status']='pending'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='review'): reviewed_manifest(path)
    data['tiles'][0]['review_status']='accepted'; data['tiles'][2]['location']=data['tiles'][0]['location']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='leakage'): reviewed_manifest(path)


def test_encoder_weight_and_preprocessing_invalidation(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoint'; checkpoint.write_bytes(b'weights')
    model=SmallModel().eval(); cache=FeatureCache(tmp_path/'cache',checkpoint)
    first=cache.get(SAMPLE,model)
    with torch.no_grad(): model.encoder.weight.add_(1)
    second=cache.get(SAMPLE,model)
    assert not torch.equal(first,second) and cache.encoder_passes==2
    import methanepulse.features as module
    monkeypatch.setattr(module,'PREPROCESSING_VERSION','changed')
    cache.get(SAMPLE,model)
    assert cache.encoder_passes==3


def test_batch_consistency_executes_only_budgeted_extra_passes(monkeypatch):
    monkeypatch.setattr(runtime,'sha256',lambda path:'fixture')
    calls=[]
    def check(tile,checkpoint,threshold,enabled):
        calls.append(enabled)
        return dict(base_passes=1,extra_passes=int(enabled),signal=None)
    monkeypatch.setattr(reliability,'consistency_check',check)
    result=reliability.consistency_batch([dict(tile='fixture',flagged=True)]*19,'fixture','automatic')
    assert len(calls)==19 and sum(calls)==1 and result['extra_passes']==1


def test_temporal_single_date_empty_and_duplicate_content(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoint'; checkpoint.write_bytes(b'weights')
    monkeypatch.setattr(runtime,'predict_tile',lambda *args:SimpleNamespace(probability=np.zeros((512,512),dtype=np.float32)))
    path=temporal_manifest(tmp_path,1)
    result=reliability.temporal_review(path,checkpoint,tmp_path/'cache')
    assert result['observations'][0]['candidates']==[] and result['pairs']==[]
    path=temporal_manifest(tmp_path,2); data=json.loads(path.read_text())
    data['observations'][1]['tile']=data['observations'][0]['tile']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Duplicate image'): reliability.temporal_review(path,checkpoint,tmp_path/'cache')


def test_training_preparation_budget_saves_failure_report(tmp_path,monkeypatch):
    import methanepulse.training as module
    path=training_manifest(tmp_path); checkpoint=tmp_path/'checkpoint'; checkpoint.write_bytes(b'weights')
    monkeypatch.setattr(module,'_build_model',lambda *args:SmallModel())
    monkeypatch.setattr(FeatureCache,'get',lambda *args:(_ for _ in ()).throw(TimeoutError('fixture timeout')))
    report=train(path,checkpoint,tmp_path/'budget')
    assert report['status']=='budget_exhausted_during_preparation'
    assert (tmp_path/'budget'/'report.json').exists()
