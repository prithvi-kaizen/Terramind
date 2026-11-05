import numpy as np
import pytest
from pyproj import CRS
from rasterio.transform import Affine, from_origin
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from methanepulse.candidates import CandidateConfig,extract_candidates

TRANSFORM=from_origin(500000,3000000,20,20)


def extract(prob,**kwargs):
    return extract_candidates(np.asarray(prob,dtype='float32'),TRANSFORM,'EPSG:32614',**kwargs)


def test_empty_collection_and_four_connected_components():
    empty=extract(np.zeros((5,5)))
    assert empty['candidates']==[] and empty['geojson']['features']==[]
    prob=np.zeros((5,5));prob[1,1]=prob[2,2]=.9
    result=extract(prob,config=CandidateConfig(min_area_m2=0))
    assert len(result['candidates'])==2
    assert [c['pixel_count'] for c in result['candidates']]==[1,1]
    assert all(c['evidence_state']=='candidate' for c in result['candidates'])
    assert np.array_equal(result['filtered_mask'],prob>.5)


@pytest.mark.parametrize('crs,transform',[
    ('EPSG:32614',TRANSFORM),
    ('EPSG:4326',Affine(.001,.0002,-99,.0001,-.001,30)),
])
def test_polygon_holes_and_roundtrip_alignment(crs,transform):
    prob=np.zeros((5,5),dtype='float32');prob[1:4,1:4]=.9;prob[2,2]=0
    result=extract_candidates(prob,transform,crs)
    candidate=result['candidates'][0]
    assert candidate['pixel_count']==8 and candidate['bbox_pixels']==[1,1,4,4]
    assert len(candidate['geometry_wgs84']['coordinates'])==2
    native=transform_geom('EPSG:4326',crs,candidate['geometry_wgs84'])
    reconstructed=rasterize([(native,1)],out_shape=prob.shape,transform=transform)
    assert np.array_equal(reconstructed,prob>.5)
    assert candidate['area_m2']>0
    x,y=transform*(2.5,2.5)
    assert candidate['centroid_source']==pytest.approx([x,y])


def test_geodesic_area_respects_projected_feet():
    # Texas South Central has US survey-foot coordinates, not metres.
    transform=from_origin(2000000,10000000,20,20)
    result=extract_candidates(np.array([[.9]]),transform,'EPSG:2277')
    area=result['candidates'][0]['area_m2']
    factor=CRS.from_epsg(2277).axis_info[0].unit_conversion_factor
    assert area==pytest.approx(400*factor**2,rel=.02)
    assert area<40  # treating map units as metres would incorrectly yield 400


def test_geographic_area_and_rotated_affine():
    transform=Affine(.01,.002,10,.003,-.01,20)
    result=extract_candidates(np.array([[.9]]),transform,'EPSG:4326')
    c=result['candidates'][0]
    assert c['centroid_source']==pytest.approx([10.006,19.9965])
    assert 1e6<c['area_m2']<1.5e6
    assert c['bbox_source']==pytest.approx([10,19.99,10.012,20.003])


def test_flags_are_informational_until_filter_explicitly_enabled():
    prob=np.zeros((5,5));prob[0,0]=.51
    raw=extract(prob)
    c=raw['candidates'][0]
    assert set(c['quality_flags'])=={'edge_contact','small_area','threshold_sensitive'}
    assert c['evidence_state']=='needs_review'
    assert raw['filtered_mask'].sum()==1
    selected=extract(prob,config=CandidateConfig(filter_flags=('small_area',)))
    assert selected['filtered_mask'].sum()==0
    assert selected['labels'].sum()==1
    assert selected['candidates'][0]['removal_reasons']==['small_area']
    assert prob[0,0]==.51


def test_nodata_exclusion_is_separate_from_heuristic_filters():
    prob=np.zeros((5,5));prob[2,2:4]=.9
    valid=np.ones((5,5),dtype=bool);valid[2,2]=False
    result=extract(prob,valid_mask=valid)
    assert result['labels'][2,2]>0 and result['filtered_mask'][2,2]==0
    assert result['filtered_mask'].sum()==1
    assert result['validity']['raw_foreground_on_invalid_pixels']==1
    c=result['candidates'][0]
    assert c['invalid_pixel_count']==1 and 'invalid_pixel_overlap' in c['quality_flags']
    assert not c['removed_by_experimental_filter']
    assert 0<=c['threshold_sensitivity']['fraction']<=1


def test_threshold_sensitivity_reuses_neighbouring_probabilities():
    prob=np.zeros((5,5));prob[2,2]=.51
    prob[1,2]=prob[3,2]=prob[2,1]=prob[2,3]=.49
    result=extract(prob)
    sensitivity=result['candidates'][0]['threshold_sensitivity']
    assert sensitivity['local_lower_pixel_count']==5
    assert sensitivity['upper_surviving_pixel_count']==0
    assert sensitivity['fraction']==1


def test_config_hash_and_fragmentation_budget():
    a=CandidateConfig(filter_flags=('small_area','edge_contact'))
    b=CandidateConfig(filter_flags=('edge_contact','small_area'))
    assert a.manifest(.5)==b.manifest(.5)
    assert a.manifest(.5)['config_sha256']!=a.manifest(.6)['config_sha256']
    with pytest.raises(ValueError,match='budget'):
        extract(np.eye(5)*.9,config=CandidateConfig(max_candidates=2))


@pytest.mark.parametrize('kwargs',[{'min_area_m2':float('nan')},{'threshold_delta':0},{'filter_flags':('unknown',)},{'max_candidates':False}])
def test_invalid_config_fails(kwargs):
    with pytest.raises(ValueError):CandidateConfig(**kwargs)


def test_numerically_equivalent_settings_have_canonical_hash():
    assert CandidateConfig(min_area_m2=0).manifest(.5)==CandidateConfig(min_area_m2=0.0).manifest(.5)
