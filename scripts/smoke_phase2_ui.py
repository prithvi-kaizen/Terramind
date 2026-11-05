"""Actual CPU UI workflow plus explicitly synthetic empty/error state fixtures."""
import os
os.environ['METHANEPULSE_DEVICE']='cpu'
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from methanepulse.runtime import clear_model_cache,cache_info


def main():
    root=Path(__file__).resolve().parents[1]
    clear_model_cache()
    app=AppTest.from_file(str(root/'deployment/streamlit_app.py')).run(timeout=120)
    assert not app.exception,list(app.exception)
    app.button[0].click().run(timeout=120)
    assert not app.exception,list(app.exception)
    packet=app.session_state['packet']
    assert packet['schema_version']=='2.0' and packet['candidate_count']>0
    assert not packet['filtered']['enabled']
    assert len(app.get('download_button'))==5
    passes=cache_info()['forward_passes']
    app.selectbox[0].set_value(packet['candidates'][-1]['id']).run(timeout=120)
    assert not app.exception,list(app.exception)
    assert cache_info()['forward_passes']==passes
    app.checkbox[0].check().run(timeout=120)
    app.number_input[0].set_value(1e12).run(timeout=120)
    assert cache_info()['forward_passes']==passes
    app.button[0].click().run(timeout=120)
    assert not app.exception,list(app.exception)
    filtered=app.session_state['packet']
    assert filtered['filtered']['enabled'] and filtered['filtered']['plume_pixel_count']==0
    assert filtered['detection']['plume_pixel_count']==packet['detection']['plume_pixel_count']
    assert cache_info()['forward_passes']==passes+1
    assert cache_info()['loads']==1
    app.radio[1].set_value('Experimental filtered mask').run(timeout=120)
    assert not app.exception,list(app.exception)
    assert cache_info()['forward_passes']==passes+1
    from methanepulse import runtime
    from methanepulse.preprocessing import load_and_preprocess
    from methanepulse.artifacts import write_artifacts
    with tempfile.TemporaryDirectory(dir=root/'outputs'/'predictions',prefix='ui-fixture-') as directory:
        def empty_fixture(**kwargs):
            _,crs,bounds,transform,profile=load_and_preprocess(kwargs['tiff_path'],torch.device('cpu'))
            return write_artifacts(kwargs['tiff_path'],directory,kwargs['confidence_threshold'],
                np.zeros((512,512),dtype='float32'),np.zeros((512,512),dtype='uint8'),0,0.,
                crs,bounds,transform,profile,runtime._cached_model,
                identity={**packet['model'],'test_fixture':True},candidate_config=kwargs['candidate_config'])
        with patch('methanepulse.runtime.run_inference',side_effect=empty_fixture):
            app.button[0].click().run(timeout=120)
            assert not app.exception,list(app.exception)
            assert app.session_state['packet']['candidate_count']==0
            assert any('No connected candidates' in item.value for item in app.info)
            assert len(app.get('download_button'))==5
    with patch('methanepulse.runtime.run_inference',side_effect=ValueError('test-only invalid input fixture')):
        app.button[0].click().run(timeout=120)
        assert not app.exception,list(app.exception)
        assert any('test-only invalid input' in error.value for error in app.error)
    print('PASS: real CPU candidate inspection, 5 downloads, explicit filtering, raw preservation, no extra inspection forward pass, and isolated empty/error fixtures')

if __name__=='__main__':main()
