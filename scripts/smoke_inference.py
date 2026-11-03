"""Real offline CPU adapter/cache regression check (includes cloud entrypoint)."""
import os
os.environ['METHANEPULSE_DEVICE']='cpu'
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import rasterio
from streamlit.testing.v1 import AppTest
from methanepulse.runtime import run_inference, predict_tile, clear_model_cache, cache_info
from methanepulse.validation import check_reference_mask
import endpoint


def main():
    root=Path(__file__).resolve().parents[1]
    clear_model_cache()
    sample=next((root/'sample_input').glob('*.tiff'))
    checkpoint=root/'terramind_methane_best.pt'
    with tempfile.TemporaryDirectory() as directory:
        packet=run_inference(sample,checkpoint,directory,device='cpu')
        expected=next((root/'results').glob('*_mask.tiff'))
        extra_passes=0
        with rasterio.open(packet['artifacts']['mask_path']) as actual,rasterio.open(expected) as reference:
            actual_mask,reference_mask=actual.read(1),reference.read(1)
            assert actual.crs==reference.crs and actual.transform==reference.transform
        probability=None
        if not np.array_equal(actual_mask,reference_mask):
            probability=predict_tile(sample,checkpoint,.5,'cpu').probability
            extra_passes=1
        diagnostic=check_reference_mask(actual_mask,reference_mask,probability)
        print('REFERENCE_DIAGNOSTIC',diagnostic,flush=True)
        assert packet['schema_version']=='1.1'
        assert int(reference_mask.sum())==1244
        assert packet['detection']['plume_pixel_count']==int(actual_mask.sum())
        previous=endpoint.RESULTS
        try:
            endpoint.RESULTS=Path(directory)/'api'
            with sample.open('rb') as stream:
                response=endpoint.app.test_client().post('/predict',data={'file':(stream,'sample.tiff'),'threshold':'0.5'})
            assert response.status_code==200,response.json
            assert response.json['detection']==packet['detection']
        finally:
            endpoint.RESULTS=previous
        app=AppTest.from_file(str(root/'deployment/streamlit_app.py')).run(timeout=120)
        assert not app.exception,list(app.exception)
        app.button[0].click().run(timeout=120)
        assert not app.exception,list(app.exception)
        assert app.session_state['packet']['detection']==packet['detection']
        assert cache_info()['loads']==1,cache_info()
        assert cache_info()['forward_passes']==3+extra_passes,cache_info()
        print(f'PASS: bounded sample mask regression/CRS/transform, exact API/cloud adapter equivalence, 1 load / {3+extra_passes} passes')

if __name__=='__main__':
    main()
