from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin
import torch

from infer import load_and_preprocess, plume_centroids, run_inference

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = next((ROOT / "sample_input").glob("*.tiff"))


def test_sample_preprocessing_matches_model_contract():
    x, crs, bounds, transform, profile = load_and_preprocess(SAMPLE, torch.device("cpu"))
    assert x.shape == (1, 2, 512, 512)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert 0 <= x.min() <= x.max() <= 1
    assert crs == "EPSG:32614"


def write_raster(path, bands, width=512, height=512, crs="EPSG:32614", value=0.0):
    with rasterio.open(path, "w", driver="GTiff", count=bands, width=width,
                       height=height, dtype="float32", crs=crs,
                       transform=from_origin(739500, 3350240, 20, 20)) as dst:
        dst.write(np.full((bands, height, width), value, dtype=np.float32))


@pytest.mark.parametrize("bands,width,height", [(2, 512, 512), (13, 256, 512)])
def test_rejects_unsupported_tile_shape(tmp_path, bands, width, height):
    tile = tmp_path / "invalid.tiff"
    write_raster(tile, bands, width, height)
    with pytest.raises(ValueError, match="13-band, 512x512"):
        load_and_preprocess(tile, torch.device("cpu"))


def test_rejects_missing_crs(tmp_path):
    tile = tmp_path / "no-crs.tiff"
    write_raster(tile, 13, crs=None)
    with pytest.raises(ValueError, match="coordinate reference"):
        load_and_preprocess(tile, torch.device("cpu"))


def test_rejects_nonfinite_pixels(tmp_path):
    tile = tmp_path / "nan.tiff"
    write_raster(tile, 13, value=float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        load_and_preprocess(tile, torch.device("cpu"))


@pytest.mark.parametrize("threshold", [0, 1, -0.5, 1.5, float("nan"), float("inf")])
def test_threshold_is_validated_before_model_loading(tmp_path, threshold):
    with pytest.raises(ValueError, match="threshold"):
        run_inference(SAMPLE, tmp_path / "absent.pt", tmp_path, threshold)


def test_centroid_uses_pixel_centres_and_rotation():
    mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    transform = Affine(0.01, 0.002, 10, 0.003, -0.01, 20)
    projected, wgs84 = plume_centroids(mask, transform, "EPSG:4326")
    assert projected == pytest.approx([10.016, 19.9995])
    assert wgs84 == projected


def test_utm_centroid_is_converted_to_longitude_latitude():
    mask = np.ones((1, 1), dtype=np.uint8)
    projected, wgs84 = plume_centroids(mask, from_origin(500000, 1000000, 20, 20), "EPSG:32614")
    assert projected == [500010.0, 999990.0]
    assert wgs84[0] == pytest.approx(-99.0, abs=0.001)
    assert 8 < wgs84[1] < 10


def test_empty_mask_has_no_centroid():
    assert plume_centroids(np.zeros((2, 2), dtype=np.uint8), Affine.identity(), "EPSG:4326") == (None, None)
