import io
from pathlib import Path

import pytest

import endpoint
from rasterio.errors import RasterioIOError


@pytest.fixture
def client():
    endpoint.app.config["TESTING"] = True
    return endpoint.app.test_client()


def test_health_does_not_load_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok", "checkpoint_present": True}


def test_missing_upload_returns_json_error(client):
    response = client.post("/predict")
    assert response.status_code == 400
    assert "missing" in response.json["error"]


@pytest.mark.parametrize("threshold", ["bad", "nan", "inf", "0", "1", "-1"])
def test_invalid_threshold_does_not_start_inference(client, monkeypatch, threshold):
    def unexpected(**kwargs):
        pytest.fail("Invalid input must not trigger model inference")
    monkeypatch.setattr(endpoint, "run_inference", unexpected)
    response = client.post("/predict", data={"file": (io.BytesIO(b"tile"), "tile.tiff"), "threshold": threshold})
    assert response.status_code == 400
    assert "threshold" in response.json["error"]


def test_non_geotiff_filename_is_rejected(client):
    response = client.post("/predict", data={"file": (io.BytesIO(b"image"), "image.png")})
    assert response.status_code == 400


def test_upload_limit_returns_json(client, monkeypatch):
    monkeypatch.setitem(endpoint.app.config, "MAX_CONTENT_LENGTH", 64)
    response = client.post("/predict", data={"file": (io.BytesIO(b"x" * 100), "tile.tiff")})
    assert response.status_code == 413
    assert "limit" in response.json["error"]


def test_success_passes_threshold_and_removes_temp_upload(client, monkeypatch, tmp_path):
    seen = {}
    def stub(**kwargs):
        seen.update(kwargs)
        assert Path(kwargs["tiff_path"]).read_bytes() == b"test tile"
        return {"schema_version": "1.1", "status": "CLEAR"}
    monkeypatch.setattr(endpoint, "run_inference", stub)
    monkeypatch.setattr(endpoint, "RESULTS", tmp_path)
    response = client.post("/predict", data={"file": (io.BytesIO(b"test tile"), "tile.tiff"), "threshold": "0.7"})
    assert response.status_code == 200
    assert response.json["status"] == "CLEAR"
    assert seen["confidence_threshold"] == 0.7
    assert not Path(seen["tiff_path"]).exists()


def test_input_validation_error_returns_422_and_cleans_up(client, monkeypatch):
    seen = {}
    def stub(**kwargs):
        seen.update(kwargs)
        raise ValueError("Unsupported tile shape")
    monkeypatch.setattr(endpoint, "run_inference", stub)
    response = client.post("/predict", data={"file": (io.BytesIO(b"bad"), "tile.tiff")})
    assert response.status_code == 422
    assert not Path(seen["tiff_path"]).exists()


def test_corrupt_raster_returns_422(client, monkeypatch):
    def stub(**kwargs):
        raise RasterioIOError("Not a readable GeoTIFF")
    monkeypatch.setattr(endpoint, "run_inference", stub)
    response = client.post("/predict", data={"file": (io.BytesIO(b"bad"), "tile.tiff")})
    assert response.status_code == 422
