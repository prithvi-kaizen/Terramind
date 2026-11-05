"""One model and one active prediction per worker, with content-keyed invalidation."""
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import threading

import numpy as np
import torch

from .model import TerraMindMethanePulse
from .preprocessing import load_and_preprocess, PREPROCESSING_VERSION
from .artifacts import write_artifacts
from .candidates import CandidateConfig

ARCHITECTURE_VERSION = "terramind-small-swir-heads-v1"
_LOCK = threading.RLock()
_cached_key = None
_cached_model = None
_load_count = 0
_forward_count = 0


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def get_device():
    configured = os.environ.get('METHANEPULSE_DEVICE')
    if configured:
        if configured not in ('cpu', 'cuda', 'mps'):
            raise ValueError('METHANEPULSE_DEVICE must be cpu, cuda, or mps')
        return torch.device(configured)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _build_model(checkpoint, device):
    from terratorch import BACKBONE_REGISTRY
    encoder = BACKBONE_REGISTRY.build('terramind_v1_small', pretrained=False,
        modalities=['S2L2A'], bands={'S2L2A': ['SWIR_1', 'SWIR_2']}).to(device).float()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    model = TerraMindMethanePulse(encoder).to(device).float()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


def clear_model_cache():
    global _cached_key, _cached_model, _load_count, _forward_count
    with _LOCK:
        _cached_key = _cached_model = None
        _load_count = _forward_count = 0


def cache_info():
    with _LOCK:
        return {'entries': int(_cached_model is not None), 'loads': _load_count,
                'forward_passes': _forward_count, 'key': _cached_key}


def _get_model(checkpoint, device):
    global _cached_key, _cached_model, _load_count
    # Hash the actual file on each request so replacements at the same path invalidate.
    key = (sha256(checkpoint), str(device), ARCHITECTURE_VERSION, PREPROCESSING_VERSION)
    if key != _cached_key:
        _cached_model = None  # evict before allocating a replacement
        _cached_key = None
        model = _build_model(checkpoint, device)
        if sha256(checkpoint) != key[0]:
            raise RuntimeError('Checkpoint changed during model loading; retry with a stable file')
        _cached_model, _cached_key = model, key
        _load_count += 1
    return _cached_model


@dataclass
class Prediction:
    probability: np.ndarray
    mask: np.ndarray
    severity_class: int
    severity_confidence: float
    metadata: tuple
    model: object


def _predict(tiff_path, checkpoint_path, threshold, device):
    global _forward_count
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('threshold must be a finite number between 0 and 1 (exclusive)')
    x, *metadata = load_and_preprocess(tiff_path, device)
    model = _get_model(checkpoint_path, device)
    with torch.inference_mode():
        logits, severity = model(x)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        cls = int(severity.argmax(dim=1).item())
        confidence = float(torch.softmax(severity, dim=1).max().item())
    _forward_count += 1
    return Prediction(prob, (prob > threshold).astype(np.uint8), cls, confidence, tuple(metadata), model)


def predict_tile(tiff_path, checkpoint_path, threshold=0.5, device=None):
    """Predict without writing files; serialize loading and execution across adapters."""
    with _LOCK:
        return _predict(tiff_path, checkpoint_path, threshold, torch.device(device) if device else get_device())


def run_inference(tiff_path, checkpoint_path, output_dir, confidence_threshold=0.5, device=None, candidate_config=None):
    if candidate_config is not None and not isinstance(candidate_config,CandidateConfig):
        raise ValueError('candidate_config must be a CandidateConfig instance')
    with _LOCK:
        prediction = _predict(tiff_path, checkpoint_path, confidence_threshold,
                              torch.device(device) if device else get_device())
        return write_artifacts(tiff_path, output_dir, confidence_threshold,
            prediction.probability, prediction.mask, prediction.severity_class,
            prediction.severity_confidence, *prediction.metadata, prediction.model, identity={"checkpoint_sha256": _cached_key[0],
                "architecture_version": ARCHITECTURE_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
                "training_run_association": "unknown"}, candidate_config=candidate_config)
