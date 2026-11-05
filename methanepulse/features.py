"""Content-addressed frozen features; only fixed, declared augmentations are cached."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .preprocessing import PREPROCESSING_VERSION, load_and_preprocess
from .runtime import ARCHITECTURE_VERSION, sha256

FEATURE_VERSION = 'frozen-tokens-v1'
AUGMENTATIONS = ('identity', 'flip-x')


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class FeatureCache:
    def __init__(self, directory, checkpoint, device='cpu'):
        self.directory = Path(directory)
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha = sha256(checkpoint)
        self.device = torch.device(device)
        self.encoder_passes = 0
        self.hits = 0
        self.seconds = 0.
        self._encoder_signature = None
        self._encoder_sha = None

    def identity(self, tile, augmentation, model):
        if augmentation not in AUGMENTATIONS:
            raise ValueError('Unsupported feature augmentation')
        if sha256(self.checkpoint) != self.checkpoint_sha:
            raise ValueError('Checkpoint changed during cache session')
        state = model.encoder.state_dict()
        signature = tuple((name, value.data_ptr(), value._version) for name,value in state.items())
        if signature != self._encoder_signature:
            digest = hashlib.sha256()
            for name,value in state.items():
                digest.update(name.encode())
                digest.update(str((tuple(value.shape),value.dtype)).encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            self._encoder_sha, self._encoder_signature = digest.hexdigest(), signature
        return dict(version=FEATURE_VERSION, input_sha256=sha256(tile), encoder_sha256=self._encoder_sha,
                    checkpoint_sha256=self.checkpoint_sha, architecture=ARCHITECTURE_VERSION,
                    preprocessing=PREPROCESSING_VERSION, bands=['B11', 'B12'],
                    augmentation=augmentation, torch=torch.__version__, device=str(self.device))

    def get(self, tile, model, augmentation='identity'):
        start = time.perf_counter()
        identity = self.identity(tile, augmentation, model)
        path = self.directory / (digest_json(identity) + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(str(saved['identity'])) != identity:
                    raise ValueError('Feature cache identity mismatch')
                values = saved['features'].copy()
            self.hits += 1
        else:
            x, *_ = load_and_preprocess(tile, self.device)
            if augmentation == 'flip-x':
                x = torch.flip(x, [-1])
            model.encoder.eval()
            with torch.no_grad():
                values = model.encode(x).detach().cpu().numpy()
            self.encoder_passes += 1
            self._validate(values)
            if self.identity(tile, augmentation, model) != identity:
                raise ValueError('Input changed during feature extraction')
            atomic_npz(path, features=values, identity=np.asarray(json.dumps(identity, sort_keys=True)))
        self._validate(values)
        self.seconds += time.perf_counter() - start
        return torch.from_numpy(values).to(self.device)

    @staticmethod
    def _validate(values):
        if values.shape != (1, 196, 384) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('Invalid cached frozen feature tensor')

    def report(self):
        files = list(self.directory.glob('*.npz'))
        return dict(encoder_passes=self.encoder_passes, cache_hits=self.hits,
                    cache_seconds=self.seconds, cache_bytes=sum(p.stat().st_size for p in files),
                    cache_entries=len(files), version=FEATURE_VERSION)
