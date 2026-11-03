"""Explicit binary segmentation conventions and leak-checked split manifests."""
import json
from pathlib import Path
import numpy as np
import rasterio


def counts(prediction, target):
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError('Prediction/label shapes differ')
    return {'tp': int((prediction & target).sum()),
            'fp': int((prediction & ~target).sum()), 'fn': int((~prediction & target).sum())}


def scores(values):
    tp, fp, fn = (values[k] for k in ('tp', 'fp', 'fn'))
    return {'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.}


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1 or not data.get('tiles'):
        raise ValueError('Expected manifest schema_version 1 and nonempty tiles')
    seen, groups, content = set(), {}, {}
    from .runtime import sha256
    for row in data['tiles']:
        required = ('id', 'split', 'tile', 'mask', 'location', 'acquisition', 'label_source', 'verified_negative')
        if any(key not in row for key in required) or row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Incomplete manifest row or unsupported split')
        if row['id'] in seen or type(row['verified_negative']) is not bool:
            raise ValueError('Duplicate ID or invalid negative verification')
        seen.add(row['id'])
        for name in ('location', 'acquisition', 'plume_id'):
            value = row.get(name)
            if name != 'plume_id' and not value:
                raise ValueError('Location and acquisition identifiers are required')
            if value:
                key = name, str(value)
                if key in groups and groups[key] != row['split']:
                    raise ValueError(f'Split leakage for {name}: {value}')
                groups[key] = row['split']
        for name in ('tile', 'mask'):
            item = (path.parent / row[name]).resolve()
            digest = sha256(item)
            key = name, digest
            if name == 'tile' and key in content:
                raise ValueError(f'Duplicate {name} content across manifest rows')
            content[key] = row['id']
            if row.get(name + '_sha256') and row[name + '_sha256'] != digest:
                raise ValueError(f'{name} checksum mismatch')
            row[name] = str(item)
            row[name + '_sha256'] = digest
    return data


def read_target(path, tile_path):
    with rasterio.open(path) as label, rasterio.open(tile_path) as tile:
        if label.count != 1 or label.shape != tile.shape or label.crs != tile.crs or label.transform != tile.transform:
            raise ValueError('Label must be a single band aligned with tile shape/CRS/transform')
        target = label.read(1)
        if not label.read_masks(1).all() or not np.isfinite(target).all() or not np.isin(target, [0, 1]).all():
            raise ValueError('Labels must be fully valid binary 0/1 masks')
        return target.astype(bool)
