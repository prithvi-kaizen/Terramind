"""Evaluate a fixed configuration; requires external labeled data."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methanepulse.evaluation import load_manifest, read_target, counts, scores
from methanepulse.runtime import predict_tile, sha256, ARCHITECTURE_VERSION
from methanepulse.preprocessing import PREPROCESSING_VERSION


def evaluate(manifest, checkpoint, split='validation', threshold=0.5, device='cpu'):
    data = load_manifest(manifest)
    rows = [r for r in data['tiles'] if r['split'] == split]
    if not rows:
        raise ValueError(f'No rows in requested split: {split}')
    total = dict(tp=0, fp=0, fn=0)
    scenes = dict(tp=0, fp=0, fn=0)
    negatives = false_positive_scenes = unverified_negatives = 0
    results = []
    for row in rows:
        target = read_target(row['mask'], row['tile'])
        if row['verified_negative'] and target.any():
            raise ValueError('Verified negative has a positive label')
        prediction = predict_tile(row['tile'], checkpoint, threshold, device)
        item = counts(prediction.mask, target)
        for k in total:
            total[k] += item[k]
        positive, detected = bool(target.any()), bool(prediction.mask.any())
        if positive:
            scenes['tp' if detected else 'fn'] += 1
        elif row['verified_negative']:
            negatives += 1
            false_positive_scenes += int(detected)
            scenes['fp'] += int(detected)
        else:
            unverified_negatives += 1
        results.append({**row, **item, **scores(item), 'scene_detected': detected})
    precision_den, recall_den = scenes['tp'] + scenes['fp'], scenes['tp'] + scenes['fn']
    return {'evaluation_id': uuid.uuid4().hex, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'split': split, 'manifest_sha256': sha256(manifest), 'checkpoint_sha256': sha256(checkpoint),
        'training_run_association': 'unknown', 'architecture': ARCHITECTURE_VERSION,
        'preprocessing': PREPROCESSING_VERSION, 'threshold': threshold, 'device': device,
        'python': platform.python_version(), 'dependencies': {p: importlib.metadata.version(p) for p in ('numpy', 'torch', 'rasterio')},
        'conventions': {'pixel_aggregation': 'global counts; per-tile scores also included',
            'empty_prediction_and_label_score': 1, 'scene_detection': 'any foreground pixel',
            'plume_matching': 'not implemented; scene metrics do not localize individual plumes',
            'unverified_negative_scenes': 'excluded from scene precision and false-positive scene rate'},
        'pixels': {**total, **scores(total)}, 'scenes': {**scenes,
            'precision': scenes['tp']/precision_den if precision_den else None,
            'recall': scenes['tp']/recall_den if recall_den else None,
            'verified_negative_count': negatives, 'unverified_negative_count': unverified_negatives,
            'false_positive_scene_rate': false_positive_scenes/negatives if negatives else None}, 'tiles': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', default='terramind_methane_best.pt')
    parser.add_argument('--split', choices=['train','validation','test'], default='validation')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.checkpoint, args.split, args.threshold)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

if __name__ == '__main__':
    main()
