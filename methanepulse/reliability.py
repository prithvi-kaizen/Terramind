"""Opt-in flip agreement and bounded temporal evidence; no leak confirmation."""
from datetime import date
import json
import math
from pathlib import Path

import numpy as np
from pyproj import Geod
import rasterio
from scipy.optimize import linear_sum_assignment
import torch

from . import runtime
from .candidates import CandidateConfig, VERSION, extract_candidates
from .features import atomic_npz, digest_json
from .preprocessing import PREPROCESSING_VERSION, load_and_preprocess

RESULT_VERSION = 'review-predictions-v1'


def agreement(raw, other, threshold):
    first, second = raw > threshold, other > threshold
    union = (first | second).sum()
    return dict(mask_iou=float((first & second).sum() / union) if union else 1.,
                both_empty=not bool(union), mean_absolute_probability_difference=float(np.abs(raw-other).mean()),
                calibrated_uncertainty=False)


def consistency_check(tile, checkpoint, threshold=.5, enabled=False, device='cpu'):
    """One base pass, and exactly one extra horizontal-flip pass when explicitly enabled."""
    with runtime._LOCK:
        base = runtime.predict_tile(tile, checkpoint, threshold, device)
        result = dict(enabled=bool(enabled), base_passes=1, extra_passes=0, signal=None,
                      input_sha256=runtime.sha256(tile),checkpoint_sha256=runtime.cache_info()['key'][0],
                      preprocessing=PREPROCESSING_VERSION,architecture=runtime.ARCHITECTURE_VERSION,
                      threshold=threshold,transform='horizontal flip, inverted before comparison')
        if enabled:
            x, *_ = load_and_preprocess(tile, device)
            with torch.inference_mode():
                logits, _ = base.model(torch.flip(x, [-1]))
                flipped = torch.flip(torch.sigmoid(logits), [-1]).squeeze().cpu().numpy()
            runtime._forward_count += 1
            result.update(extra_passes=1, signal=agreement(base.probability, flipped, threshold))
        return result


def route_consistency(flagged, mode='off'):
    """Input order is deterministic; automatic routing rounds the 10% budget down."""
    if mode not in ('off', 'automatic'):
        raise ValueError('Routing mode must be off or automatic')
    if any(type(item) is not bool for item in flagged):
        raise ValueError('Routing flags must be booleans')
    budget = len(flagged) // 10 if mode == 'automatic' else 0
    routed = 0
    rows = []
    for index, flag in enumerate(flagged):
        selected = flag and routed < budget
        routed += int(selected)
        rows.append(dict(index=index, flagged=flag, selected=selected,
                         reason='within_flagged_budget' if selected else 'off' if mode == 'off' else
                         'not_flagged' if not flag else 'budget_exhausted'))
    return dict(mode=mode, extra_pass_budget=budget, extra_passes=routed,
                maximum_total_passes=len(flagged)+routed, decisions=rows)


def consistency_batch(rows, checkpoint, mode='off', threshold=.5):
    routing = route_consistency([row['flagged'] for row in rows], mode)
    results=[]
    for row,decision in zip(rows,routing['decisions']):
        results.append(dict(tile=str(row['tile']),
                            routing=decision, **consistency_check(row['tile'],checkpoint,threshold,
                                                                 enabled=decision['selected'])))
    return dict(checkpoint_sha256=runtime.sha256(checkpoint), threshold=threshold,routing=routing,
                base_passes=len(rows),extra_passes=sum(r['extra_passes'] for r in results),tiles=results)


class PredictionCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.passes = self.hits = 0

    def get(self, tile, checkpoint, threshold, config, device='cpu'):
        identity = dict(version=RESULT_VERSION, input_sha256=runtime.sha256(tile),
                        checkpoint_sha256=runtime.sha256(checkpoint), architecture=runtime.ARCHITECTURE_VERSION,
                        preprocessing=PREPROCESSING_VERSION, threshold=float(threshold),
                        postprocessing=config.manifest(threshold), torch=torch.__version__, device=str(device))
        identity = json.loads(json.dumps(identity, sort_keys=True))
        path = self.directory / (digest_json(identity)+'.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(str(saved['identity'])) != identity:
                    raise ValueError('Prediction cache identity mismatch')
                probability = saved['probability'].copy()
            self.hits += 1
        else:
            prediction = runtime.predict_tile(tile, checkpoint, threshold, device)
            probability = prediction.probability
            self.passes += 1
            if runtime.sha256(tile) != identity['input_sha256'] or runtime.sha256(checkpoint) != identity['checkpoint_sha256']:
                raise ValueError('Input/checkpoint changed while predicting')
            atomic_npz(path, probability=probability, identity=np.asarray(json.dumps(identity, sort_keys=True)))
        if probability.shape != (512, 512) or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
            raise ValueError('Invalid cached probability map')
        return probability


def match_candidates(first, second, tolerance_m):
    """One-to-one centroid proximity, never a source-location estimate."""
    if not math.isfinite(tolerance_m) or not 0 < tolerance_m <= 1000:
        raise ValueError('Spatial tolerance must be in (0, 1000] metres')
    if not first or not second:
        return []
    if len(first)*len(second) > 1_000_000:
        raise ValueError('Temporal matching exceeds pair budget')
    geod = Geod(ellps='WGS84')
    distances = np.array([[geod.inv(*a['centroid_wgs84'], *b['centroid_wgs84'])[2]
                           for b in second] for a in first])
    # Penalise invalid edges enough to maximise the number of valid pairs first.
    penalty = (min(len(first),len(second))+1) * (tolerance_m+1)
    rows, columns = linear_sum_assignment(np.where(distances<=tolerance_m, distances, penalty))
    return [dict(first_id=first[i]['id'], second_id=second[j]['id'], displacement_m=float(distances[i,j]))
            for i,j in zip(rows,columns) if distances[i,j]<=tolerance_m]


def temporal_review(manifest_path, checkpoint, cache_dir, threshold=.5, tolerance_m=200., config=None):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    observations = manifest.get('observations', [])
    if not manifest.get('location') or not 1 <= len(observations) <= 3:
        raise ValueError('Require one location and one to three explicitly supplied dates')
    config = config or CandidateConfig()
    match_candidates([], [], tolerance_m)
    dates, tiles, grids, signature = [], [], [], None
    contents = set()
    input_hashes = {}
    for row in observations:
        day = date.fromisoformat(row['date'])
        if day in dates:
            raise ValueError('Observation dates must be distinct')
        dates.append(day)
        tile = (path.parent / row['tile']).resolve()
        content = runtime.sha256(tile)
        if content in contents:
            raise ValueError('Duplicate image content cannot establish a distinct observation')
        contents.add(content)
        input_hashes[tile] = content
        # Attested checks are required in addition to equality of raster grids.
        residual = row.get('registration_residual_pixels')
        if type(residual) not in (int,float) or not math.isfinite(residual) or not 0 <= residual <= .25 or not row.get('registration_evidence'):
            raise ValueError('Require reviewed registration evidence with residual at most 0.25 pixels')
        scale = row.get('radiometry', {})
        if scale.get('processing_level') not in ('L1C', 'L2A') or scale.get('band_order_verified') is not True or not scale.get('evidence'):
            raise ValueError('Require documented processing level, scaling and verified band order')
        if not isinstance(scale.get('scale'), (int,float)) or not math.isfinite(scale['scale']) or scale['scale'] <= 0 or not isinstance(scale.get('offset'),(int,float)) or not math.isfinite(scale['offset']):
            raise ValueError('Invalid radiometric scale/offset')
        if signature is None:
            signature = (scale['processing_level'], scale['scale'], scale['offset'])
        elif signature != (scale['processing_level'], scale['scale'], scale['offset']):
            raise ValueError('Dates have incompatible radiometry')
        load_and_preprocess(tile, 'cpu')  # validates all bands, full coverage and finite SWIR
        with rasterio.open(tile) as src:
            grid = (src.crs, src.transform, src.shape)
        if grids and grid != grids[0]:
            raise ValueError('Dates must have identical CRS, transform, resolution and coverage')
        grids.append(grid)
        tiles.append(tile)
    order = sorted(range(len(dates)), key=dates.__getitem__)
    dates, tiles = [dates[i] for i in order], [tiles[i] for i in order]
    cache = PredictionCache(cache_dir)
    checkpoint_identity = runtime.sha256(checkpoint)
    results = []
    for day, tile in zip(dates,tiles):
        if runtime.sha256(checkpoint) != checkpoint_identity or runtime.sha256(tile) != input_hashes[tile]:
            raise ValueError('Temporal inputs changed after validation')
        probability = cache.get(tile, checkpoint, threshold, config)
        with rasterio.open(tile) as src:
            candidates = extract_candidates(probability, src.transform, src.crs, threshold, config)['candidates']
        results.append(dict(date=day.isoformat(), input_sha256=runtime.sha256(tile), candidates=candidates))
    if runtime.sha256(checkpoint) != checkpoint_identity or any(runtime.sha256(tile)!=input_hashes[tile] for tile in tiles):
        raise ValueError('Temporal inputs changed during review')
    pairs = []
    for i in range(len(results)):
        for j in range(i+1,len(results)):
            pairs.append(dict(first_date=results[i]['date'], second_date=results[j]['date'],
                              matches=match_candidates(results[i]['candidates'],results[j]['candidates'],tolerance_m)))
    for observation in results:
        for candidate in observation['candidates']:
            evidence = {observation['date']}
            displacements = []
            for pair in pairs:
                for match in pair['matches']:
                    if observation['date']==pair['first_date'] and candidate['id']==match['first_id']:
                        evidence.add(pair['second_date']); displacements.append(match['displacement_m'])
                    if observation['date']==pair['second_date'] and candidate['id']==match['second_id']:
                        evidence.add(pair['first_date']); displacements.append(match['displacement_m'])
            candidate['temporal_evidence'] = dict(dates=sorted(evidence), observation_count=len(evidence),
                                                  displacements_m=displacements,
                                                  state='temporally_corroborated' if len(evidence)>1 else 'single_observation')
    return dict(schema_version=1, method='pairwise one-to-one WGS84 centroid proximity',
                location=manifest['location'], manifest_sha256=runtime.sha256(path),
                checkpoint_sha256=checkpoint_identity, threshold=threshold,
                postprocessing=config.manifest(threshold), tolerance_m=tolerance_m,
                observations=results, pairs=pairs, date_passes=cache.passes, cache_hits=cache.hits,
                limitations='Proximity corroborates model detections, not methane or an emission source. A missed date is not proof of absence. Registration and radiometry evidence is supplied by the reviewer; grid equality alone cannot verify image alignment.')
