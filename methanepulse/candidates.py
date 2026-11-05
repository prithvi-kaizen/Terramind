"""Four-connected candidates and uncalibrated, explicitly configured review signals."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from pyproj import CRS, Geod, Transformer
from rasterio.features import shapes
from scipy import ndimage

VERSION = 'candidates-v1'
FILTER_FLAGS = ('small_area', 'edge_contact', 'threshold_sensitive')
GEOD = Geod(ellps='WGS84')


@dataclass(frozen=True)
class CandidateConfig:
    min_area_m2: float = 1600.0
    threshold_delta: float = 0.05
    sensitive_fraction: float = 0.25
    filter_flags: tuple = ()
    max_candidates: int = 4096

    def __post_init__(self):
        if not math.isfinite(self.min_area_m2) or self.min_area_m2 < 0:
            raise ValueError('min_area_m2 must be finite and non-negative')
        if not math.isfinite(self.threshold_delta) or not 0 < self.threshold_delta < .5:
            raise ValueError('threshold_delta must be between 0 and 0.5')
        if not math.isfinite(self.sensitive_fraction) or not 0 <= self.sensitive_fraction <= 1:
            raise ValueError('sensitive_fraction must be between 0 and 1')
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 16384:
            raise ValueError('max_candidates must be an integer from 1 to 16384')
        for name in ('min_area_m2','threshold_delta','sensitive_fraction'):
            object.__setattr__(self,name,float(getattr(self,name)))
        flags = tuple(sorted(set(self.filter_flags)))
        if any(flag not in FILTER_FLAGS for flag in flags):
            raise ValueError('Unsupported experimental filter flag')
        object.__setattr__(self, 'filter_flags', flags)

    def manifest(self, threshold):
        data = {'version': VERSION, 'connectivity': 4, 'threshold': float(threshold), **asdict(self)}
        return {**data, 'config_sha256': hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()}


def geographic_geometry(geometry, transformer):
    """Transform native polygon rings and enforce GeoJSON exterior/hole orientation."""
    polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
    converted, total_area = [], 0.0
    for polygon in polygons:
        rings = []
        for index, ring in enumerate(polygon):
            x, y = np.asarray(ring, dtype=float).T
            lon, lat = transformer.transform(x, y, errcheck=True)
            if not np.isfinite(lon).all() or not np.isfinite(lat).all() or np.any(np.abs(lat)>90) or np.any(np.abs(lon)>180):
                raise ValueError('Candidate cannot be represented in WGS84')
            area, _ = GEOD.polygon_area_perimeter(lon, lat)
            coords = np.column_stack([lon, lat]).tolist()
            if (index == 0 and area < 0) or (index > 0 and area > 0):
                coords.reverse()
            rings.append(coords)
            total_area += abs(area) if index == 0 else -abs(area)
        converted.append(rings)
    result = {'type': geometry['type'], 'coordinates': converted[0] if geometry['type'] == 'Polygon' else converted}
    return result, max(0.0, float(total_area))


def extract_candidates(probability, transform, crs, threshold=.5, config=None, valid_mask=None):
    config = config or CandidateConfig()
    probability = np.asarray(probability)
    if probability.ndim != 2 or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Probability map must be a finite 2D array in [0, 1]')
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Threshold must be between 0 and 1')
    if transform.determinant == 0 or not all(math.isfinite(v) for v in transform[:6]):
        raise ValueError('Affine transform must be finite and invertible')
    valid = np.ones(probability.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != probability.shape:
        raise ValueError('Valid-pixel mask shape differs')
    raw = probability > threshold
    labels, count = ndimage.label(raw)  # default structure: four-neighbour connectivity
    if count > config.max_candidates:
        raise ValueError(f'Mask has {count} regions, exceeding the {config.max_candidates} candidate budget; inspect the raw result or use a higher threshold')
    slices = ndimage.find_objects(labels)
    transformer = Transformer.from_crs(CRS.from_user_input(crs), 'EPSG:4326', always_xy=True)
    geometry_by_id = {int(value): geometry for geometry, value in shapes(labels.astype('int32'), mask=raw, transform=transform, connectivity=4)}
    candidates, features = [], []
    filtered = raw & valid
    lower, upper = max(0., threshold-config.threshold_delta), min(1., threshold+config.threshold_delta)
    for label_id, section in enumerate(slices, start=1):
        ys, xs = section
        local = labels[section] == label_id
        yy, xx = np.where(local)
        values = probability[section][local]
        geometry, area = geographic_geometry(geometry_by_id[label_id], transformer)
        native_geometry = geometry_by_id[label_id]
        coordinates = np.asarray([point for ring in native_geometry['coordinates'] for point in ring])
        x, y = transform * (float(xx.mean()+xs.start)+.5, float(yy.mean()+ys.start)+.5)
        lon, lat = transformer.transform(x, y, errcheck=True)
        # One-pixel neighbourhood measures local expansion at the lower threshold.
        window = (slice(max(0,ys.start-1),min(raw.shape[0],ys.stop+1)), slice(max(0,xs.start-1),min(raw.shape[1],xs.stop+1)))
        component = labels[window] == label_id
        support = ndimage.binary_dilation(component) & ((labels[window] == 0) | component) & valid[window]
        support |= component & valid[window]
        low_count = int(((probability[window] > lower) & support).sum())
        high_count = int(((probability[window] > upper) & component & valid[window]).sum())
        sensitivity = 1-high_count/max(low_count,1)
        invalid = int((local & ~valid[section]).sum())
        flags = []
        if invalid: flags.append('invalid_pixel_overlap')
        if ys.start == 0 or xs.start == 0 or ys.stop == raw.shape[0] or xs.stop == raw.shape[1]: flags.append('edge_contact')
        if area < config.min_area_m2: flags.append('small_area')
        if sensitivity > config.sensitive_fraction: flags.append('threshold_sensitive')
        removed = [flag for flag in flags if flag in config.filter_flags]
        if removed:
            view = filtered[section];view[local] = False
        candidate = {'source_crs': CRS.from_user_input(crs).to_string(), 'id': f'candidate-{label_id:04d}', 'label_id': label_id, 'pixel_count': int(local.sum()),
            'area_m2': area, 'area_method': 'WGS84 ellipsoidal polygon area, holes subtracted',
            'bbox_pixels': [xs.start,ys.start,xs.stop,ys.stop],
            'bbox_source': [float(coordinates[:,0].min()),float(coordinates[:,1].min()),float(coordinates[:,0].max()),float(coordinates[:,1].max())],
            'centroid_source': [x,y], 'centroid_wgs84': [lon,lat],
            'probability': {'mean': float(values.mean()), 'min': float(values.min()), 'max': float(values.max()), 'calibrated': False},
            'quality_flags': flags, 'evidence_state': 'needs_review' if flags else 'candidate',
            'invalid_pixel_count': invalid,
            'threshold_sensitivity': {'lower': lower,'upper': upper,'local_lower_pixel_count':low_count,'upper_surviving_pixel_count':high_count,'fraction':float(sensitivity)},
            'removed_by_experimental_filter': bool(removed), 'removal_reasons':removed,
            'geometry_wgs84': geometry}
        candidates.append(candidate)
        features.append({'type':'Feature','id':candidate['id'],'geometry':geometry,
                         'properties':{k:v for k,v in candidate.items() if k!='geometry_wgs84'}})
    return {'candidates':candidates, 'geojson':{'type':'FeatureCollection','features':features},
            'labels':labels.astype('int32'), 'filtered_mask':filtered.astype('uint8'),
            'config':config.manifest(threshold),
            'validity':{'invalid_pixel_count':int((~valid).sum()),'raw_foreground_on_invalid_pixels':int((raw&~valid).sum()),
                        'handling':'invalid pixels excluded from filtered mask independently of heuristic rules'},
            'threshold_counts':{'lower':int((probability>lower).sum()),'raw':int(raw.sum()),'upper':int((probability>upper).sum())}}
