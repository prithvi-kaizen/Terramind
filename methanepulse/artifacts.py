"""Georeferenced schema 1.1 output with isolated request directories."""
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import rasterio
from rasterio.warp import transform as transform_coordinates
from .candidates import extract_candidates

SEVERITY_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
SEVERITY_COLOR  = {0: "#f5c518", 1: "#ff8c00", 2: "#e03131"}

def plume_centroids(pred_mask, affine_transform, crs):
    """Return projected and WGS84 centroids at raster pixel centres."""
    ys, xs = np.where(pred_mask == 1)
    if not xs.size:
        return None, None
    x, y = affine_transform * (float(xs.mean()) + 0.5, float(ys.mean()) + 0.5)
    lons, lats = transform_coordinates(crs, "EPSG:4326", [x], [y])
    return [round(x, 6), round(y, 6)], [round(lons[0], 6), round(lats[0], 6)]


def write_artifacts(tiff_path, output_dir, confidence_threshold, prob_map, pred_mask, sev_class, sev_conf, crs, bounds, transform, profile, model, identity=None, candidate_config=None):
    plume_px  = int(pred_mask.sum())
    plume_pct = round(100 * plume_px / pred_mask.size, 4)
    mean_prob = round(float(prob_map[pred_mask == 1].mean()) if plume_px > 0 else 0.0, 4)

    centroid_geo, centroid_wgs84 = plume_centroids(pred_mask, transform, crs)

    # ── Save mask .tiff ───────────────────────────────────────────
    out_dir  = Path(output_dir) / uuid.uuid4().hex
    out_dir.mkdir(parents=True, exist_ok=True)
    stem     = Path(tiff_path).stem
    mask_profile = profile.copy()
    mask_profile.update({"count": 1, "dtype": "uint8", "nodata": None})
    if not mask_profile.get("tiled"):
        mask_profile.pop("blockxsize", None)
        mask_profile.pop("blockysize", None)
    mask_out = out_dir / f"{stem}_mask.tiff"
    with rasterio.open(mask_out, "w", **mask_profile) as dst:
        dst.write(pred_mask[np.newaxis, :, :])
    print(f"Mask saved  → {mask_out}")

    review = extract_candidates(prob_map, transform, crs, confidence_threshold, candidate_config)
    paths = {}
    for key, suffix, array, dtype in [
        ('probability_path', 'probability', prob_map, 'float32'),
        ('candidate_labels_path', 'candidates', review['labels'], 'int32'),
        ('filtered_mask_path', 'filtered_mask', review['filtered_mask'], 'uint8'),
    ]:
        path = out_dir / f'{stem}_{suffix}.tiff'
        extra_profile = {**mask_profile, 'dtype': dtype}
        with rasterio.open(path, 'w', **extra_profile) as dst:
            dst.write(array.astype(dtype), 1)
        paths[key] = str(path.resolve())
    review['geojson']['provenance'] = {'model': identity or {}, 'postprocessing': review['config'],
        'geometry_crs': 'EPSG:4326', 'coordinate_order':'longitude,latitude',
        'pixel_bounds':'[col_min,row_min,col_max_exclusive,row_max_exclusive]'}
    geojson_path = out_dir / f'{stem}_candidates.geojson'
    geojson_path.write_text(json.dumps(review['geojson'], allow_nan=False))
    paths['geojson_path'] = str(geojson_path.resolve())

    # ── Save alert JSON ───────────────────────────────────────────
    alert_out = out_dir / f"{stem}_alert.json"
    alert = {
        "alert_id"       : str(uuid.uuid4()),
        "schema_version" : "2.0",
        "timestamp_utc"  : datetime.now(timezone.utc).isoformat(),
        "source": {
            "tile_path": str(tiff_path), "sensor": "Sentinel-2",
            "bands_used": ["B11 (SWIR_1)", "B12 (SWIR_2)"], "crs": crs,
            "bounds": {"left": round(bounds.left, 6), "bottom": round(bounds.bottom, 6),
                       "right": round(bounds.right, 6), "top": round(bounds.top, 6)}
        },
        "detection": {
            "plume_detected": bool(plume_px > 0),
            "confidence_threshold": confidence_threshold,
            "plume_pixel_count": plume_px, "plume_coverage_pct": plume_pct,
            "mean_plume_prob": mean_prob, "centroid_geographic": centroid_geo,
            "centroid_wgs84": centroid_wgs84
        },
        "severity": {
            "class_id": sev_class, "label": SEVERITY_LABELS[sev_class],
            "confidence": round(sev_conf, 4), "color_hex": SEVERITY_COLOR[sev_class]
        },
        "model": {"name": "TerraMindMethanePulse",
                  "encoder": "TerraMind-1.0-small (frozen)",
                  "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "status": "ALERT" if plume_px > 0 else "CLEAR",
        "artifacts": {"mask_path": str(mask_out.resolve()), "alert_path": str(alert_out.resolve())}
    }
    alert['artifacts'].update(paths)
    alert['candidates'] = review['candidates']
    alert['candidate_count'] = len(review['candidates'])
    alert['postprocessing'] = review['config']
    alert['quality'] = {'validity': review['validity'], 'threshold_counts': review['threshold_counts'],
                        'interpretation': 'Review flags are uncalibrated signals, not confirmation of methane'}
    retained=set(int(value) for value in np.unique(review['labels'][review['filtered_mask'].astype(bool)]))
    alert['filtered'] = {'enabled': bool(review['config']['filter_flags']),
        'plume_pixel_count': int(review['filtered_mask'].sum()),
        'removed_pixel_count': int(pred_mask.sum()-review['filtered_mask'].sum()),
        'retained_candidate_ids': [c['id'] for c in review['candidates'] if c['label_id'] in retained]}
    alert["model"].update(identity or {})
    with open(alert_out, "w") as f:
        json.dump(alert, f, indent=2)
    print(f"Alert saved → {alert_out}")

    print("\n── Inference Summary ──────────────────────────")
    print(f"  Status       : {alert['status']}")
    print(f"  Plume pixels : {plume_px:,}  ({plume_pct}% of tile)")
    print(f"  Severity     : {SEVERITY_LABELS[sev_class]}  (conf: {sev_conf:.3f})")
    if centroid_wgs84:
        print(f"  Centroid WGS84 [lon, lat]: {centroid_wgs84}")
    print("───────────────────────────────────────────────")
    (out_dir / ".completed").touch()
    return alert
