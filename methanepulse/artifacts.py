"""Georeferenced schema 1.1 output with isolated request directories."""
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import rasterio
from rasterio.warp import transform as transform_coordinates

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


def write_artifacts(tiff_path, output_dir, confidence_threshold, prob_map, pred_mask, sev_class, sev_conf, crs, bounds, transform, profile, model, identity=None):
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

    # ── Save alert JSON ───────────────────────────────────────────
    alert_out = out_dir / f"{stem}_alert.json"
    alert = {
        "alert_id"       : str(uuid.uuid4()),
        "schema_version" : "1.1",
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
