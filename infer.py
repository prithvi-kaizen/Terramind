"""
MethanePulse — Inference Script
Usage:
    python infer.py --input sample_input/sample.tiff
    python infer.py --input sample_input/sample.tiff --checkpoint terramind_methane_best.pt --output results/
"""

import argparse, json, uuid, sys
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.warp import transform as transform_coordinates
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    if torch.backends.mps.is_available():  return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")

class TerraMindMethanePulse(nn.Module):
    def __init__(self, encoder, embed_dim=384, img_size=512):
        super().__init__()
        self.encoder  = encoder
        self.img_size = img_size
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  kernel_size=2, stride=2),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        B = x.shape[0]
        x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            features = self.encoder(x_224)
        feat     = features[-1]
        feat_map = feat.permute(0, 2, 1).contiguous().reshape(B, 384, 14, 14)
        seg_logits = self.seg_decoder(feat_map)
        seg_logits = F.interpolate(seg_logits, size=(512, 512),
                                   mode='bilinear', align_corners=False)
        pooled          = feat.mean(dim=1)
        severity_logits = self.severity_head(pooled)
        return seg_logits, severity_logits


def load_and_preprocess(tiff_path, device):
    with rasterio.open(tiff_path) as src:
        if src.count != 13 or (src.height, src.width) != (512, 512):
            raise ValueError("Expected a 13-band, 512x512 Sentinel-2 GeoTIFF in the documented band order")
        if src.crs is None:
            raise ValueError("Input GeoTIFF must declare its coordinate reference system")
        img       = src.read().astype(np.float32)
        if not np.isfinite(img).all():
            raise ValueError("Input GeoTIFF contains non-finite values")
        crs       = str(src.crs)
        bounds    = src.bounds
        transform = src.transform
        profile   = src.profile
    eps = 1e-6
    b11, b12 = img[11], img[12]
    def norm(band):
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        return np.clip((band - lo) / (hi - lo + eps), 0, 1).astype(np.float32)
    x = torch.tensor(np.stack([norm(b11), norm(b12)], axis=0),
                     dtype=torch.float32).unsqueeze(0).to(device)
    return x, crs, bounds, transform, profile


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


def run_inference(tiff_path, checkpoint_path, output_dir, confidence_threshold=0.5):
    if not math.isfinite(confidence_threshold) or not 0 < confidence_threshold < 1:
        raise ValueError("threshold must be a finite number between 0 and 1 (exclusive)")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = get_device()
    print(f"Device       : {device}")
    print(f"Input tile   : {tiff_path}")
    print(f"Checkpoint   : {checkpoint_path}")
    x, crs, bounds, transform, profile = load_and_preprocess(tiff_path, device)

    # ── Load TerraMind via BACKBONE_REGISTRY ──────────────────────
    print("Loading TerraMind encoder...")
    try:
        from terratorch import BACKBONE_REGISTRY
        terramind = BACKBONE_REGISTRY.build(
            'terramind_v1_small',
            # The included checkpoint supplies the complete encoder and head state.
            pretrained=False,
            modalities=['S2L2A'],
            bands={'S2L2A': ['SWIR_1', 'SWIR_2']}   # 2-band subset matching checkpoint
        ).to(device).float()
        for p in terramind.parameters():
            p.requires_grad = False
        terramind.eval()
        print("TerraMind encoder loaded")
    except Exception as e:
        print(f"Failed to load TerraMind: {e}")
        raise RuntimeError(f"Failed to load TerraMind encoder: {e}") from e

    # ── Build model + load weights ────────────────────────────────
    model = TerraMindMethanePulse(encoder=terramind, embed_dim=384, img_size=512)
    model = model.to(device).float()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print("Weights loaded")

    # ── Forward pass ──────────────────────────────────────────────
    with torch.no_grad():
        seg_logits, sev_logits = model(x)

    prob_map  = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
    pred_mask = (prob_map > confidence_threshold).astype(np.uint8)
    sev_class = int(sev_logits.argmax(dim=1).item())
    sev_conf  = float(torch.softmax(sev_logits, dim=1).max().item())

    plume_px  = int(pred_mask.sum())
    plume_pct = round(100 * plume_px / pred_mask.size, 4)
    mean_prob = round(float(prob_map[pred_mask == 1].mean()) if plume_px > 0 else 0.0, 4)

    centroid_geo, centroid_wgs84 = plume_centroids(pred_mask, transform, crs)

    # ── Save mask .tiff ───────────────────────────────────────────
    out_dir  = Path(output_dir)
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
    return alert


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MethanePulse inference — Sentinel-2 methane plume detection"
    )
    parser.add_argument("--input",      required=True)
    parser.add_argument("--checkpoint", default="terramind_methane_best.pt")
    parser.add_argument("--output",     default="outputs/inference/")
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()
    run_inference(args.input, args.checkpoint, args.output, args.threshold)
