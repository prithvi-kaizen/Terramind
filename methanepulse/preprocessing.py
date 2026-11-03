"""Version 1: unchanged percentile normalization for supported tiles."""
import math
import re
import numpy as np
import rasterio
import torch

PREPROCESSING_VERSION = "swir-percentile-v1"

def load_and_preprocess(tiff_path, device):
    with rasterio.open(tiff_path) as src:
        if src.count != 13 or (src.height, src.width) != (512, 512):
            raise ValueError("Expected a 13-band, 512x512 Sentinel-2 GeoTIFF in the documented band order")
        if src.crs is None:
            raise ValueError("Input GeoTIFF must declare its coordinate reference system")
        if not all(math.isfinite(v) for v in src.transform[:6]) or src.transform.determinant == 0:
            raise ValueError("Input GeoTIFF must have a finite, invertible affine transform")
        expected = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")
        def canonical(name):
            match = re.match(r"^B0?([1-9]|1[0-2]|8A)(?:\s|$)", name or "")
            return "B" + match.group(1) if match else None
        if any(src.descriptions) and tuple(canonical(name) for name in src.descriptions) != expected:
            raise ValueError("Band descriptions do not match the documented Sentinel-2 band order")
        if not src.read_masks().all():
            raise ValueError("Nodata pixels are unsupported by preprocessing version 1; provide a fully valid tile")
        img       = src.read().astype(np.float32)
        if not np.isfinite(img).all():
            raise ValueError("Input GeoTIFF contains non-finite values")
        crs       = str(src.crs)
        bounds    = src.bounds
        transform = src.transform
        profile   = src.profile
    eps = 1e-6
    b11, b12 = img[11], img[12]
    if any(np.ptp(band) == 0 for band in (b11, b12)):
        raise ValueError("Constant SWIR bands are unsupported")
    def norm(band):
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        return np.clip((band - lo) / (hi - lo + eps), 0, 1).astype(np.float32)
    x = torch.tensor(np.stack([norm(b11), norm(b12)], axis=0),
                     dtype=torch.float32).unsqueeze(0).to(device)
    return x, crs, bounds, transform, profile
