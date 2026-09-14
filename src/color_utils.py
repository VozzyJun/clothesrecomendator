"""
color_utils.py
==============
Small shared color-science helpers used by both the offline garment
indexer (`garment_indexer.py`) and the interactive "My Closet" upload
flow (`app.py` / `closet_engine.py`):

  * dominant_lab_color   - KMeans dominant-color extraction from a crop
  * crop_by_margins      - percentage-based rectangle crop (for the
                            slider-based cropper in the Streamlit UI)
  * hue_chroma            - Lab a*/b* -> hue angle + chroma (saturation)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans

from src.skin_analyzer import SkinAnalyzer

Lab = Tuple[float, float, float]
RGB = Tuple[float, float, float]


def dominant_lab_color(
    crop_rgb: np.ndarray, k: int = 3, max_samples: int = 2000, random_state: int = 42
) -> dict:
    """Find the dominant color of an RGB image crop via KMeans, and return
    it in several representations.

    Returns a dict: {"rgb": (r,g,b) floats, "hex": "#rrggbb", "lab": (L,a,b)}
    """
    pixels = crop_rgb.reshape(-1, 3).astype(np.float64)
    if len(pixels) == 0:
        rgb = (128.0, 128.0, 128.0)
    else:
        if len(pixels) > max_samples:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(pixels), size=max_samples, replace=False)
            pixels = pixels[idx]

        k_eff = min(k, len(pixels))
        if k_eff < 1:
            rgb = tuple(pixels.mean(axis=0))
        else:
            km = KMeans(n_clusters=k_eff, n_init=4, random_state=random_state)
            labels = km.fit_predict(pixels)
            counts = np.bincount(labels)
            dominant_idx = int(np.argmax(counts))
            rgb = tuple(float(c) for c in km.cluster_centers_[dominant_idx])

    lab = SkinAnalyzer.rgb_to_lab(rgb)
    hex_code = "#{:02x}{:02x}{:02x}".format(
        *[int(round(max(0, min(255, c)))) for c in rgb]
    )
    return {"rgb": tuple(round(c, 2) for c in rgb), "hex": hex_code, "lab": tuple(round(c, 3) for c in lab)}


def crop_by_margins(
    image: np.ndarray,
    left_pct: float,
    right_pct: float,
    top_pct: float,
    bottom_pct: float,
) -> np.ndarray:
    """Crop an image using percentage margins (0-100) from each edge."""
    h, w = image.shape[:2]
    x0 = int(w * left_pct / 100.0)
    x1 = int(w * (1 - right_pct / 100.0))
    y0 = int(h * top_pct / 100.0)
    y1 = int(h * (1 - bottom_pct / 100.0))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return image
    return image[y0:y1, x0:x1]


def hue_chroma(lab: Lab) -> Tuple[float, float]:
    """Convert Lab a*/b* to a hue angle (0-360 degrees) and chroma
    (saturation magnitude)."""
    _, a, b = lab
    chroma = math.hypot(a, b)
    hue = math.degrees(math.atan2(b, a))
    if hue < 0:
        hue += 360.0
    return hue, chroma
