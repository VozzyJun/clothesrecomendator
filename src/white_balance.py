"""
white_balance.py
=================
Automatic White Balance (AWB) correction applied to a photo *before* skin
sampling, so a camera with a color cast (common on cheap webcams, or under
tinted indoor lighting) doesn't distort the measured skin tone as badly as
an uncorrected shot would.

Two classic illuminant-estimation algorithms are implemented from scratch
(no extra dependency, consistent with the rest of this project):

  * **Face-excluded Gray World** -- the preferred method whenever a face
    bounding box is available. It estimates the color cast from the
    *background* pixels only, so the correction targets the camera's
    white balance rather than the skin's own (real) warm/cool undertone.

    This distinction matters a lot for this app specifically: a selfie
    frame is mostly skin. Running plain whole-image gray-world (as most
    "auto white balance" demos do) would try to neutralize the average
    color of the photo toward gray -- which means it would partially
    neutralize the skin's *real* undertone right along with the camera's
    color cast. That's the opposite of what we want, since undertone is
    exactly what we're trying to measure. Estimating the cast from
    everything *except* the face sidesteps this.

  * **Shades of Gray** (Minkowski p-norm, p=6) -- a whole-image fallback
    used only when there isn't enough background to trust (e.g. a tight
    face crop that fills most of the frame). It's less prone than plain
    mean-based gray world to being dominated by one large, uniformly
    colored region (van de Weijer et al., "Edge-Based Color Constancy",
    2007), though it is still a whole-image estimate and therefore a
    weaker choice than the face-excluded method above.

Both methods are gain-clamped and applied at partial strength (blended
with the original) so an unusual background or scene can't overcorrect
the photo.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Per-channel correction gain is clamped to this range so an unusual
# background/scene can't swing the whole photo too far.
GAIN_MIN, GAIN_MAX = 0.6, 1.6

# How strongly to apply the estimated correction (1.0 = full correction,
# 0.0 = no correction). Partial strength meaningfully improves skin
# sampling while limiting the risk of overcorrecting a real camera cast
# into a different, equally wrong cast.
DEFAULT_STRENGTH = 0.65

# A background region needs at least this many pixels to be trusted for
# face-excluded gray-world; otherwise we fall back to shades-of-gray.
MIN_BACKGROUND_PIXELS = 5000

BGR = Tuple[float, float, float]


def _clamp_gains(gains: np.ndarray) -> np.ndarray:
    return np.clip(gains, GAIN_MIN, GAIN_MAX)


def _apply_gains(image_bgr: np.ndarray, gains_bgr: np.ndarray, strength: float) -> np.ndarray:
    """Blend `strength` of the full correction into the image."""
    blended_gains = 1.0 + strength * (gains_bgr - 1.0)
    corrected = image_bgr.astype(np.float64) * blended_gains.reshape(1, 1, 3)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def shades_of_gray_gains(image_bgr: np.ndarray, p: int = 6) -> np.ndarray:
    """Estimate per-channel (B, G, R) correction gains using the
    'Shades of Gray' Minkowski p-norm illuminant estimate over the whole
    image."""
    pixels = image_bgr.reshape(-1, 3).astype(np.float64)
    p_mean = (np.mean(pixels ** p, axis=0)) ** (1.0 / p)
    overall = float(np.mean(p_mean))
    gains = overall / np.maximum(p_mean, 1e-6)
    return _clamp_gains(gains)


def _saturation(mean_bgr: np.ndarray) -> float:
    """Cheap 0-1 'how colorful is this average color' proxy (HSV-style
    saturation of a single averaged color)."""
    b, g, r = mean_bgr
    channel_max, channel_min = max(b, g, r), min(b, g, r)
    if channel_max < 1e-6:
        return 0.0
    return float((channel_max - channel_min) / channel_max)


def _background_confidence(background_pixels: np.ndarray) -> float:
    """Estimate how much to trust a gray-world correction derived from
    this background.

    A fairly neutral-looking background (walls, plain floor/backdrop) is
    a decent stand-in for "should average out to gray." A strongly
    colorful background (a red curtain, a mustard sweater filling the
    frame) is not -- gray-world would just push the whole photo toward
    canceling out that background color, which is the wrong correction
    to make. This keeps that failure mode from doing real damage instead
    of silently trusting every background equally.

    Returns a 0.15-1.0 multiplier (never fully zero, since a little
    correction is usually still better than none).
    """
    mean_bgr = background_pixels.mean(axis=0)
    sat = _saturation(mean_bgr)
    # Saturation 0 (neutral-looking background) -> full confidence.
    # Saturation >= ~0.35 (clearly colorful background) -> minimum trust.
    confidence = 1.0 - min(sat / 0.35, 1.0)
    return max(confidence, 0.15)


def face_excluded_gray_world_gains(
    image_bgr: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    margin: float = 0.25,
) -> Optional[Tuple[np.ndarray, float]]:
    """Estimate per-channel (B, G, R) correction gains from the pixels
    *outside* an (expanded) face bounding box.

    Returns `(gains, confidence)`, or None if too few background pixels
    remain to be trustworthy (e.g. a very tight face crop), so the caller
    can fall back to a whole-image estimate instead. `confidence` (see
    `_background_confidence`) reflects how neutral that background
    actually looks, and should be used to damp the correction strength.
    """
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = face_bbox
    dx, dy = int(bw * margin), int(bh * margin)
    x0, y0 = max(x - dx, 0), max(y - dy, 0)
    x1, y1 = min(x + bw + dx, w), min(y + bh + dy, h)

    mask = np.ones((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = False
    background = image_bgr[mask]

    if len(background) < MIN_BACKGROUND_PIXELS:
        return None

    pixels = background.astype(np.float64)
    channel_means = pixels.mean(axis=0)
    overall = float(channel_means.mean())
    gains = overall / np.maximum(channel_means, 1e-6)
    confidence = _background_confidence(pixels)
    return _clamp_gains(gains), confidence


def auto_white_balance(
    image_bgr: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
    strength: float = DEFAULT_STRENGTH,
) -> Tuple[np.ndarray, dict]:
    """Correct a likely camera color cast before skin sampling.

    Prefers estimating the cast from background pixels so the face's own
    true undertone is left untouched by the correction; falls back to a
    whole-image 'Shades of Gray' estimate when there isn't a usable
    background (e.g. the face fills almost the entire frame).

    Returns `(corrected_image_bgr, info)` where `info` describes which
    method ran and the gains applied -- surfaced in the UI so the person
    can see (and, if they like, disable) what correction was made.
    """
    method = "none"
    gains = np.array([1.0, 1.0, 1.0])
    effective_strength = strength

    if face_bbox is not None:
        result = face_excluded_gray_world_gains(image_bgr, face_bbox)
        if result is not None:
            gains, confidence = result
            effective_strength = strength * confidence
            method = "face_excluded_gray_world"

    if method == "none":
        gains = shades_of_gray_gains(image_bgr)
        effective_strength = strength
        method = "shades_of_gray"

    corrected = _apply_gains(image_bgr, gains, effective_strength)
    info = {
        "method": method,
        "gains_bgr": tuple(round(float(g), 3) for g in gains),
        "strength": round(effective_strength, 3),
    }
    return corrected, info
