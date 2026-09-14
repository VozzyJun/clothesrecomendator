"""
skin_analyzer.py
================
Uses Google MediaPipe FaceMesh to locate a face in an image, samples skin
pixels from both cheeks, and derives:

  * Mean skin RGB / Hex
  * CIELAB coordinates (L*, a*, b*)
  * Individual Typology Angle (ITA) -> skin tone category
  * Undertone (Warm / Cool / Neutral) from the Lab hue angle

The categories produced here are chosen to line up exactly with the
`Skin Tone` and `Under Tone` columns found in
`data/raw/fashion_and_color_recommendation.csv`:

    Skin Tone : Very Fair, Fair, Medium, Olive, Brown, Very Dark
    Under Tone: Warm, Cool, Neutral

Note: skin-tone/undertone estimation from a single RGB image is inherently
approximate (it is sensitive to white balance and ambient lighting). The
thresholds below follow the commonly cited ITA bands from dermatological
literature (Del Bino & Bernerd) and are meant as a reasonable heuristic for
a styling app rather than a clinical measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe is required for skin_analyzer.py. Install with "
        "`pip install mediapipe`."
    ) from exc


# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe FaceMesh, 468/478-point topology)
# ---------------------------------------------------------------------------
# These are well-documented "flat cheek" landmark points that sit on the
# malar (cheek) region away from hair, eyes, nostrils and specular
# highlights along the nose bridge.
LEFT_CHEEK_IDX = [50, 101, 118, 229, 137]
RIGHT_CHEEK_IDX = [280, 330, 347, 449, 366]

# Used only to estimate a robust face scale so cheek patches scale with
# the size of the face in the frame.
FACE_WIDTH_IDX = (234, 454)  # left face edge, right face edge


@dataclass
class SkinAnalysisResult:
    """Container for a single skin-tone analysis pass."""

    rgb: Tuple[int, int, int]
    hex: str
    lab: Tuple[float, float, float]
    ita: float
    skin_tone: str
    undertone: str
    face_bbox: Tuple[int, int, int, int]  # x, y, w, h in pixel coords
    cheek_points_px: list = field(default_factory=list)
    annotated_bgr: Optional[np.ndarray] = None


class NoFaceDetectedError(RuntimeError):
    """Raised when MediaPipe cannot find a face in the supplied image."""


class SkinAnalyzer:
    """Wraps MediaPipe FaceMesh + CIELAB/ITA computation."""

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self) -> "SkinAnalyzer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Landmark / patch extraction
    # ------------------------------------------------------------------
    def _detect_landmarks(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    @staticmethod
    def _landmarks_to_px(landmarks, w: int, h: int) -> np.ndarray:
        pts = np.array(
            [(lm.x * w, lm.y * h) for lm in landmarks.landmark], dtype=np.float32
        )
        return pts

    @staticmethod
    def _sample_patch(
        image_bgr: np.ndarray, center: Tuple[float, float], radius: int
    ) -> np.ndarray:
        """Return an (N, 3) array of BGR pixel values around `center`."""
        h, w = image_bgr.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        x0, x1 = max(cx - radius, 0), min(cx + radius, w)
        y0, y1 = max(cy - radius, 0), min(cy + radius, h)
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 3))
        patch = image_bgr[y0:y1, x0:x1].reshape(-1, 3)
        return patch

    def _cheek_pixels(
        self, image_bgr: np.ndarray, pts_px: np.ndarray
    ) -> Tuple[np.ndarray, list]:
        x_left, x_right = pts_px[FACE_WIDTH_IDX[0]][0], pts_px[FACE_WIDTH_IDX[1]][0]
        face_width = max(abs(x_right - x_left), 1.0)
        radius = max(int(face_width * 0.045), 4)

        all_pixels = []
        sampled_points = []
        for idx_group in (LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX):
            centers = pts_px[idx_group]
            centroid = centers.mean(axis=0)
            sampled_points.append(tuple(centroid))
            patch = self._sample_patch(image_bgr, centroid, radius)
            if patch.size:
                all_pixels.append(patch)

        if not all_pixels:
            return np.empty((0, 3)), sampled_points
        return np.vstack(all_pixels), sampled_points

    @staticmethod
    def _filter_outlier_pixels(pixels_bgr: np.ndarray) -> np.ndarray:
        """Trim specular highlights / shadows using percentile clipping on
        luminance, keeping the central 60% of samples."""
        if len(pixels_bgr) < 5:
            return pixels_bgr
        luminance = pixels_bgr.mean(axis=1)
        lo, hi = np.percentile(luminance, [20, 80])
        mask = (luminance >= lo) & (luminance <= hi)
        filtered = pixels_bgr[mask]
        return filtered if len(filtered) else pixels_bgr

    # ------------------------------------------------------------------
    # Color science
    # ------------------------------------------------------------------
    @staticmethod
    def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
        c = c / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    @classmethod
    def rgb_to_lab(cls, rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Convert a single sRGB (0-255) triplet to CIELAB (D65)."""
        r, g, b = (np.array(rgb, dtype=np.float64))
        lin = cls._srgb_to_linear(np.array([r, g, b]))
        r_lin, g_lin, b_lin = lin

        # sRGB -> XYZ (D65)
        x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
        y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
        z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

        # Normalize by D65 white point
        xn, yn, zn = 0.95047, 1.00000, 1.08883
        x, y, z = x / xn, y / yn, z / zn

        def f(t):
            delta = 6.0 / 29.0
            return t ** (1.0 / 3.0) if t > delta ** 3 else (t / (3 * delta ** 2)) + 4.0 / 29.0

        fx, fy, fz = f(x), f(y), f(z)
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b_ = 200.0 * (fy - fz)
        return float(L), float(a), float(b_)

    @staticmethod
    def compute_ita(L: float, b: float) -> float:
        """Individual Typology Angle, in degrees.

        ITA = arctan((L* - 50) / b*) * 180 / pi
        """
        if abs(b) < 1e-6:
            b = 1e-6
        return math.degrees(math.atan2((L - 50.0), b))

    @staticmethod
    def classify_skin_tone(ita: float) -> str:
        """Map ITA (degrees) to the six-bucket skin-tone scale used by the
        recommendation dataset, following Del Bino & Bernerd style bands."""
        if ita > 55:
            return "Very Fair"
        if ita > 41:
            return "Fair"
        if ita > 28:
            return "Medium"
        if ita > 19:
            return "Olive"
        if ita > 10:
            return "Brown"
        return "Very Dark"

    @staticmethod
    def classify_undertone(a: float, b: float) -> str:
        """Heuristic undertone classification from CIELAB a*/b*.

        Warm undertones skew strongly yellow (high b* relative to a*),
        cool undertones skew pink/red (a* comparatively higher, b* muted),
        and neutral sits between the two.
        """
        a_safe = a if abs(a) > 1e-6 else 1e-6
        ratio = b / a_safe
        if ratio >= 1.55:
            return "Warm"
        if ratio <= 1.05:
            return "Cool"
        return "Neutral"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self, image_bgr: np.ndarray, draw_debug: bool = True
    ) -> SkinAnalysisResult:
        """Run full skin analysis on a single BGR image (e.g. a webcam
        frame or an uploaded selfie).

        Raises NoFaceDetectedError if no face could be located.
        """
        h, w = image_bgr.shape[:2]
        landmarks = self._detect_landmarks(image_bgr)
        if landmarks is None:
            raise NoFaceDetectedError(
                "No face detected. Please retake the photo with your face "
                "clearly visible, well lit, and facing the camera."
            )

        pts_px = self._landmarks_to_px(landmarks, w, h)
        raw_pixels_bgr, cheek_points = self._cheek_pixels(image_bgr, pts_px)
        pixels_bgr = self._filter_outlier_pixels(raw_pixels_bgr)

        mean_bgr = pixels_bgr.mean(axis=0)
        mean_rgb = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))

        L, a, b = self.rgb_to_lab(mean_rgb)
        ita = self.compute_ita(L, b)
        skin_tone = self.classify_skin_tone(ita)
        undertone = self.classify_undertone(a, b)

        rgb_int = tuple(int(round(c)) for c in mean_rgb)
        hex_code = "#{:02x}{:02x}{:02x}".format(*rgb_int)

        xs, ys = pts_px[:, 0], pts_px[:, 1]
        face_bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min()),
            int(ys.max() - ys.min()),
        )

        annotated = None
        if draw_debug:
            annotated = image_bgr.copy()
            face_width = abs(pts_px[FACE_WIDTH_IDX[1]][0] - pts_px[FACE_WIDTH_IDX[0]][0])
            radius = max(int(face_width * 0.045), 4)
            for pt in cheek_points:
                cv2.circle(
                    annotated, (int(pt[0]), int(pt[1])), radius, (0, 255, 0), 2
                )
            x, y, bw, bh = face_bbox
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 200, 0), 1)

        return SkinAnalysisResult(
            rgb=rgb_int,
            hex=hex_code,
            lab=(round(L, 2), round(a, 2), round(b, 2)),
            ita=round(ita, 2),
            skin_tone=skin_tone,
            undertone=undertone,
            face_bbox=face_bbox,
            cheek_points_px=cheek_points,
            annotated_bgr=annotated,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python skin_analyzer.py <path_to_image>")
        raise SystemExit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit(f"Could not read image: {sys.argv[1]}")

    with SkinAnalyzer() as analyzer:
        result = analyzer.analyze(img)

    print(f"RGB       : {result.rgb}")
    print(f"Hex       : {result.hex}")
    print(f"Lab       : {result.lab}")
    print(f"ITA       : {result.ita}")
    print(f"Skin Tone : {result.skin_tone}")
    print(f"Undertone : {result.undertone}")
