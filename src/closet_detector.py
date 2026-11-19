"""
closet_detector.py
==================
Auto-detects and crops garments for the "My Closet" module using:
  1. YOLOv8-Pose for person-worn clothes (auto-separates Top and Bottom).
  2. Foreground saliency / contour detection for standalone/flat-lay garments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

_POSE_MODEL: Optional[YOLO] = None


def get_pose_model(model_path: str = "yolov8n-pose.pt") -> YOLO:
    """Lazy load and cache YOLOv8-pose model."""
    global _POSE_MODEL
    if _POSE_MODEL is None:
        _POSE_MODEL = YOLO(model_path)
    return _POSE_MODEL


@dataclass
class DetectedGarment:
    category: str  # 'shirt', 'jacket', 'pants', 'shorts', 'skirt', 'dress'
    label: str
    box: Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax
    crop_bgr: np.ndarray
    confidence: float
    detection_source: str  # 'yolo_pose' or 'saliency'


def detect_garments_from_image(
    image_bgr: np.ndarray, model_path: str = "yolov8n-pose.pt"
) -> List[DetectedGarment]:
    """Auto-detect garments in a camera frame or uploaded photo."""
    h, w = image_bgr.shape[:2]
    detected_items: List[DetectedGarment] = []

    # -----------------------------------------------------------------------
    # Strategy 1: YOLOv8-Pose (Person wearing or holding clothes)
    # -----------------------------------------------------------------------
    try:
        model = get_pose_model(model_path)
        results = model(image_bgr, verbose=False)
        if results and len(results[0]) > 0 and results[0].keypoints is not None:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confidences))
            person_conf = float(confidences[best_idx])

            if person_conf >= 0.35:
                kpts = results[0].keypoints.data[best_idx].cpu().numpy()  # (17, 3)

                # Keypoints:
                # 5: L_shoulder, 6: R_shoulder
                # 11: L_hip, 12: R_hip
                # 13: L_knee, 14: R_knee
                # 15: L_ankle, 16: R_ankle
                ls, rs = kpts[5], kpts[6]
                lh, rh = kpts[11], kpts[12]
                lk, rk = kpts[13], kpts[14]
                la, ra = kpts[15], kpts[16]

                # --- 1A. Top Garment (Shirt / Jacket) ---
                if ls[2] > 0.3 and rs[2] > 0.3:
                    top_y1 = int(min(ls[1], rs[1]))
                    if lh[2] > 0.25 and rh[2] > 0.25:
                        top_y2 = int(max(lh[1], rh[1]))
                    else:
                        shoulder_w = abs(ls[0] - rs[0])
                        top_y2 = min(h, int(top_y1 + shoulder_w * 1.3))

                    top_x1 = max(0, int(min(ls[0], rs[0]) - abs(ls[0] - rs[0]) * 0.15))
                    top_x2 = min(w, int(max(ls[0], rs[0]) + abs(ls[0] - rs[0]) * 0.15))

                    bw = max(1, top_x2 - top_x1)
                    bh = max(1, top_y2 - top_y1)
                    ix = int(bw * 0.12)
                    iy = int(bh * 0.10)

                    cx1, cy1 = max(0, top_x1 + ix), max(0, top_y1 + iy)
                    cx2, cy2 = min(w, top_x2 - ix), min(h, top_y2 - iy)

                    if cx2 > cx1 and cy2 > cy1:
                        crop = image_bgr[cy1:cy2, cx1:cx2]
                        detected_items.append(
                            DetectedGarment(
                                category="shirt",
                                label="Detected Top (Shirt)",
                                box=(top_x1, top_y1, top_x2, top_y2),
                                crop_bgr=crop,
                                confidence=float(min(ls[2], rs[2])),
                                detection_source="yolo_pose",
                            )
                        )

                # --- 1B. Bottom Garment (Pants / Skirt / Shorts) ---
                if lh[2] > 0.25 and rh[2] > 0.25:
                    bot_y1 = int(max(lh[1], rh[1]))
                    if la[2] > 0.25 and ra[2] > 0.25:
                        bot_y2 = int(max(la[1], ra[1]))
                        bot_x1 = int(min(lh[0], rh[0], lk[0], rk[0], la[0], ra[0]))
                        bot_x2 = int(max(lh[0], rh[0], lk[0], rk[0], la[0], ra[0]))
                        cat = "pants"
                    elif lk[2] > 0.25 and rk[2] > 0.25:
                        bot_y2 = int(max(lk[1], rk[1]))
                        bot_x1 = int(min(lh[0], rh[0], lk[0], rk[0]))
                        bot_x2 = int(max(lh[0], rh[0], lk[0], rk[0]))
                        cat = "shorts"
                    else:
                        est_len = int(abs(lh[1] - ls[1]) * 1.25) if ls[2] > 0.3 else int(h * 0.4)
                        bot_y2 = min(h, bot_y1 + est_len)
                        bot_x1 = int(min(lh[0], rh[0]) - 30)
                        bot_x2 = int(max(lh[0], rh[0]) + 30)
                        cat = "pants"

                    bot_x1 = max(0, bot_x1)
                    bot_x2 = min(w, bot_x2)

                    bw = max(1, bot_x2 - bot_x1)
                    bh = max(1, bot_y2 - bot_y1)
                    ix = int(bw * 0.15)
                    iy = int(bh * 0.08)

                    cx1, cy1 = max(0, bot_x1 + ix), max(0, bot_y1 + iy)
                    cx2, cy2 = min(w, bot_x2 - ix), min(h, bot_y2 - iy)

                    if cx2 > cx1 and cy2 > cy1:
                        crop = image_bgr[cy1:cy2, cx1:cx2]
                        detected_items.append(
                            DetectedGarment(
                                category=cat,
                                label=f"Detected Bottom ({cat.title()})",
                                box=(bot_x1, bot_y1, bot_x2, bot_y2),
                                crop_bgr=crop,
                                confidence=float(min(lh[2], rh[2])),
                                detection_source="yolo_pose",
                            )
                        )
    except Exception:
        pass

    # If YOLO-Pose detected garments, return them!
    if detected_items:
        return detected_items

    # -----------------------------------------------------------------------
    # Strategy 2: Standalone / Flat-lay Garment Saliency Detector
    # -----------------------------------------------------------------------
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = h * w
    best_cnt = None
    max_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0.06 * total_area < area < 0.95 * total_area:
            if area > max_area:
                max_area = area
                best_cnt = cnt

    if best_cnt is not None:
        x, y, bw, bh = cv2.boundingRect(best_cnt)
        # Margin inset 5%
        mx = int(bw * 0.05)
        my = int(bh * 0.05)
        cx1 = max(0, x + mx)
        cy1 = max(0, y + my)
        cx2 = min(w, x + bw - mx)
        cy2 = min(h, y + bh - my)

        aspect = bh / max(1.0, bw)
        cat = "pants" if aspect > 1.35 else "shirt"
        crop = image_bgr[cy1:cy2, cx1:cx2]

        return [
            DetectedGarment(
                category=cat,
                label=f"Detected {cat.title()}",
                box=(x, y, x + bw, y + bh),
                crop_bgr=crop,
                confidence=0.82,
                detection_source="saliency",
            )
        ]

    # -----------------------------------------------------------------------
    # Strategy 3: Safe Center Fallback (70% central crop)
    # -----------------------------------------------------------------------
    cx1 = int(w * 0.15)
    cy1 = int(h * 0.15)
    cx2 = int(w * 0.85)
    cy2 = int(h * 0.85)
    crop = image_bgr[cy1:cy2, cx1:cx2]
    return [
        DetectedGarment(
            category="shirt",
            label="Garment (Center View)",
            box=(cx1, cy1, cx2, cy2),
            crop_bgr=crop,
            confidence=0.50,
            detection_source="center_fallback",
        )
    ]


def annotate_detected_garments(
    image_bgr: np.ndarray, items: List[DetectedGarment]
) -> np.ndarray:
    """Draw bounding boxes and detection badges on the camera snapshot."""
    out = image_bgr.copy()
    colors = [(40, 180, 80), (180, 100, 40), (220, 140, 40), (60, 60, 220)]

    for i, item in enumerate(items):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = item.box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        label_txt = f"{item.label} ({item.category})"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        badge_y1 = max(0, y1 - th - 12)
        badge_y2 = y1
        badge_x2 = min(out.shape[1], x1 + tw + 16)

        cv2.rectangle(out, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            out,
            label_txt,
            (x1 + 8, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out
