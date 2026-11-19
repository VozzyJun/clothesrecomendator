"""
ootd_analyzer.py
================
Real-time / Image OOTD (Outfit of the Day) Full-Body Analyzer using:
  1. YOLOv8-Pose: 17 keypoint human body pose & garment localization
  2. MediaPipe Face Mesh & ITA / CIELAB: Skin tone, undertone & seasonal palette
  3. Color Science (CIEDE2000, CIELAB K-Means): Personal color match & outfit harmony
  4. Body Silhouette & Proportion Geometry: Auto-detects torso length & body shape
  5. Styling Rulebase: Looks up fitting style, patterns, materials & do/don't exaggerate
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.closet_engine import (
    HarmonyResult,
    PersonalMatch,
    harmony_score,
    personal_match,
    verdict_label,
)
from src.color_utils import dominant_lab_color
from src.palette_engine import PaletteEngine
from src.skin_analyzer import SkinAnalysisResult, SkinAnalyzer

# Default paths
DEFAULT_RULES_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "fashion_and_color_recommendation.csv"
DEFAULT_CATALOG_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "garment_catalog.csv"


@dataclass
class GarmentRegion:
    label: str
    box: Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax
    dominant_rgb: Tuple[int, int, int]
    dominant_hex: str
    dominant_lab: Tuple[float, float, float]
    personal_match: Optional[PersonalMatch] = None
    crop_bgr: Optional[np.ndarray] = None


@dataclass
class BodyProportionResult:
    torso_length_label: str  # "Short torso", "Balanced", "Long torso"
    torso_ratio: float
    silhouette_shape: str    # "Hourglass", "Inverted Triangle", "Pear", "Rectangle"
    shoulder_hip_ratio: float
    styling_tips: Dict[str, str] = field(default_factory=dict)


@dataclass
class OOTDResult:
    person_detected: bool
    confidence: float = 0.0
    skin_tone: str = "Medium"
    undertone: str = "Warm"
    season: str = "Warm Autumn"
    face_box: Optional[Tuple[int, int, int, int]] = None
    top_garment: Optional[GarmentRegion] = None
    bottom_garment: Optional[GarmentRegion] = None
    outfit_harmony: Optional[HarmonyResult] = None
    body_proportions: Optional[BodyProportionResult] = None
    overall_score: float = 0.0
    feedback_notes: List[str] = field(default_factory=list)
    shopee_suggestions: List[Dict[str, str]] = field(default_factory=list)


class OOTDAnalyzer:
    """End-to-end full-body OOTD analyzer combining YOLOv8-pose with personal color analysis."""

    def __init__(
        self,
        yolo_model_path: str = "yolov8n-pose.pt",
        rules_csv: Path = DEFAULT_RULES_CSV,
        catalog_csv: Path = DEFAULT_CATALOG_CSV,
    ):
        self.yolo = YOLO(yolo_model_path)
        self.palette_engine = PaletteEngine(rules_csv)
        self.skin_analyzer = SkinAnalyzer(static_image_mode=True)
        self.catalog_csv = catalog_csv

    def analyze(
        self,
        image_bgr: np.ndarray,
        user_skin_tone: Optional[str] = None,
        user_undertone: Optional[str] = None,
        target_audience: str = "Semua",
    ) -> OOTDResult:
        """Run full OOTD analysis on a single BGR image or video frame."""
        h, w = image_bgr.shape[:2]

        # 1. Run YOLOv8-pose inference
        results = self.yolo(image_bgr, verbose=False)
        if not results or len(results[0]) == 0 or results[0].keypoints is None:
            return OOTDResult(person_detected=False)

        # Pick the person with the highest detection confidence
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confidences))
        person_conf = float(confidences[best_idx])

        # Extract 17 keypoints: [x, y, conf]
        kpts = results[0].keypoints.data[best_idx].cpu().numpy()  # shape (17, 3)

        # 2. Face & Skin Tone Pipeline
        skin_tone = user_skin_tone
        undertone = user_undertone
        face_box = None

        # Determine head box from nose(0), eyes(1,2), ears(3,4)
        head_indices = [0, 1, 2, 3, 4]
        head_kpts = kpts[head_indices]
        valid_head = head_kpts[head_kpts[:, 2] > 0.35]

        if len(valid_head) >= 2:
            hx1 = max(0, int(np.min(valid_head[:, 0]) - 35))
            hy1 = max(0, int(np.min(valid_head[:, 1]) - 45))
            hx2 = min(w, int(np.max(valid_head[:, 0]) + 35))
            hy2 = min(h, int(np.max(valid_head[:, 1]) + 40))
            face_box = (hx1, hy1, hx2, hy2)

            if skin_tone is None or undertone is None:
                # Try MediaPipe Face Mesh on cropped head (higher resolution for small faces)
                head_crop = image_bgr[hy1:hy2, hx1:hx2]
                if head_crop.size > 0:
                    try:
                        # Resize crop to standard selfie size for MediaPipe
                        head_scaled = cv2.resize(head_crop, (280, 280), interpolation=cv2.INTER_LINEAR)
                        mesh_res = self.skin_analyzer.analyze(head_scaled, apply_awb=True)
                        if mesh_res.face_detected:
                            skin_tone = mesh_res.skin_tone
                            undertone = mesh_res.undertone
                    except Exception:
                        pass

        # Fallback defaults if face mesh is obstructed
        skin_tone = skin_tone or "Medium"
        undertone = undertone or "Warm"

        # 3. Retrieve Seasonal Palette from PaletteEngine
        palette = self.palette_engine.get_palette(skin_tone, undertone)
        season = palette.get("season", f"{undertone} {skin_tone}")

        # 4. Upper Body (Top Garment) Localization & Color Extraction
        # Shoulders: 5 (left), 6 (right); Hips: 11 (left), 12 (right)
        ls, rs = kpts[5], kpts[6]
        lh, rh = kpts[11], kpts[12]

        top_garment = None
        if ls[2] > 0.35 and rs[2] > 0.35 and lh[2] > 0.25 and rh[2] > 0.25:
            # Garment box from shoulders to hips
            top_y1 = int(min(ls[1], rs[1]))
            top_y2 = int(max(lh[1], rh[1]))
            top_x1 = int(min(ls[0], rs[0], lh[0], rh[0]))
            top_x2 = int(max(ls[0], rs[0], lh[0], rh[0]))

            # Inset box by 15% to avoid skin, hair, and background leakage
            box_w = max(1, top_x2 - top_x1)
            box_h = max(1, top_y2 - top_y1)
            inset_x = int(box_w * 0.15)
            inset_y = int(box_h * 0.12)

            crop_x1 = max(0, top_x1 + inset_x)
            crop_y1 = max(0, top_y1 + inset_y)
            crop_x2 = min(w, top_x2 - inset_x)
            crop_y2 = min(h, top_y2 - inset_y)

            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                top_crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                top_rgb = cv2.cvtColor(top_crop, cv2.COLOR_BGR2RGB)
                dom_info = dominant_lab_color(top_rgb)
                dom_rgb = tuple(int(round(c)) for c in dom_info["rgb"])
                dom_lab = dom_info["lab"]
                dom_hex = dom_info["hex"]
                t_match = personal_match(dom_lab, palette)

                top_garment = GarmentRegion(
                    label="Top",
                    box=(top_x1, top_y1, top_x2, top_y2),
                    dominant_rgb=dom_rgb,
                    dominant_hex=dom_hex,
                    dominant_lab=dom_lab,
                    personal_match=t_match,
                    crop_bgr=top_crop,
                )

        # 5. Lower Body (Bottom Garment) Localization & Color Extraction
        # Hips: 11, 12; Knees: 13, 14; Ankles: 15, 16
        lk, rk = kpts[13], kpts[14]
        la, ra = kpts[15], kpts[16]

        bottom_garment = None
        if lh[2] > 0.3 and rh[2] > 0.3:
            bot_y1 = int(max(lh[1], rh[1]))
            # Use ankles if visible, else knees, else estimate
            if la[2] > 0.25 and ra[2] > 0.25:
                bot_y2 = int(max(la[1], ra[1]))
                bot_x1 = int(min(lh[0], rh[0], lk[0], rk[0], la[0], ra[0]))
                bot_x2 = int(max(lh[0], rh[0], lk[0], rk[0], la[0], ra[0]))
            elif lk[2] > 0.25 and rk[2] > 0.25:
                bot_y2 = int(max(lk[1], rk[1]))
                bot_x1 = int(min(lh[0], rh[0], lk[0], rk[0]))
                bot_x2 = int(max(lh[0], rh[0], lk[0], rk[0]))
            else:
                # Estimate leg extent if only hips are visible
                est_leg_len = int(abs(lh[1] - ls[1]) * 1.25)
                bot_y2 = min(h, bot_y1 + est_leg_len)
                bot_x1 = int(min(lh[0], rh[0]))
                bot_x2 = int(max(lh[0], rh[0]))

            bot_w = max(1, bot_x2 - bot_x1)
            bot_h = max(1, bot_y2 - bot_y1)
            inset_x = int(bot_w * 0.18)
            inset_y = int(bot_h * 0.10)

            crop_x1 = max(0, bot_x1 + inset_x)
            crop_y1 = max(0, bot_y1 + inset_y)
            crop_x2 = min(w, bot_x2 - inset_x)
            crop_y2 = min(h, bot_y2 - inset_y)

            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                bot_crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                bot_rgb = cv2.cvtColor(bot_crop, cv2.COLOR_BGR2RGB)
                dom_info = dominant_lab_color(bot_rgb)
                dom_rgb = tuple(int(round(c)) for c in dom_info["rgb"])
                dom_lab = dom_info["lab"]
                dom_hex = dom_info["hex"]
                b_match = personal_match(dom_lab, palette)

                bottom_garment = GarmentRegion(
                    label="Bottom",
                    box=(bot_x1, bot_y1, bot_x2, bot_y2),
                    dominant_rgb=dom_rgb,
                    dominant_hex=dom_hex,
                    dominant_lab=dom_lab,
                    personal_match=b_match,
                    crop_bgr=bot_crop,
                )

        # 6. Outfit Harmony Scoring (Top + Bottom Pair)
        outfit_harmony = None
        if top_garment and bottom_garment:
            outfit_harmony = harmony_score(top_garment.dominant_lab, bottom_garment.dominant_lab)

        # 7. Body Silhouette & Proportions via Keypoint Geometry
        body_proportions = self._estimate_body_proportions(kpts, skin_tone, undertone)

        # 8. Overall OOTD Score Calculation
        overall_score, notes = self._calculate_overall_score(
            top_garment, bottom_garment, outfit_harmony, season
        )

        # 9. Smart Shopee Suggestions for Outfit Improvement
        shopee_suggestions = self._build_shopee_suggestions(
            top_garment, bottom_garment, palette, target_audience
        )

        return OOTDResult(
            person_detected=True,
            confidence=person_conf,
            skin_tone=skin_tone,
            undertone=undertone,
            season=season,
            face_box=face_box,
            top_garment=top_garment,
            bottom_garment=bottom_garment,
            outfit_harmony=outfit_harmony,
            body_proportions=body_proportions,
            overall_score=overall_score,
            feedback_notes=notes,
            shopee_suggestions=shopee_suggestions,
        )

    def _estimate_body_proportions(
        self, kpts: np.ndarray, skin_tone: str, undertone: str
    ) -> BodyProportionResult:
        """Calculate torso-to-leg ratio and shoulder-to-hip ratio from pose keypoints."""
        ls, rs = kpts[5], kpts[6]
        lh, rh = kpts[11], kpts[12]
        la, ra = kpts[15], kpts[16]

        shoulder_w = float(np.linalg.norm(ls[:2] - rs[:2]))
        hip_w = float(np.linalg.norm(lh[:2] - rh[:2]))
        shoulder_hip_ratio = (shoulder_w / max(1.0, hip_w))

        if shoulder_hip_ratio > 1.18:
            shape = "Inverted Triangle"
        elif shoulder_hip_ratio < 0.88:
            shape = "Triangle"
        elif abs(shoulder_w - hip_w) / max(1.0, shoulder_w) < 0.12:
            shape = "Hourglass"
        else:
            shape = "Rectangle"

        # Torso vs leg height
        mid_shoulder_y = (ls[1] + rs[1]) / 2.0
        mid_hip_y = (lh[1] + rh[1]) / 2.0
        torso_len = abs(mid_hip_y - mid_shoulder_y)

        if la[2] > 0.2 and ra[2] > 0.2:
            mid_ankle_y = (la[1] + ra[1]) / 2.0
            leg_len = abs(mid_ankle_y - mid_hip_y)
        else:
            leg_len = torso_len * 1.35  # anthropometric baseline estimate

        torso_ratio = float(torso_len / max(1.0, leg_len))
        if torso_ratio < 0.65:
            torso_label = "Short Torso"
        elif torso_ratio > 0.85:
            torso_label = "Long Torso"
        else:
            torso_label = "Balanced"

        # Query styling tips from PaletteEngine rules
        try:
            full_palette = self.palette_engine.get_palette(
                skin_tone=skin_tone,
                undertone=undertone,
                body_proportion=shape,
                torso_length=torso_label,
            )
            rules = full_palette.get("style", {})
        except Exception:
            rules = {}

        return BodyProportionResult(
            torso_length_label=torso_label,
            torso_ratio=round(torso_ratio, 2),
            silhouette_shape=shape,
            shoulder_hip_ratio=round(shoulder_hip_ratio, 2),
            styling_tips=rules,
        )

    def _calculate_overall_score(
        self,
        top: Optional[GarmentRegion],
        bottom: Optional[GarmentRegion],
        harmony: Optional[HarmonyResult],
        season: str,
    ) -> Tuple[float, List[str]]:
        """Synthesize individual matches and pair harmony into a 0-100 score."""
        notes = []
        scores = []
        weights = []

        if top and top.personal_match:
            top_score = top.personal_match.pct
            scores.append(top_score)
            weights.append(0.35)
            if top.personal_match.flagged_avoid:
                notes.append(f"Atasan ({top.dominant_hex}) mendekati palet yang perlu dihindari.")
            elif top_score >= 80:
                notes.append(f"Atasan ({top.dominant_hex}) sangat selaras dengan musim {season}!")
            else:
                notes.append(f"Atasan ({top.dominant_hex}) berpadu cukup baik ({top.personal_match.verdict}).")

        if bottom and bottom.personal_match:
            bot_score = bottom.personal_match.pct
            scores.append(bot_score)
            weights.append(0.30)
            if bottom.personal_match.flagged_avoid:
                notes.append(f"Bawahan ({bottom.dominant_hex}) bertolak belakang dengan palet kulitmu.")
            elif bot_score >= 80:
                notes.append(f"Bawahan ({bottom.dominant_hex}) sangat serasi dengan undertonemu.")

        if harmony:
            scores.append(harmony.pct)
            weights.append(0.35)
            notes.append(f"Harmoni perpaduan atasan & bawahan: {harmony.relationship} ({harmony.pct:.0f}%).")

        if not scores:
            return 50.0, ["Pakaian belum terdeteksi secara optimal."]

        total_weight = sum(weights)
        overall = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return round(overall, 1), notes

    def _build_shopee_suggestions(
        self,
        top: Optional[GarmentRegion],
        bottom: Optional[GarmentRegion],
        palette: dict,
        target_audience: str,
    ) -> List[Dict[str, str]]:
        """Recommend complementary items on Shopee when an item has low match."""
        suggestions = []
        rec_swatches = palette.get("recommended_swatches", [])
        if not rec_swatches:
            return suggestions

        audience_prefix = f"{target_audience} " if target_audience != "Semua" else ""

        # If top has lower score or needs complement
        if top and top.personal_match and top.personal_match.pct < 75:
            best_color = rec_swatches[0]["name"]
            url = f"https://shopee.co.id/search?keyword={audience_prefix}baju+{best_color}".replace(" ", "%20")
            suggestions.append({
                "type": "Top Upgrade",
                "reason": f"Ganti atasan dengan warna {best_color} agar seirama dengan palet {palette.get('season')}",
                "url": url,
            })

        # Complementary bottom
        if bottom and bottom.personal_match and bottom.personal_match.pct < 75:
            alt_color = rec_swatches[min(2, len(rec_swatches) - 1)]["name"]
            url = f"https://shopee.co.id/search?keyword={audience_prefix}celana+{alt_color}".replace(" ", "%20")
            suggestions.append({
                "type": "Bottom Upgrade",
                "reason": f"Pilihan bawahan warna {alt_color} memberikan kontras seimbang",
                "url": url,
            })

        return suggestions

    def annotate(self, image_bgr: np.ndarray, result: OOTDResult) -> np.ndarray:
        """Draw an informative, clean HUD overlay on the BGR image."""
        if not result.person_detected:
            annotated = image_bgr.copy()
            cv2.putText(
                annotated,
                "No person detected in frame",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 220),
                2,
                cv2.LINE_AA,
            )
            return annotated

        out = image_bgr.copy()
        h, w = out.shape[:2]

        # Draw Face Box
        if result.face_box:
            fx1, fy1, fx2, fy2 = result.face_box
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (230, 200, 160), 2)
            cv2.putText(
                out,
                f"Face: {result.season}",
                (fx1, max(20, fy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw Top Garment Box & Color Swatch
        if result.top_garment:
            tx1, ty1, tx2, ty2 = result.top_garment.box
            color_bgr = tuple(int(c) for c in result.top_garment.dominant_rgb[::-1])
            cv2.rectangle(out, (tx1, ty1), (tx2, ty2), (180, 100, 40), 2)
            # Color swatch square
            cv2.rectangle(out, (tx1 + 5, ty1 + 5), (tx1 + 35, ty1 + 35), color_bgr, -1)
            cv2.rectangle(out, (tx1 + 5, ty1 + 5), (tx1 + 35, ty1 + 35), (255, 255, 255), 1)

            verdict = result.top_garment.personal_match.verdict if result.top_garment.personal_match else ""
            pct = result.top_garment.personal_match.pct if result.top_garment.personal_match else 0
            cv2.putText(
                out,
                f"Top: {result.top_garment.dominant_hex} ({verdict} {pct:.0f}%)",
                (tx1 + 42, ty1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw Bottom Garment Box & Color Swatch
        if result.bottom_garment:
            bx1, by1, bx2, by2 = result.bottom_garment.box
            color_bgr = tuple(int(c) for c in result.bottom_garment.dominant_rgb[::-1])
            cv2.rectangle(out, (bx1, by1), (bx2, by2), (40, 160, 80), 2)
            cv2.rectangle(out, (bx1 + 5, by1 + 5), (bx1 + 35, by1 + 35), color_bgr, -1)
            cv2.rectangle(out, (bx1 + 5, by1 + 5), (bx1 + 35, by1 + 35), (255, 255, 255), 1)

            verdict = result.bottom_garment.personal_match.verdict if result.bottom_garment.personal_match else ""
            pct = result.bottom_garment.personal_match.pct if result.bottom_garment.personal_match else 0
            cv2.putText(
                out,
                f"Bottom: {result.bottom_garment.dominant_hex} ({verdict} {pct:.0f}%)",
                (bx1 + 42, by1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Top HUD Banner: Overall OOTD Score
        hud_h = 75
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (35, 28, 22), -1)
        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

        score_color = (60, 220, 100) if result.overall_score >= 75 else ((60, 180, 240) if result.overall_score >= 50 else (80, 80, 240))
        cv2.putText(
            out,
            f"OOTD SCORE: {result.overall_score:.0f}/100",
            (20, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            score_color,
            2,
            cv2.LINE_AA,
        )

        harmony_rel = result.outfit_harmony.relationship if result.outfit_harmony else "N/A"
        body_label = f"{result.body_proportions.silhouette_shape} | {result.body_proportions.torso_length_label}" if result.body_proportions else ""
        cv2.putText(
            out,
            f"Harmony: {harmony_rel}  |  Body: {body_label}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        return out
