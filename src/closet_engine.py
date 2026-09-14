"""
closet_engine.py
=================
Powers the "My Closet" feature: scoring your own uploaded clothing items
against your personal color palette, scoring how well two (or more) items
work together as an outfit, and generating/ranking mix-and-match combos
across everything you've uploaded.

Two distinct questions are answered here, and they use different math:

  1. "Does this ONE item suit me?"
     -> CIEDE2000 distance from the garment's dominant color to your
        recommended / avoid palette (same engine as `recommender.py`,
        just applied to a single ad-hoc color instead of the indexed
        catalog).

  2. "Do THESE items work together as an outfit?"
     -> classic color-theory harmony: hue relationship (monochrome /
        analogous / complementary / "muddy middle") + whether either
        piece is a neutral (grays/beige/navy/black/white pair with
        almost everything).

Both scores are heuristic by design (there is no single "correct" answer
in fashion color theory) and are surfaced as 0-100% "match" scores with
a short verdict label, in keeping with the rest of the app's honesty
about the heuristic nature of automated color analysis.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.color_utils import dominant_lab_color, hue_chroma
from src.recommender import ciede2000

Lab = Tuple[float, float, float]

# Which "slot" each garment category occupies when assembling outfits.
CATEGORY_SLOT: Dict[str, str] = {
    "shirt": "top",
    "jacket": "outer",
    "pants": "bottom",
    "shorts": "bottom",
    "skirt": "bottom",
    "dress": "dress",
}

NEUTRAL_CHROMA_THRESHOLD = 12.0  # below this, a color is treated as "neutral"


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
def verdict_label(pct: float) -> str:
    if pct >= 75:
        return "Great match"
    if pct >= 55:
        return "Good match"
    if pct >= 35:
        return "Wear with caution"
    return "Better to avoid"


def _distance_to_percent(d: float, d_max: float = 25.0) -> float:
    """Map a CIEDE2000 distance to a 0-100% closeness score.
    0 distance -> 100%; >= d_max distance -> 0%."""
    pct = 100.0 * (1.0 - min(d, d_max) / d_max)
    return max(0.0, min(100.0, pct))


# ---------------------------------------------------------------------------
# 1. Personal match: one garment color vs. your palette
# ---------------------------------------------------------------------------
@dataclass
class PersonalMatch:
    pct: float
    verdict: str
    closest_recommended: Optional[dict]
    recommended_distance: Optional[float]
    avoid_distance: Optional[float]
    flagged_avoid: bool


def personal_match(garment_lab: Lab, palette: dict) -> PersonalMatch:
    """Score a single garment color against a palette dict produced by
    `PaletteEngine.get_palette(...)`."""
    rec_swatches = palette.get("recommended_swatches", [])
    avoid_swatches = palette.get("avoid_swatches", [])

    if not rec_swatches:
        return PersonalMatch(0.0, "Unknown", None, None, None, False)

    rec_dists = [(s, ciede2000(garment_lab, tuple(s["lab"]))) for s in rec_swatches]
    closest_swatch, rec_d = min(rec_dists, key=lambda t: t[1])

    avoid_d = None
    flagged = False
    base_pct = _distance_to_percent(rec_d, d_max=25.0)
    if avoid_swatches:
        avoid_d = min(ciede2000(garment_lab, tuple(s["lab"])) for s in avoid_swatches)
        if avoid_d < 12.0:
            base_pct *= 0.5
            flagged = True
        elif avoid_d < 20.0:
            base_pct *= 0.8

    pct = round(base_pct, 1)
    return PersonalMatch(
        pct=pct,
        verdict=verdict_label(pct),
        closest_recommended=closest_swatch,
        recommended_distance=round(rec_d, 2),
        avoid_distance=round(avoid_d, 2) if avoid_d is not None else None,
        flagged_avoid=flagged,
    )


# ---------------------------------------------------------------------------
# 2. Outfit harmony: garment color vs. garment color
# ---------------------------------------------------------------------------
@dataclass
class HarmonyResult:
    pct: float
    relationship: str


def harmony_score(lab1: Lab, lab2: Lab) -> HarmonyResult:
    """Score how well two garment colors work together using hue
    relationships + neutral detection."""
    hue1, chroma1 = hue_chroma(lab1)
    hue2, chroma2 = hue_chroma(lab2)

    is_neutral_1 = chroma1 < NEUTRAL_CHROMA_THRESHOLD
    is_neutral_2 = chroma2 < NEUTRAL_CHROMA_THRESHOLD

    if is_neutral_1 or is_neutral_2:
        # Neutrals (black/white/gray/beige/navy) pair with almost anything.
        pct = 90.0 if not (is_neutral_1 and is_neutral_2) else 82.0
        return HarmonyResult(pct=pct, relationship="Neutral pairing")

    diff = abs(hue1 - hue2)
    diff = min(diff, 360.0 - diff)  # 0-180

    if diff <= 15:
        base, label = 88.0, "Monochrome"
    elif diff <= 50:
        base, label = 92.0, "Analogous"
    elif diff <= 80:
        base, label = 58.0, "Soft clash"
    elif diff <= 100:
        base, label = 40.0, "Clashing"
    elif diff <= 160:
        base, label = 65.0, "Split-complementary"
    else:
        base, label = 78.0, "Complementary"

    # Two highly saturated colors in the clash zone fight harder for
    # attention; muted colors in the same zone read as more intentional.
    if 50 < diff <= 100:
        if chroma1 > 35 and chroma2 > 35:
            base -= 15
        elif min(chroma1, chroma2) < 20:
            base += 10

    pct = max(0.0, min(100.0, base))
    return HarmonyResult(pct=round(pct, 1), relationship=label)


# ---------------------------------------------------------------------------
# 3. Combos across a closet
# ---------------------------------------------------------------------------
@dataclass
class ClosetItem:
    item_id: str
    category: str
    label: str
    lab: Lab
    hex: str
    personal: PersonalMatch
    rgb: Tuple[float, float, float] = (128.0, 128.0, 128.0)
    thumbnail: Optional[str] = None  # path or None


@dataclass
class OutfitCombo:
    items: List[ClosetItem]
    personal_pct: float
    harmony_pct: float
    overall_pct: float
    verdict: str
    harmony_details: List[Tuple[str, str, str]] = field(default_factory=list)
    # harmony_details: list of (item_a_label, item_b_label, relationship)


def _combo_score(items: Sequence[ClosetItem]) -> OutfitCombo:
    personal_avg = sum(i.personal.pct for i in items) / len(items)

    pairs = list(itertools.combinations(items, 2))
    if pairs:
        harmonies = [harmony_score(a.lab, b.lab) for a, b in pairs]
        harmony_avg = sum(h.pct for h in harmonies) / len(harmonies)
        details = [
            (a.label, b.label, h.relationship)
            for (a, b), h in zip(pairs, harmonies)
        ]
    else:
        harmony_avg = 100.0  # a single dress worn alone has nothing to clash with
        details = []

    overall = round(0.5 * personal_avg + 0.5 * harmony_avg, 1)
    return OutfitCombo(
        items=list(items),
        personal_pct=round(personal_avg, 1),
        harmony_pct=round(harmony_avg, 1),
        overall_pct=overall,
        verdict=verdict_label(overall),
        harmony_details=details,
    )


def analyze_uploaded_item(
    crop_bgr: np.ndarray,
    item_id: str,
    label: str,
    category: str,
    palette: dict,
) -> ClosetItem:
    """Full pipeline for one uploaded-and-cropped closet photo: extract its
    dominant Lab color and score it against the user's palette, returning
    a ready-to-use `ClosetItem`."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    dom = dominant_lab_color(crop_rgb)
    pm = personal_match(tuple(dom["lab"]), palette)
    return ClosetItem(
        item_id=item_id,
        category=category,
        label=label,
        lab=tuple(dom["lab"]),
        hex=dom["hex"],
        personal=pm,
        rgb=tuple(dom["rgb"]),
    )


def generate_combos(
    closet: Sequence[ClosetItem], top_n: int = 8, max_combos: int = 500
) -> List[OutfitCombo]:
    """Generate valid outfit combos from a set of closet items and rank
    them by overall score, highest first.

    Valid slot patterns: dress (alone), dress+outer, top+bottom,
    top+bottom+outer.
    """
    by_slot: Dict[str, List[ClosetItem]] = {"top": [], "bottom": [], "outer": [], "dress": []}
    for item in closet:
        slot = CATEGORY_SLOT.get(item.category)
        if slot:
            by_slot[slot].append(item)

    combos: List[Tuple[ClosetItem, ...]] = []

    for top, bottom in itertools.product(by_slot["top"], by_slot["bottom"]):
        combos.append((top, bottom))
    for top, bottom, outer in itertools.product(
        by_slot["top"], by_slot["bottom"], by_slot["outer"]
    ):
        combos.append((top, bottom, outer))
    for dress in by_slot["dress"]:
        combos.append((dress,))
    for dress, outer in itertools.product(by_slot["dress"], by_slot["outer"]):
        combos.append((dress, outer))

    combos = combos[:max_combos]
    scored = [_combo_score(c) for c in combos]
    scored.sort(key=lambda c: c.overall_pct, reverse=True)
    return scored[:top_n]
