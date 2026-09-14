"""
palette_engine.py
==================
Loads `data/raw/fashion_and_color_recommendation.csv` (Skin Tone / Under
Tone -> Recommended & Avoid Clothing Colors, styling heuristics) and turns
the free-text color phrases ("Earth Tones", "Jewel Tones", "Warm Red", ...)
into concrete, displayable swatches with Hex + CIELAB coordinates.

This module is the bridge between:
  * the rules dataset (human-readable style guidance), and
  * the matching engine (`recommender.py`), which needs numeric Lab colors
    to run CIEDE2000 comparisons against indexed garment colors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.skin_analyzer import SkinAnalyzer

DEFAULT_CSV_PATH = Path("data/raw/fashion_and_color_recommendation.csv")


# ---------------------------------------------------------------------------
# Color phrase -> representative swatch(es)
# ---------------------------------------------------------------------------
# Some phrases in the dataset describe a single color ("Mustard"), others
# describe a *family* of colors ("Earth Tones", "Jewel Tones"). Families map
# to several representative hex swatches so the palette UI and the matching
# engine both have concrete colors to work with.
COLOR_PHRASE_TO_HEXES: Dict[str, List[str]] = {
    # --- Recommended-side single colors ---
    "coral": ["#FF7F6E"],
    "emerald": ["#0F8B5A"],
    "icy blue": ["#AFDFE8"],
    "lavender": ["#B497D6"],
    "mustard": ["#D8A928"],
    "neutral beige": ["#D8C3A5"],
    "olive": ["#6E7A3B"],
    "peach": ["#FFC7A6"],
    "plums": ["#7B3F63"],
    "silver": ["#C4C6C8"],
    "soft pinks": ["#F3C6CE"],
    "teal": ["#217A79"],
    "warm red": ["#C1392B"],
    # --- Families (multiple representative swatches) ---
    "earth tones": ["#8A6A4B", "#B08D57", "#6B4423", "#A9743B"],
    "jewel tones": ["#0B6E4F", "#7B1E3C", "#1F3A93", "#4B2E83"],
    # --- Avoid-side colors (still need swatches to render "avoid" chips) ---
    "brown": ["#6B4423"],
    "cool blue": ["#3E6BA8"],
    "fluorescents": ["#CCFF00"],
    "harsh yellow": ["#FFE800"],
    "icy gray": ["#C7CDD1"],
    "orange": ["#F26B21"],
}


@dataclass
class ColorSwatch:
    name: str
    hex: str
    lab: tuple

    def to_dict(self) -> dict:
        return {"name": self.name, "hex": self.hex, "lab": self.lab}


@staticmethod
def _hex_to_rgb(hex_code: str):
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


def _swatches_for_phrase(phrase: str) -> List[ColorSwatch]:
    key = phrase.strip().lower()
    hexes = COLOR_PHRASE_TO_HEXES.get(key)
    if not hexes:
        # Unknown phrase: skip silently gracefully by returning empty list.
        return []
    swatches = []
    for h in hexes:
        rgb = _hex_to_rgb(h)
        lab = SkinAnalyzer.rgb_to_lab(rgb)
        swatches.append(ColorSwatch(name=phrase.strip(), hex=h, lab=lab))
    return swatches


class PaletteEngine:
    """Loads and queries the color/style recommendation rules dataset."""

    def __init__(self, csv_path: Path | str = DEFAULT_CSV_PATH) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self._normalize_columns()

    def _normalize_columns(self) -> None:
        self.df.columns = [c.strip() for c in self.df.columns]

    # ------------------------------------------------------------------
    def available_skin_tones(self) -> List[str]:
        return sorted(self.df["Skin Tone"].dropna().unique().tolist())

    def available_undertones(self) -> List[str]:
        return sorted(self.df["Under Tone"].dropna().unique().tolist())

    def available_body_proportions(self) -> List[str]:
        return sorted(self.df["Body Proportion"].dropna().unique().tolist())

    def available_torso_lengths(self) -> List[str]:
        return sorted(self.df["Torso length"].dropna().unique().tolist())

    # ------------------------------------------------------------------
    @staticmethod
    def _split_phrases(cell: str) -> List[str]:
        if not isinstance(cell, str):
            return []
        return [p.strip() for p in cell.split(",") if p.strip()]

    def get_rules(
        self,
        skin_tone: str,
        undertone: str,
        body_proportion: str | None = None,
        torso_length: str | None = None,
    ) -> pd.DataFrame:
        """Filter the raw rules table for the given attributes."""
        mask = (self.df["Skin Tone"] == skin_tone) & (self.df["Under Tone"] == undertone)
        if body_proportion:
            mask &= self.df["Body Proportion"] == body_proportion
        if torso_length:
            mask &= self.df["Torso length"] == torso_length
        return self.df[mask]

    def get_palette(
        self,
        skin_tone: str,
        undertone: str,
        body_proportion: str | None = None,
        torso_length: str | None = None,
    ) -> dict:
        """Build a full palette + styling recommendation payload for the
        given skin tone / undertone (and optional body-shape filters).
        """
        rows = self.get_rules(skin_tone, undertone, body_proportion, torso_length)
        if rows.empty:
            # Fall back to skin tone + undertone only, ignoring shape filters.
            rows = self.get_rules(skin_tone, undertone)
        if rows.empty:
            raise ValueError(
                f"No rules found for Skin Tone={skin_tone!r}, Under Tone={undertone!r}"
            )

        rec_phrases: List[str] = []
        avoid_phrases: List[str] = []
        for cell in rows["Recommended Clothing Colors"].unique():
            rec_phrases.extend(self._split_phrases(cell))
        for cell in rows["Avoid Clothing Colors"].unique():
            avoid_phrases.extend(self._split_phrases(cell))
        rec_phrases = list(dict.fromkeys(rec_phrases))  # de-dupe, keep order
        avoid_phrases = list(dict.fromkeys(avoid_phrases))

        rec_swatches: List[ColorSwatch] = []
        for phrase in rec_phrases:
            rec_swatches.extend(_swatches_for_phrase(phrase))
        avoid_swatches: List[ColorSwatch] = []
        for phrase in avoid_phrases:
            avoid_swatches.extend(_swatches_for_phrase(phrase))

        def _unique_col(col: str) -> List[str]:
            return sorted({v for v in rows[col].dropna().unique().tolist()})

        style = {
            "fitting_style": _unique_col("Recommended Fitting Style"),
            "materials": _unique_col("Recommended Materials"),
            "patterns": _unique_col("Recommended Patterns"),
            "jewelry_metal": _unique_col("Recommended Jewelry Metal"),
            "shoes": _unique_col("Recommended Shoes"),
            "fabric_nature": _unique_col("Fabric Nature"),
            "dont_exaggerate": _unique_col("Don't Exaggerate"),
            "do_exaggerate": _unique_col("Do Exaggerate"),
        }

        return {
            "skin_tone": skin_tone,
            "undertone": undertone,
            "recommended_phrases": rec_phrases,
            "avoid_phrases": avoid_phrases,
            "recommended_swatches": [s.to_dict() for s in rec_swatches],
            "avoid_swatches": [s.to_dict() for s in avoid_swatches],
            "style": style,
        }


if __name__ == "__main__":
    engine = PaletteEngine()
    print("Skin tones :", engine.available_skin_tones())
    print("Undertones :", engine.available_undertones())
    palette = engine.get_palette("Fair", "Warm")
    import json

    print(json.dumps(palette, indent=2))
