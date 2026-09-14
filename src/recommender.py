"""
recommender.py
===============
Implements the CIE Delta-E 2000 (CIEDE2000) perceptual color-difference
formula and uses it to rank indexed garments (from
`data/processed/garment_catalog.csv`, produced by `garment_indexer.py`)
against a target color palette (from `palette_engine.py`).

Lower Delta-E 2000 == perceptually closer color match.
  * ~0-1   : imperceptible difference
  * ~1-2   : perceptible on close inspection
  * ~2-10  : perceptible at a glance
  * >10    : clearly different colors
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from src.closet_store import USER_GARMENTS_CSV, load_user_garments

DEFAULT_CATALOG_CSV = Path("data/processed/garment_catalog.csv")

Lab = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# CIEDE2000
# ---------------------------------------------------------------------------
def ciede2000(lab1: Lab, lab2: Lab) -> float:
    """Compute the CIEDE2000 color difference between two CIELAB colors.

    Reference: Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference
    Formula: Implementation Notes, Supplementary Test Data, and
    Mathematical Observations."
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    kl = kc = kh = 1.0

    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0

    c_bar7 = c_bar ** 7
    g = 0.5 * (1 - math.sqrt(c_bar7 / (c_bar7 + 25.0 ** 7)))

    a1p = (1 + g) * a1
    a2p = (1 + g) * a2

    c1p = math.hypot(a1p, b1)
    c2p = math.hypot(a2p, b2)

    def _hue_deg(a, b):
        if a == 0 and b == 0:
            return 0.0
        h = math.degrees(math.atan2(b, a))
        return h + 360.0 if h < 0 else h

    h1p = _hue_deg(a1p, b1)
    h2p = _hue_deg(a2p, b2)

    delta_L = L2 - L1
    delta_C = c2p - c1p

    if c1p * c2p == 0:
        delta_h = 0.0
    else:
        dh = h2p - h1p
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        delta_h = dh

    delta_H = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(delta_h) / 2.0)

    L_bar = (L1 + L2) / 2.0
    C_bar_p = (c1p + c2p) / 2.0

    if c1p * c2p == 0:
        h_bar_p = h1p + h2p
    else:
        h_sum = h1p + h2p
        if abs(h1p - h2p) > 180:
            h_bar_p = (h_sum + 360.0) / 2.0 if h_sum < 360 else (h_sum - 360.0) / 2.0
        else:
            h_bar_p = h_sum / 2.0

    T = (
        1
        - 0.17 * math.cos(math.radians(h_bar_p - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar_p))
        + 0.32 * math.cos(math.radians(3 * h_bar_p + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar_p - 63))
    )

    delta_theta = 30 * math.exp(-(((h_bar_p - 275) / 25.0) ** 2))
    c_bar_p7 = C_bar_p ** 7
    R_c = 2 * math.sqrt(c_bar_p7 / (c_bar_p7 + 25.0 ** 7))
    S_l = 1 + (0.015 * (L_bar - 50) ** 2) / math.sqrt(20 + (L_bar - 50) ** 2)
    S_c = 1 + 0.045 * C_bar_p
    S_h = 1 + 0.015 * C_bar_p * T
    R_t = -math.sin(math.radians(2 * delta_theta)) * R_c

    term_l = delta_L / (kl * S_l)
    term_c = delta_C / (kc * S_c)
    term_h = delta_H / (kh * S_h)

    delta_e = math.sqrt(term_l ** 2 + term_c ** 2 + term_h ** 2 + R_t * term_c * term_h)
    return float(delta_e)


# ---------------------------------------------------------------------------
# Matching engine
# ---------------------------------------------------------------------------
class Recommender:
    """Ranks catalog garments against a target palette using CIEDE2000."""

    def __init__(
        self,
        catalog_csv: Path | str = DEFAULT_CATALOG_CSV,
        include_user_garments: bool = True,
        user_garments_csv: Path | str = USER_GARMENTS_CSV,
    ) -> None:
        self.catalog_csv = Path(catalog_csv)
        self.user_garments_csv = Path(user_garments_csv)
        self.include_user_garments = include_user_garments
        if not self.catalog_csv.exists():
            raise FileNotFoundError(
                f"Garment catalog not found at {self.catalog_csv}. "
                "Run `python -m src.garment_indexer` first."
            )
        self.reload()

    def reload(self) -> None:
        """(Re)load the base catalog and merge in any user-contributed
        garments saved via `closet_store.add_user_garment`. Call this
        after a new item has been added to pick it up without restarting."""
        base = pd.read_csv(self.catalog_csv)
        if "source" not in base.columns:
            base["source"] = "dataset"
        if self.include_user_garments and self.user_garments_csv.exists():
            user_df = load_user_garments(self.user_garments_csv)
            self.catalog = pd.concat([base, user_df], ignore_index=True)
        else:
            self.catalog = base

    def categories(self) -> List[str]:
        return sorted(self.catalog["category"].dropna().unique().tolist())

    def rank(
        self,
        recommended_labs: Sequence[Lab],
        avoid_labs: Optional[Sequence[Lab]] = None,
        category: Optional[str] = None,
        top_n: int = 12,
        avoid_penalty_threshold: float = 15.0,
    ) -> pd.DataFrame:
        """Rank garments by minimum CIEDE2000 distance to any color in
        `recommended_labs`.

        Garments whose color is *too close* (Delta-E < avoid_penalty_threshold)
        to any color in `avoid_labs` are filtered out, unless doing so would
        leave zero results -- in that case the filter is relaxed.
        """
        df = self.catalog.copy()
        if category:
            df = df[df["category"] == category]
        if df.empty:
            return df.assign(match_distance=[], avoid_distance=[])

        avoid_labs = list(avoid_labs) if avoid_labs else []

        match_distances = []
        avoid_distances = []
        for _, row in df.iterrows():
            garment_lab = (row["lab_L"], row["lab_a"], row["lab_b"])
            rec_d = min(ciede2000(garment_lab, rl) for rl in recommended_labs)
            match_distances.append(rec_d)
            if avoid_labs:
                avoid_d = min(ciede2000(garment_lab, al) for al in avoid_labs)
            else:
                avoid_d = float("inf")
            avoid_distances.append(avoid_d)

        df = df.assign(match_distance=match_distances, avoid_distance=avoid_distances)
        df = df.sort_values("match_distance", ascending=True)

        if avoid_labs:
            safe = df[df["avoid_distance"] >= avoid_penalty_threshold]
            df = safe if not safe.empty else df

        return df.head(top_n).reset_index(drop=True)

    def rank_from_palette(
        self,
        palette: dict,
        category: Optional[str] = None,
        top_n: int = 12,
        avoid_penalty_threshold: float = 15.0,
    ) -> pd.DataFrame:
        """Convenience wrapper: takes the dict returned by
        `PaletteEngine.get_palette(...)` directly."""
        recommended_labs = [tuple(s["lab"]) for s in palette.get("recommended_swatches", [])]
        avoid_labs = [tuple(s["lab"]) for s in palette.get("avoid_swatches", [])]
        if not recommended_labs:
            raise ValueError("Palette has no recommended swatches to match against.")
        return self.rank(
            recommended_labs=recommended_labs,
            avoid_labs=avoid_labs,
            category=category,
            top_n=top_n,
            avoid_penalty_threshold=avoid_penalty_threshold,
        )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.palette_engine import PaletteEngine

    engine = PaletteEngine()
    palette = engine.get_palette("Fair", "Warm")

    rec = Recommender()
    results = rec.rank_from_palette(palette, top_n=10)
    cols = ["garment_id", "category", "dominant_hex", "match_distance", "avoid_distance"]
    print(results[cols].to_string(index=False))
