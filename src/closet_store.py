"""
closet_store.py
================
Lets users "save" an analyzed closet item into a persistent catalog so it
shows up in future `Recommender` results too -- without touching the
original `garment_catalog.csv` built from the Colorful Fashion dataset.

User-contributed items go into a separate CSV + crops folder
(`data/processed/user_garments.csv`, `data/processed/user_crops/`) with
the exact same schema as the main catalog (plus a `source` column), so
`recommender.Recommender` can simply concatenate the two.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

USER_GARMENTS_CSV = Path("data/processed/user_garments.csv")
USER_CROPS_DIR = Path("data/processed/user_crops")

CATALOG_COLUMNS = [
    "garment_id",
    "source_image",
    "category",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "image_width",
    "image_height",
    "dominant_r",
    "dominant_g",
    "dominant_b",
    "dominant_hex",
    "lab_L",
    "lab_a",
    "lab_b",
    "crop_path",
    "source_image_path",
    "source",
]


def _next_user_id(csv_path: Path = USER_GARMENTS_CSV) -> int:
    if not csv_path.exists():
        return 0
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return 0
        return int(df["garment_id"].str.replace("u", "", regex=False).astype(int).max()) + 1
    except Exception:
        return 0


def add_user_garment(
    crop_bgr: np.ndarray,
    category: str,
    dominant_rgb: Tuple[float, float, float],
    dominant_hex: str,
    lab: Tuple[float, float, float],
    original_filename: str = "user_upload.jpg",
    csv_path: Path = USER_GARMENTS_CSV,
    crops_dir: Path = USER_CROPS_DIR,
) -> str:
    """Persist one user-uploaded garment crop + its metadata.

    Returns the newly-assigned garment_id (e.g. "u000003").
    """
    csv_path = Path(csv_path)
    crops_dir = Path(crops_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    next_id = _next_user_id(csv_path)
    garment_id = f"u{next_id:06d}"

    crop_path = crops_dir / f"{garment_id}.jpg"
    cv2.imwrite(str(crop_path), crop_bgr)

    h, w = crop_bgr.shape[:2]
    row = {
        "garment_id": garment_id,
        "source_image": original_filename,
        "category": category,
        "xmin": 0,
        "ymin": 0,
        "xmax": w,
        "ymax": h,
        "image_width": w,
        "image_height": h,
        "dominant_r": round(dominant_rgb[0], 2),
        "dominant_g": round(dominant_rgb[1], 2),
        "dominant_b": round(dominant_rgb[2], 2),
        "dominant_hex": dominant_hex,
        "lab_L": round(lab[0], 3),
        "lab_a": round(lab[1], 3),
        "lab_b": round(lab[2], 3),
        "crop_path": str(crop_path),
        "source_image_path": "",
        "source": "user_upload",
    }

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row], columns=CATALOG_COLUMNS)
    updated.to_csv(csv_path, index=False)

    return garment_id


def load_user_garments(csv_path: Path = USER_GARMENTS_CSV) -> pd.DataFrame:
    """Load the user-contributed catalog, or an empty frame if none exist."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    return pd.read_csv(csv_path)


def clear_user_garments(
    csv_path: Path = USER_GARMENTS_CSV, crops_dir: Path = USER_CROPS_DIR
) -> None:
    """Remove all user-contributed items (used by a 'reset my closet
    additions' control, if exposed in the UI)."""
    csv_path = Path(csv_path)
    crops_dir = Path(crops_dir)
    if csv_path.exists():
        csv_path.unlink()
    if crops_dir.exists():
        for f in crops_dir.glob("*.jpg"):
            f.unlink()
