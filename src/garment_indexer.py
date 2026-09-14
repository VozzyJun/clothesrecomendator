"""
garment_indexer.py
===================
Walks the "Colorful Fashion" object-detection dataset
(`data/raw/colorful_fashion/{JPEGImages,Annotations}`), which stores one
PASCAL-VOC XML per image with bounding boxes for garments (shirt, pants,
jacket, dress, skirt, shorts, ...) plus accessories (bag, shoe, hat,
sunglass).

For every *clothing* bounding box (accessories are skipped by default) this
module:
  1. Crops the garment region from the source photo (shrinking the box
     slightly to reduce background bleed).
  2. Saves a small thumbnail crop to `data/processed/crops/` (used later by
     the Streamlit UI to show *why* an item was recommended).
  3. Runs KMeans over the crop's pixels to find the dominant color cluster.
  4. Converts that dominant color to CIELAB (needed for CIEDE2000 matching
     in `recommender.py`).

Output: `data/processed/garment_catalog.csv` with one row per garment
instance.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import pandas as pd
from tqdm import tqdm

from src.color_utils import dominant_lab_color

# Garment (clothing) categories we want to index for color matching.
# Accessories are excluded since the recommendation dataset speaks to
# *clothing* colors specifically (shoes/jewelry have their own columns).
DEFAULT_GARMENT_CLASSES = {"shirt", "pants", "jacket", "dress", "skirt", "shorts"}

DEFAULT_IMAGES_DIR = Path("data/raw/colorful_fashion/JPEGImages")
DEFAULT_ANNOTATIONS_DIR = Path("data/raw/colorful_fashion/Annotations")
DEFAULT_OUTPUT_CSV = Path("data/processed/garment_catalog.csv")
DEFAULT_CROPS_DIR = Path("data/processed/crops")


def _parse_voc_xml(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename", default=xml_path.stem + ".jpg")
    size_node = root.find("size")
    img_w = int(size_node.findtext("width", default="0")) if size_node is not None else 0
    img_h = int(size_node.findtext("height", default="0")) if size_node is not None else 0

    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = int(float(bbox.findtext("xmin", default="0")))
        ymin = int(float(bbox.findtext("ymin", default="0")))
        xmax = int(float(bbox.findtext("xmax", default="0")))
        ymax = int(float(bbox.findtext("ymax", default="0")))
        objects.append(
            {"name": name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        )

    return {
        "filename": filename,
        "width": img_w,
        "height": img_h,
        "objects": objects,
    }


def _shrink_box(
    xmin: int, ymin: int, xmax: int, ymax: int, shrink: float = 0.12
) -> tuple:
    """Shrink a bbox inward by `shrink` fraction on each side to reduce the
    chance of sampling background pixels around the garment edges."""
    w, h = xmax - xmin, ymax - ymin
    dx, dy = int(w * shrink), int(h * shrink)
    return xmin + dx, ymin + dy, max(xmax - dx, xmin + dx + 1), max(ymax - dy, ymin + dy + 1)


def index_garments(
    images_dir: Path = DEFAULT_IMAGES_DIR,
    annotations_dir: Path = DEFAULT_ANNOTATIONS_DIR,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    crops_dir: Path = DEFAULT_CROPS_DIR,
    garment_classes: Optional[Iterable[str]] = None,
    k: int = 3,
    limit: Optional[int] = None,
    save_crops: bool = True,
) -> pd.DataFrame:
    """Index every garment bounding box found in the VOC annotations,
    computing a dominant Lab color for each, and write the result to
    `output_csv`. Returns the resulting DataFrame as well.
    """
    garment_classes = set(c.lower() for c in (garment_classes or DEFAULT_GARMENT_CLASSES))
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    output_csv = Path(output_csv)
    crops_dir = Path(crops_dir)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if limit:
        xml_files = xml_files[:limit]

    rows: List[dict] = []
    garment_counter = 0

    for xml_path in tqdm(xml_files, desc="Indexing garments"):
        try:
            parsed = _parse_voc_xml(xml_path)
        except ET.ParseError:
            continue

        image_path = images_dir / parsed["filename"]
        if not image_path.exists():
            continue

        garment_objs = [o for o in parsed["objects"] if o["name"] in garment_classes]
        if not garment_objs:
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        h_img, w_img = image_bgr.shape[:2]

        for obj in garment_objs:
            xmin, ymin, xmax, ymax = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
            xmin, ymin = max(xmin, 0), max(ymin, 0)
            xmax, ymax = min(xmax, w_img), min(ymax, h_img)
            if xmax - xmin < 6 or ymax - ymin < 6:
                continue

            sxmin, symin, sxmax, symax = _shrink_box(xmin, ymin, xmax, ymax)
            crop_bgr = image_bgr[symin:symax, sxmin:sxmax]
            if crop_bgr.size == 0:
                crop_bgr = image_bgr[ymin:ymax, xmin:xmax]
            if crop_bgr.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            dom = dominant_lab_color(crop_rgb, k=k)
            dom_r, dom_g, dom_b = dom["rgb"]
            L, a, b_ = dom["lab"]
            hex_code = dom["hex"]

            garment_id = f"g{garment_counter:06d}"
            crop_path = ""
            if save_crops:
                crop_path = str(crops_dir / f"{garment_id}.jpg")
                cv2.imwrite(crop_path, crop_bgr)

            rows.append(
                {
                    "garment_id": garment_id,
                    "source_image": parsed["filename"],
                    "category": obj["name"],
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "image_width": w_img,
                    "image_height": h_img,
                    "dominant_r": round(dom_r, 2),
                    "dominant_g": round(dom_g, 2),
                    "dominant_b": round(dom_b, 2),
                    "dominant_hex": hex_code,
                    "lab_L": round(L, 3),
                    "lab_a": round(a, 3),
                    "lab_b": round(b_, 3),
                    "crop_path": crop_path,
                    "source_image_path": str(image_path),
                }
            )
            garment_counter += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index garment bounding boxes -> dominant Lab colors."
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    parser.add_argument("--k", type=int, default=3, help="KMeans cluster count")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process first N XML files (debug)"
    )
    parser.add_argument(
        "--no-crops", action="store_true", help="Skip writing cropped thumbnail images"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated garment classes to index (default: shirt,pants,jacket,dress,skirt,shorts)",
    )
    args = parser.parse_args()

    classes = args.classes.split(",") if args.classes else None

    df = index_garments(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        output_csv=args.output_csv,
        crops_dir=args.crops_dir,
        garment_classes=classes,
        k=args.k,
        limit=args.limit,
        save_crops=not args.no_crops,
    )
    print(f"Indexed {len(df)} garments -> {args.output_csv}")
    if not df.empty:
        print(df["category"].value_counts())


if __name__ == "__main__":
    main()
