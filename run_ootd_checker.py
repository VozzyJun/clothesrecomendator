"""
run_ootd_checker.py
===================
Standalone CLI and Live Webcam runner for the OOTD Full-Body Live Checker.
Uses YOLOv8-pose + Personal Color Analysis + Outfit Harmony Scoring.

Usage:
  # 1. Run demo with a synthetic full-body test subject:
  python run_ootd_checker.py --demo

  # 2. Analyze a full-body photo or mirror selfie:
  python run_ootd_checker.py --image path/to/ootd.jpg --output annotated_ootd.jpg

  # 3. Live webcam mode:
  python run_ootd_checker.py --webcam
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.ootd_analyzer import OOTDAnalyzer


def create_demo_ootd_image() -> np.ndarray:
    """Create a realistic synthetic full-body silhouette for headless testing."""
    img = np.full((720, 480, 3), (242, 240, 238), dtype=np.uint8)

    # Head (skin tone)
    cv2.circle(img, (240, 110), 45, (160, 190, 225), -1)

    # Neck
    cv2.rectangle(img, (228, 150), (252, 175), (150, 180, 215), -1)

    # Top garment (Rich warm terracotta #A0522D / BGR: (45, 82, 160))
    cv2.rectangle(img, (170, 175), (310, 360), (45, 82, 160), -1)
    # Arms
    cv2.rectangle(img, (135, 185), (170, 320), (45, 82, 160), -1)
    cv2.rectangle(img, (310, 185), (345, 320), (45, 82, 160), -1)

    # Hands
    cv2.circle(img, (152, 340), 16, (160, 190, 225), -1)
    cv2.circle(img, (328, 340), 16, (160, 190, 225), -1)

    # Bottom garment (Classic Dark Navy #1C2833 / BGR: (51, 40, 28))
    # Left leg
    cv2.rectangle(img, (180, 360), (235, 590), (51, 40, 28), -1)
    # Right leg
    cv2.rectangle(img, (245, 360), (300, 590), (51, 40, 28), -1)

    # Shoes (Warm brown leather / BGR: (30, 60, 100))
    cv2.rectangle(img, (170, 590), (235, 625), (30, 60, 100), -1)
    cv2.rectangle(img, (245, 590), (310, 625), (30, 60, 100), -1)

    return img


def print_ootd_report(res) -> None:
    print("\n" + "=" * 60)
    print("        OOTD FULL-BODY LIVE CHECKER REPORT")
    print("=" * 60)
    if not res.person_detected:
        print("[!] No person detected in the image.")
        return

    print(f"Overall OOTD Score  : {res.overall_score:.1f} / 100")
    print(f"Personal Profile    : {res.skin_tone} Skin, {res.undertone} Undertone ({res.season})")

    if res.body_proportions:
        bp = res.body_proportions
        print(f"Silhouette & Torso  : {bp.silhouette_shape} | {bp.torso_length_label} (Ratio: {bp.torso_ratio:.2f})")

    print("-" * 60)
    if res.top_garment:
        top = res.top_garment
        verdict = top.personal_match.verdict if top.personal_match else "N/A"
        pct = top.personal_match.pct if top.personal_match else 0
        print(f"Top Garment (Atasan): {top.dominant_hex} | Match: {verdict} ({pct:.0f}%)")

    if res.bottom_garment:
        bot = res.bottom_garment
        verdict = bot.personal_match.verdict if bot.personal_match else "N/A"
        pct = bot.personal_match.pct if bot.personal_match else 0
        print(f"Bottom (Bawahan)    : {bot.dominant_hex} | Match: {verdict} ({pct:.0f}%)")

    if res.outfit_harmony:
        h = res.outfit_harmony
        print(f"Outfit Harmony      : {h.relationship} ({h.pct:.0f}%)")

    print("-" * 60)
    print("Styling Insights & Critique:")
    for note in res.feedback_notes:
        print(f"  * {note}")

    if res.body_proportions and res.body_proportions.styling_tips:
        tips = res.body_proportions.styling_tips
        avoid_tip = tips.get("Don't Exaggerate")
        fit_tip = tips.get("Recommended Fitting Style")
        do_tip = tips.get("Do Exaggerate")
        if fit_tip:
            print(f"  * Fitting Style  : {fit_tip}")
        if do_tip:
            print(f"  * Do Exaggerate  : {do_tip}")
        if avoid_tip:
            print(f"  * Avoid          : {avoid_tip}")

    if res.shopee_suggestions:
        print("\nShopee Upgrade Recommendations:")
        for s in res.shopee_suggestions:
            print(f"  [+] {s['type']}: {s['reason']}")
            print(f"      Link: {s['url']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="OOTD Full-Body Live Checker")
    parser.add_argument("--image", type=str, help="Path to full-body image")
    parser.add_argument("--output", type=str, default="ootd_result.jpg", help="Path to save annotated output")
    parser.add_argument("--webcam", action="store_true", help="Launch live webcam checker")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic full-body test image")
    parser.add_argument("--skin-tone", type=str, default=None, help="Optional manual skin tone")
    parser.add_argument("--undertone", type=str, default=None, help="Optional manual undertone")
    args = parser.parse_args()

    print("[*] Initializing OOTD Analyzer with YOLOv8-pose...")
    analyzer = OOTDAnalyzer()
    print("[*] Analyzer ready.")

    if args.webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[!] Error: Could not open webcam.")
            return

        print("[*] Live OOTD Checker running. Press 'q' to quit, 's' to snapshot & score.")
        last_analysis_time = 0
        cached_result = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            # Run analysis every 0.5s to maintain smooth 30 FPS display
            if now - last_analysis_time > 0.5:
                cached_result = analyzer.analyze(
                    frame, user_skin_tone=args.skin_tone, user_undertone=args.undertone
                )
                last_analysis_time = now

            if cached_result:
                annotated = analyzer.annotate(frame, cached_result)
            else:
                annotated = frame

            cv2.imshow("MyHue OOTD Full-Body Live Checker", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and cached_result:
                print_ootd_report(cached_result)

        cap.release()
        cv2.destroyAllWindows()
        return

    # Image or Demo mode
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[!] Error: Image not found at {img_path}")
            return
        img_bgr = cv2.imread(str(img_path))
    else:
        # Default to demo mode
        print("[*] Running in demo mode (synthetic full-body image)...")
        img_bgr = create_demo_ootd_image()

    res = analyzer.analyze(img_bgr, user_skin_tone=args.skin_tone, user_undertone=args.undertone)
    print_ootd_report(res)

    annotated = analyzer.annotate(img_bgr, res)
    cv2.imwrite(args.output, annotated)
    print(f"[✓] Annotated OOTD image saved to: {args.output}")


if __name__ == "__main__":
    main()
