"""
app.py
======
Streamlit UI for the end-to-end Color Analysis & Clothing Recommendation
pipeline. Two modes, selectable in the sidebar:

  A) "Get My Colors" -- capture a selfie (webcam via `st.camera_input`,
     which uses your browser's camera - no extra setup needed) or upload
     a photo. `SkinAnalyzer` (MediaPipe FaceMesh) samples cheek pixels
     and derives CIELAB / ITA -> Skin Tone + Undertone. `PaletteEngine`
     then returns a recommended / avoid color palette + styling tips,
     and `Recommender` ranks real clothing photos from the indexed
     catalog by CIEDE2000 distance to that palette.

  B) "My Closet" -- upload photos of your OWN clothes. Each item is
     scored against your palette ("Personal Match %"), pairs of items
     are scored for outfit color harmony ("shirt + jeans -> X% match"),
     and `closet_engine.generate_combos` ranks every valid outfit
     combination across everything you uploaded. You can also save any
     item into a persistent catalog so it shows up in future
     recommendations too.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.closet_engine import (
    CATEGORY_SLOT,
    analyze_uploaded_item,
    generate_combos,
    harmony_score,
    personal_match,
    verdict_label,
)
from src.closet_store import add_user_garment
from src.color_utils import crop_by_margins
from src.palette_engine import PaletteEngine
from src.recommender import Recommender
from src.skin_analyzer import NoFaceDetectedError, SkinAnalyzer

st.set_page_config(
    page_title="Color Analysis & Clothing Recommender",
    page_icon="🎨",
    layout="wide",
)

CATALOG_CSV = Path("data/processed/garment_catalog.csv")
RULES_CSV = Path("data/raw/fashion_and_color_recommendation.csv")
GARMENT_CATEGORIES = list(CATEGORY_SLOT.keys())  # shirt, jacket, pants, shorts, skirt, dress


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_skin_analyzer() -> SkinAnalyzer:
    return SkinAnalyzer(static_image_mode=True)


@st.cache_resource(show_spinner=False)
def get_palette_engine() -> PaletteEngine:
    return PaletteEngine(RULES_CSV)


@st.cache_resource(show_spinner=False)
def get_recommender() -> Recommender | None:
    if not CATALOG_CSV.exists():
        return None
    return Recommender(CATALOG_CSV)


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def render_swatch_row(swatches: list, chip_size_px: int = 64) -> None:
    if not swatches:
        st.caption("No swatches available.")
        return
    cols = st.columns(len(swatches))
    for col, sw in zip(cols, swatches):
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color:{sw['hex']};
                    width:100%; height:{chip_size_px}px;
                    border-radius:8px; border:1px solid rgba(0,0,0,0.15);
                "></div>
                <p style="text-align:center; font-size:0.8rem; margin-top:4px;">
                    {sw['name']}<br><code>{sw['hex']}</code>
                </p>
                """,
                unsafe_allow_html=True,
            )


def render_score_bar(pct: float, verdict: str, caption: str = "") -> None:
    """A small colored progress bar + verdict label, used throughout the
    My Closet section."""
    color = "#2E9E5B" if pct >= 75 else "#C7A400" if pct >= 55 else (
        "#D9822B" if pct >= 35 else "#C1392B"
    )
    st.markdown(
        f"""
        <div style="margin: 4px 0 2px 0;">
            <div style="background:#eee; border-radius:6px; height:14px; width:100%;">
                <div style="background:{color}; width:{pct}%; height:14px; border-radius:6px;"></div>
            </div>
            <p style="margin:2px 0 0 0; font-size:0.85rem;">
                <b>{pct:.0f}%</b> &middot; {verdict}{(' &middot; ' + caption) if caption else ''}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def swatch_dot(hex_code: str, size: int = 16) -> str:
    return (
        f'<span style="display:inline-block; width:{size}px; height:{size}px; '
        f'border-radius:4px; background:{hex_code}; border:1px solid rgba(0,0,0,0.2); '
        f'vertical-align:middle;"></span>'
    )


# ---------------------------------------------------------------------------
# Sidebar: top-level mode switch
# ---------------------------------------------------------------------------
st.sidebar.title("🎨 Color Recommender")
mode = st.sidebar.radio(
    "Mode", ["🧑 Get My Colors", "👕 My Closet"], index=0,
    help="Analyze your skin tone, or upload your own clothes to check fit & mix-and-match.",
)
st.sidebar.markdown("---")

palette_engine = get_palette_engine()
recommender = get_recommender()

if "skin_result" not in st.session_state:
    st.session_state.skin_result = None
if "active_palette" not in st.session_state:
    st.session_state.active_palette = None
if "closet_data" not in st.session_state:
    st.session_state.closet_data = []  # list of dicts, see analyze step below


# ===========================================================================
# MODE A: Get My Colors
# ===========================================================================
def page_get_my_colors() -> None:
    st.sidebar.subheader("Photo source")
    input_mode = st.sidebar.radio(
        "Photo source", ["Webcam", "Upload a photo"], index=0,
        key="colors_input_mode", label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Optional body-shape filters")
    st.sidebar.caption(
        "These refine the *styling* tips (fit, materials, patterns). "
        "Color recommendations depend only on skin tone & undertone."
    )
    body_proportion = st.sidebar.selectbox(
        "Body proportion", ["(skip)"] + palette_engine.available_body_proportions(), index=0
    )
    torso_length = st.sidebar.selectbox(
        "Torso length", ["(skip)"] + palette_engine.available_torso_lengths(), index=0
    )
    body_proportion = None if body_proportion == "(skip)" else body_proportion
    torso_length = None if torso_length == "(skip)" else torso_length

    st.sidebar.markdown("---")
    st.sidebar.subheader("Garment recommendations")
    category_options = ["All"] + (recommender.categories() if recommender is not None else [])
    category_choice = st.sidebar.selectbox("Garment type", category_options, index=0)
    top_n = st.sidebar.slider("Number of items to show", 4, 24, 12)

    if recommender is None:
        st.sidebar.warning(
            "Garment catalog not found. Run:\n\n`python -m src.garment_indexer`\n\n"
            "to build `data/processed/garment_catalog.csv` first."
        )

    st.title("🎨 Color Analysis & Clothing Recommender")
    st.write(
        "Take a selfie or upload a photo. We'll analyze your skin tone and "
        "undertone from CIELAB / ITA, then recommend real clothing items "
        "whose dominant color best matches your ideal palette (CIEDE2000)."
    )

    image_bgr = None
    if input_mode == "Webcam":
        camera_file = st.camera_input("Take a selfie")
        if camera_file is not None:
            image_bgr = decode_uploaded_image(camera_file)
    else:
        uploaded_file = st.file_uploader("Upload a selfie", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            image_bgr = decode_uploaded_image(uploaded_file)

    if image_bgr is None:
        st.info("👆 Provide a photo to get started.")
        return

    analyzer = get_skin_analyzer()
    with st.spinner("Analyzing skin tone..."):
        try:
            result = analyzer.analyze(image_bgr, draw_debug=True)
        except NoFaceDetectedError as exc:
            st.error(str(exc))
            return

    col_img, col_metrics = st.columns([1, 1])
    with col_img:
        st.subheader("Detected face & sampling regions")
        annotated_rgb = cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, width='stretch')
        st.caption("Green circles mark the cheek regions sampled for skin color.")

    with col_metrics:
        st.subheader("Skin analysis results")
        m1, m2 = st.columns(2)
        m1.metric("Skin Tone", result.skin_tone)
        m2.metric("Undertone", result.undertone)
        m3, m4 = st.columns(2)
        m3.metric("ITA (°)", f"{result.ita:.1f}")
        m4.metric("Lab L*", f"{result.lab[0]:.1f}")
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px; margin-top:8px;">
                <div style="background-color:{result.hex}; width:56px; height:56px;
                            border-radius:8px; border:1px solid rgba(0,0,0,0.2);"></div>
                <div>
                    <b>Sampled skin color</b><br>
                    RGB {result.rgb} &nbsp;·&nbsp; <code>{result.hex}</code><br>
                    Lab {result.lab}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Persist for reuse in the My Closet tab.
    st.session_state.skin_result = result

    st.markdown("---")
    st.header("Your recommended color palette")
    palette = palette_engine.get_palette(result.skin_tone, result.undertone, body_proportion, torso_length)
    st.session_state.active_palette = palette

    st.subheader("✅ Recommended clothing colors")
    render_swatch_row(palette["recommended_swatches"])
    st.subheader("🚫 Colors to avoid")
    render_swatch_row(palette["avoid_swatches"])

    with st.expander("Full styling guidance"):
        style = palette["style"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Fitting style:** {', '.join(style['fitting_style'])}")
            st.markdown(f"**Materials:** {', '.join(style['materials'])}")
            st.markdown(f"**Patterns:** {', '.join(style['patterns'])}")
            st.markdown(f"**Fabric nature:** {', '.join(style['fabric_nature'])}")
        with c2:
            st.markdown(f"**Jewelry metal:** {', '.join(style['jewelry_metal'])}")
            st.markdown(f"**Shoes:** {', '.join(style['shoes'])}")
            st.markdown(f"**Don't exaggerate:** {', '.join(style['dont_exaggerate'])}")
            st.markdown(f"**Do exaggerate:** {', '.join(style['do_exaggerate'])}")

    st.markdown("---")
    st.header("👗 Matching clothing items")
    if recommender is None:
        st.warning("No garment catalog available yet. Build it once with:\n\n```\npython -m src.garment_indexer\n```")
    else:
        category_arg = None if category_choice == "All" else category_choice
        ranked = recommender.rank_from_palette(palette, category=category_arg, top_n=top_n)
        if ranked.empty:
            st.info("No garments matched the current filters. Try 'All' categories.")
        else:
            n_cols = 4
            rows = [ranked.iloc[i : i + n_cols] for i in range(0, len(ranked), n_cols)]
            for row_chunk in rows:
                cols = st.columns(n_cols)
                for col, (_, item) in zip(cols, row_chunk.iterrows()):
                    with col:
                        crop_path = item.get("crop_path", "")
                        if crop_path and Path(crop_path).exists():
                            st.image(str(crop_path), width='stretch')
                        source_tag = " · your closet" if item.get("source") == "user_upload" else ""
                        st.markdown(
                            f"""
                            <div style="display:flex; align-items:center; gap:6px;">
                                {swatch_dot(item['dominant_hex'])}
                                <b>{item['category'].title()}</b>
                            </div>
                            <p style="font-size:0.8rem; margin:2px 0 0 0;">
                                ΔE00 match: {item['match_distance']:.2f} ·
                                <code>{item['dominant_hex']}</code>{source_tag}
                            </p>
                            """,
                            unsafe_allow_html=True,
                        )

    st.markdown("---")
    st.caption(
        "Skin tone/undertone estimation is a heuristic based on ITA and CIELAB "
        "hue and is sensitive to lighting and camera white balance. For best "
        "results, use even, natural light and a neutral-colored background."
    )


# ===========================================================================
# MODE B: My Closet -- upload your own clothes
# ===========================================================================
def page_my_closet() -> None:
    st.title("👕 My Closet")
    st.write(
        "Upload photos of your own clothes. Each item is scored against "
        "your color palette, pairs are scored for outfit harmony (e.g. "
        "*does this shirt work with these jeans?*), and every valid "
        "combination across your uploads is ranked for you."
    )

    # -- 1. Color profile (reuse the analyzed one, or pick manually) --------
    st.sidebar.subheader("Your color profile")
    has_analysis = st.session_state.skin_result is not None
    profile_mode = st.sidebar.radio(
        "Profile source",
        (["Use my analyzed profile", "Pick manually"] if has_analysis else ["Pick manually"]),
        index=0,
    )

    if profile_mode == "Use my analyzed profile" and has_analysis:
        skin_tone = st.session_state.skin_result.skin_tone
        undertone = st.session_state.skin_result.undertone
        st.sidebar.success(f"Using: {skin_tone} skin, {undertone} undertone")
    else:
        skin_tone = st.sidebar.selectbox("Skin tone", palette_engine.available_skin_tones())
        undertone = st.sidebar.selectbox("Undertone", palette_engine.available_undertones())

    palette = palette_engine.get_palette(skin_tone, undertone)
    st.info(f"Scoring against: **{skin_tone} skin / {undertone} undertone** palette.")

    # -- 2. Upload + per-item setup -----------------------------------------
    st.subheader("1. Upload your clothes")
    uploaded_files = st.file_uploader(
        "One photo per item works best (a fairly plain background helps color detection).",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("👆 Upload one or more clothing photos to get started.")
        return

    st.caption(
        "For each item: pick its category and, if needed, drag the crop "
        "margins so the frame contains mostly the garment (not skin/background)."
    )

    item_configs = []
    for i, f in enumerate(uploaded_files):
        img_bgr = decode_uploaded_image(f)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.expander(f"🧥 {f.name}", expanded=(i == 0)):
            c1, c2 = st.columns([1, 1])
            with c1:
                category = st.selectbox(
                    "Category", GARMENT_CATEGORIES, key=f"cat_{f.name}_{i}"
                )
                nickname = st.text_input(
                    "Nickname (optional)", value=Path(f.name).stem, key=f"nick_{f.name}_{i}"
                )
                st.caption("Crop margins (%) -- trim away background/skin")
                left = st.slider("Left", 0, 45, 0, key=f"l_{f.name}_{i}")
                right = st.slider("Right", 0, 45, 0, key=f"r_{f.name}_{i}")
                top = st.slider("Top", 0, 45, 0, key=f"t_{f.name}_{i}")
                bottom = st.slider("Bottom", 0, 45, 0, key=f"b_{f.name}_{i}")
            with c2:
                cropped_rgb = crop_by_margins(img_rgb, left, right, top, bottom)
                st.image(cropped_rgb, caption="Cropped preview", width='stretch')

        item_configs.append(
            {
                "filename": f.name,
                "nickname": nickname,
                "category": category,
                "crop_bgr": cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR),
            }
        )

    # -- 3. Analyze --------------------------------------------------------
    st.subheader("2. Analyze")
    if st.button("🔍 Analyze My Closet", type="primary"):
        analyzed = []
        for i, cfg in enumerate(item_configs):
            item = analyze_uploaded_item(
                crop_bgr=cfg["crop_bgr"],
                item_id=f"closet_{i}",
                label=cfg["nickname"] or cfg["filename"],
                category=cfg["category"],
                palette=palette,
            )
            analyzed.append({**cfg, "item": item, "added": False})
        st.session_state.closet_data = analyzed

    closet_data = st.session_state.closet_data
    if not closet_data:
        st.info("Set categories/crops above, then click **Analyze My Closet**.")
        return

    # -- 4. Per-item personal match results ---------------------------------
    st.subheader("3. Does each item suit you?")
    n_cols = 3
    rows = [closet_data[i : i + n_cols] for i in range(0, len(closet_data), n_cols)]
    for row_chunk in rows:
        cols = st.columns(n_cols)
        for col, entry in zip(cols, row_chunk):
            item = entry["item"]
            with col:
                st.image(
                    cv2.cvtColor(entry["crop_bgr"], cv2.COLOR_BGR2RGB),
                    width='stretch',
                )
                st.markdown(
                    f"**{item.label}** ({item.category}) {swatch_dot(item.hex)} `{item.hex}`",
                    unsafe_allow_html=True,
                )
                render_score_bar(item.personal.pct, item.personal.verdict, "personal match")
                if item.personal.flagged_avoid:
                    st.warning("This color is close to one on your **avoid** list.")
                if item.personal.closest_recommended:
                    st.caption(
                        f"Closest recommended color: {item.personal.closest_recommended['name']} "
                        f"({item.personal.closest_recommended['hex']})"
                    )
                if not entry["added"]:
                    if st.button("➕ Add to garment catalog", key=f"add_{entry['filename']}"):
                        add_user_garment(
                            crop_bgr=entry["crop_bgr"],
                            category=item.category,
                            dominant_rgb=item.rgb,
                            dominant_hex=item.hex,
                            lab=item.lab,
                            original_filename=entry["filename"],
                        )
                        if recommender is not None:
                            recommender.reload()
                        entry["added"] = True
                        st.success("Saved! It'll now show up in future recommendations too.")
                else:
                    st.caption("✅ Added to catalog")

    # -- 5. Check a specific pairing -----------------------------------------
    st.markdown("---")
    st.subheader("4. Check a specific pairing")
    st.caption("e.g. *does this shirt work with these jeans?* -- pick any two items.")
    labels = [f"{e['item'].label} ({e['item'].category})" for e in closet_data]
    c1, c2 = st.columns(2)
    idx_a = c1.selectbox("Item A", range(len(labels)), format_func=lambda i: labels[i], key="pair_a")
    idx_b = c2.selectbox(
        "Item B", range(len(labels)), format_func=lambda i: labels[i],
        index=min(1, len(labels) - 1), key="pair_b",
    )
    if idx_a == idx_b:
        st.caption("Pick two different items to compare.")
    else:
        item_a, item_b = closet_data[idx_a]["item"], closet_data[idx_b]["item"]
        h = harmony_score(item_a.lab, item_b.lab)
        personal_avg = (item_a.personal.pct + item_b.personal.pct) / 2
        overall = round(0.5 * personal_avg + 0.5 * h.pct, 1)
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.metric("Personal fit (avg)", f"{personal_avg:.0f}%")
        with cc2:
            st.metric("Outfit harmony", f"{h.pct:.0f}%", help=h.relationship)
        with cc3:
            st.metric("Overall", f"{overall:.0f}%")
        render_score_bar(overall, verdict_label(overall), h.relationship)

    # -- 6. Mix & match across the whole closet ------------------------------
    st.markdown("---")
    st.subheader("5. 🧩 Mix & match: your best outfit combos")
    items_only = [e["item"] for e in closet_data]
    combos = generate_combos(items_only, top_n=8)

    if not combos:
        st.info(
            "Add at least one **top + bottom** pair (or a **dress**) to see "
            "outfit combos -- accessories/jackets alone aren't enough."
        )
    else:
        for combo in combos:
            piece_labels = " + ".join(f"{i.label} ({i.category})" for i in combo.items)
            with st.container(border=True):
                cols = st.columns([2, 1, 1, 1])
                with cols[0]:
                    st.markdown(f"**{piece_labels}**")
                    if combo.harmony_details:
                        rel_text = "; ".join(f"{a}↔{b}: {r}" for a, b, r in combo.harmony_details)
                        st.caption(rel_text)
                cols[1].metric("Personal", f"{combo.personal_pct:.0f}%")
                cols[2].metric("Harmony", f"{combo.harmony_pct:.0f}%")
                cols[3].metric("Overall", f"{combo.overall_pct:.0f}%")
                render_score_bar(combo.overall_pct, combo.verdict)

    st.markdown("---")
    st.caption(
        "Outfit harmony uses hue-relationship color theory (monochrome / "
        "analogous / complementary) plus neutral detection -- it's a "
        "heuristic guide, not a hard rule. Trust your eye too!"
    )


# ---------------------------------------------------------------------------
if mode == "🧑 Get My Colors":
    page_get_my_colors()
else:
    page_my_closet()
