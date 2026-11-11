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

import base64
import io
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image as PILImage

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
from src.pdf_generator import build_pdf_report
from src.recommender import Recommender
from src.skin_analyzer import NoFaceDetectedError, SkinAnalyzer

st.set_page_config(
    page_title="MyHue - Color Analysis & Clothing Recommender",
    page_icon="🎨",
    layout="wide",
)

LOGO_PATH = Path("assets/logo.png")
CATALOG_CSV = Path("data/processed/garment_catalog.csv")
RULES_CSV = Path("data/raw/fashion_and_color_recommendation.csv")
GARMENT_CATEGORIES = list(CATEGORY_SLOT.keys())  # shirt, jacket, pants, shorts, skirt, dress

# Palette used throughout the custom-rendered HTML snippets below. Keep in
# sync with .streamlit/config.toml (which themes the native widgets).
THEME = {
    "accent": "#B5562E",       # terracotta -- primary accent
    "accent_dark": "#8B4020",
    "accent_soft": "#F1D9C9",
    "ink": "#2E2823",          # main text
    "ink_soft": "#6B6055",     # secondary text
    "paper": "#FFFFFF",        # card background
    "bg": "#FBF7F2",           # page background
    "border": "rgba(46,40,35,0.10)",
    "good": "#2E9E5B",
    "ok": "#C7A400",
    "caution": "#D9822B",
    "avoid": "#C1392B",
}


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* headings use a warm editorial serif */
        h1, h2, h3 {{
            font-family: 'Fraunces', Georgia, serif !important;
            color: {THEME['ink']} !important;
            font-weight: 600 !important;
        }}

        /* tighten the default top padding */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        /* sidebar */
        [data-testid="stSidebar"] {{
            background: {THEME['bg']};
            border-right: 1px solid {THEME['border']};
        }}
        [data-testid="stSidebar"] h1 {{
            font-size: 1.35rem !important;
        }}

        /* thin, accent-tinted dividers everywhere */
        hr {{
            border: none;
            border-top: 1px solid {THEME['border']};
            margin: 1.1rem 0;
        }}

        /* buttons */
        .stButton > button {{
            border-radius: 999px;
            font-weight: 600;
            border: 1px solid {THEME['border']};
            transition: transform 0.05s ease-in-out;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            border-color: {THEME['accent']};
            color: {THEME['accent']};
        }}
        .stButton > button[kind="primary"] {{
            background: {THEME['accent']};
            border: none;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: {THEME['accent_dark']};
            color: white;
        }}

        /* metrics */
        [data-testid="stMetric"] {{
            background: {THEME['paper']};
            border: 1px solid {THEME['border']};
            border-radius: 12px;
            padding: 0.6rem 0.9rem;
        }}
        [data-testid="stMetricLabel"] {{
            color: {THEME['ink_soft']};
        }}

        /* expanders */
        [data-testid="stExpander"] {{
            background: {THEME['paper']};
            border: 1px solid {THEME['border']};
            border-radius: 12px;
        }}

        /* camera / file uploader */
        [data-testid="stCameraInput"], [data-testid="stFileUploaderDropzone"] {{
            border-radius: 12px;
        }}

        /* images get soft rounded corners */
        [data-testid="stImage"] img {{
            border-radius: 10px;
        }}

        /* generic "card" surfaces used via st.container(key=...) */
        .st-key-hero_a, .st-key-hero_b {{
            background: linear-gradient(135deg, {THEME['accent']} 0%, {THEME['accent_dark']} 100%);
            border-radius: 18px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.4rem;
        }}
        .st-key-hero_a h1, .st-key-hero_b h1 {{
            color: white !important;
            margin-bottom: 0.2rem !important;
        }}
        .st-key-hero_a p, .st-key-hero_b p {{
            color: rgba(255,255,255,0.92) !important;
            font-size: 1.02rem;
            margin: 0;
        }}

        div[class*="st-key-item_card_"], div[class*="st-key-combo_card_"], div[class*="st-key-catalog_card_"] {{
            background: {THEME['paper']};
            border: 1px solid {THEME['border']};
            border-radius: 14px;
            padding: 0.9rem 1rem 1.1rem 1rem;
            box-shadow: 0 1px 3px rgba(46,40,35,0.06);
            margin-bottom: 0.9rem;
        }}

        .st-key-profile_box {{
            background: {THEME['accent_soft']};
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.6rem;
        }}

        .st-key-pdf_export_card {{
            background: linear-gradient(135deg, #FBF7F2 0%, #F5ECE3 100%);
            border: 1px solid rgba(181, 86, 46, 0.25);
            border-radius: 14px;
            padding: 1.1rem 1.4rem;
            margin: 1.2rem 0;
            box-shadow: 0 2px 6px rgba(46,40,35,0.04);
        }}

        a[href*="shopee.co.id"]:hover {{
            filter: brightness(1.08);
            transform: translateY(-1px);
            transition: all 0.15s ease-in-out;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(icon: str, title: str, subtitle: str, key: str) -> None:
    with st.container(key=key):
        st.markdown(f"# {icon} {title}")
        st.markdown(f"<p>{subtitle}</p>", unsafe_allow_html=True)


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


def render_safe_image(
    img_input: np.ndarray | str | Path | PILImage.Image,
    caption: str | None = None,
    max_height: int | None = None,
) -> None:
    """Renders images reliably as embedded base64 data URIs.

    This bypasses Streamlit's internal MediaFileManager race conditions
    (which cause images to fail with a broken icon and '0') and works identically
    across all Streamlit versions and cloud hosting environments.
    """
    if img_input is None:
        return

    mime = "image/jpeg"
    b64_str = ""

    if isinstance(img_input, (str, Path)):
        p = Path(img_input)
        if not p.exists():
            st.caption("Image not available")
            return
        try:
            with open(p, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("utf-8")
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        except Exception:
            st.caption("Unable to load image")
            return
    elif isinstance(img_input, np.ndarray):
        try:
            pil_img = PILImage.fromarray(img_input)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            mime = "image/jpeg"
        except Exception:
            st.caption("Unable to process image")
            return
    elif isinstance(img_input, PILImage.Image):
        try:
            buf = io.BytesIO()
            fmt = img_input.format or "JPEG"
            img_input.save(buf, format=fmt)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            mime = f"image/{fmt.lower()}"
        except Exception:
            st.caption("Unable to process image")
            return

    ink_soft = THEME["ink_soft"]
    cap_html = (
        f'<p style="font-size:0.8rem; color:{ink_soft}; margin-top:5px; text-align:center;">{caption}</p>'
        if caption
        else ""
    )
    style = "width:100%; border-radius:10px; display:block; object-fit:cover;"
    if max_height:
        style += f" max-height:{max_height}px;"

    alt_text = caption if caption else "image"
    st.markdown(
        f'<div style="width:100%; margin-bottom:8px;">'
        f'<img src="data:{mime};base64,{b64_str}" style="{style}" alt="{alt_text}" />'
        f'{cap_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def shopee_button_html(
    query_text: str,
    label: str = "Beli di Shopee",
    target_audience: str = "Semua",
) -> str:
    """Generates an authentic Shopee search button linking to Shopee Indonesia."""
    parts = []
    if target_audience and target_audience != "Semua":
        parts.append(target_audience.lower())

    cat_map = {
        "shirt": "kemeja baju",
        "pants": "celana",
        "jacket": "jaket outer",
        "shorts": "celana pendek",
        "skirt": "rok",
        "dress": "dress terusan",
    }
    words = query_text.split()
    translated_words = [cat_map.get(w.lower(), w) for w in words]
    parts.extend(translated_words)

    query = " ".join(parts).strip()
    encoded = urllib.parse.quote(query)
    url = f"https://shopee.co.id/search?keyword={encoded}"

    return (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="'
        f'display:flex; align-items:center; justify-content:center; gap:5px; '
        f'background-color:#EE4D2D; color:#FFFFFF !important; text-decoration:none !important; '
        f'padding:6px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; '
        f'box-shadow: 0 1px 3px rgba(238,77,45,0.25); margin-top:6px; text-align:center;">'
        f'<span style="font-size:0.85rem;">🛍️</span> {label}'
        f'</a>'
    )


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
                    border-radius:10px; border:1px solid rgba(0,0,0,0.12);
                    box-shadow: 0 1px 2px rgba(46,40,35,0.08);
                "></div>
                <p style="text-align:center; font-size:0.8rem; margin-top:6px; color:{THEME['ink_soft']};">
                    {sw['name']}<br><code>{sw['hex']}</code>
                </p>
                """,
                unsafe_allow_html=True,
            )


def render_score_bar(pct: float, verdict: str, caption: str = "") -> None:
    """A small colored progress bar + verdict label, used throughout the
    My Closet section."""
    color = (
        THEME["good"] if pct >= 75 else
        THEME["ok"] if pct >= 55 else
        THEME["caution"] if pct >= 35 else
        THEME["avoid"]
    )
    st.markdown(
        f"""
        <div style="margin: 4px 0 2px 0;">
            <div style="background:{THEME['accent_soft']}; border-radius:8px; height:12px; width:100%;">
                <div style="background:{color}; width:{pct}%; height:12px; border-radius:8px;"></div>
            </div>
            <p style="margin:4px 0 0 0; font-size:0.85rem; color:{THEME['ink_soft']};">
                <b style="color:{THEME['ink']};">{pct:.0f}%</b> &middot; {verdict}{(' &middot; ' + caption) if caption else ''}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def swatch_dot(hex_code: str, size: int = 16) -> str:
    return (
        f'<span style="display:inline-block; width:{size}px; height:{size}px; '
        f'border-radius:5px; background:{hex_code}; border:1px solid rgba(0,0,0,0.2); '
        f'vertical-align:middle;"></span>'
    )


# ---------------------------------------------------------------------------
# Sidebar: top-level mode switch
# ---------------------------------------------------------------------------
inject_custom_css()

if LOGO_PATH.exists():
    with st.sidebar:
        render_safe_image(LOGO_PATH)
else:
    st.sidebar.title("🎨 MyHue Recommender")

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
    st.sidebar.subheader("Camera correction")
    apply_awb = st.sidebar.checkbox(
        "Auto white balance (AWB)", value=True,
        help=(
            "Estimates the camera's color cast from the background around "
            "your face (not from your face itself) and corrects for it "
            "before sampling skin color -- helps even out results between "
            "a well-calibrated camera and a cheap one with a strong tint. "
            "It's a best-effort heuristic, not a perfect fix -- turn it "
            "off to compare against the raw, uncorrected reading."
        ),
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

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛍️ Belanja di Shopee")
    enable_shopee = st.sidebar.checkbox(
        "Tampilkan Link Belanja Shopee",
        value=True,
        help=(
            "Aktifkan untuk menampilkan tombol belanja langsung ke Shopee berdasarkan "
            "warna palet dan pakaian yang direkomendasikan."
        ),
    )
    target_audience = "Semua"
    if enable_shopee:
        target_audience = st.sidebar.selectbox(
            "Kategori Pengguna",
            ["Semua", "Wanita", "Pria"],
            index=0,
            help="Menyesuaikan kata kunci pencarian pakaian di Shopee",
        )

    if recommender is None:
        st.sidebar.warning(
            "Garment catalog not found. Run:\n\n`python -m src.garment_indexer`\n\n"
            "to build `data/processed/garment_catalog.csv` first."
        )

    render_hero(
        "🎨",
        "Color Analysis & Clothing Recommender",
        "Take a selfie or upload a photo. We'll analyze your skin tone and "
        "undertone from CIELAB / ITA, then recommend real clothing items "
        "whose dominant color best matches your ideal palette (CIEDE2000).",
        key="hero_a",
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
            result = analyzer.analyze(image_bgr, draw_debug=True, apply_awb=apply_awb)
        except NoFaceDetectedError as exc:
            st.error(str(exc))
            return

    col_img, col_metrics = st.columns([1, 1])
    with col_img:
        st.subheader("Detected face & sampling regions")
        annotated_rgb = cv2.cvtColor(result.annotated_bgr, cv2.COLOR_BGR2RGB)
        render_safe_image(annotated_rgb, caption="Green circles mark the cheek regions sampled for skin color.")

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

        if result.awb_applied and result.awb_method != "none":
            method_label = {
                "face_excluded_gray_world": "background-based (face excluded)",
                "shades_of_gray": "whole-image fallback",
            }.get(result.awb_method, result.awb_method)
            st.caption(
                f"🎛️ Auto white balance applied — {method_label}. "
                f"Channel gains (B,G,R): {result.awb_gains_bgr}. "
                "Turn this off in the sidebar to compare against the raw reading."
            )
        elif not result.awb_applied:
            st.caption("🎛️ Auto white balance is turned off — showing the raw, uncorrected reading.")

    # Persist for reuse in the My Closet tab.
    st.session_state.skin_result = result

    st.markdown("---")
    st.header("Your recommended color palette")
    palette = palette_engine.get_palette(result.skin_tone, result.undertone, body_proportion, torso_length)
    st.session_state.active_palette = palette

    st.subheader("✅ Recommended clothing colors")
    render_swatch_row(palette["recommended_swatches"])

    if enable_shopee and palette.get("recommended_swatches"):
        with st.expander("🛍️ Cari Baju Berdasarkan Palet Ini di Shopee"):
            st.caption("Pilih salah satu warna rekomendasi untuk mencari produk pakaian senada di Shopee Indonesia.")
            sw_names = list(dict.fromkeys([s["name"].title() for s in palette["recommended_swatches"]]))
            c_sh1, c_sh2, c_sh3 = st.columns([2, 2, 1.5])
            with c_sh1:
                chosen_color = st.selectbox("Pilih Warna", sw_names, key="shopee_color_picker")
            with c_sh2:
                chosen_item = st.selectbox(
                    "Jenis Pakaian",
                    ["Semua Pakaian", "Kemeja / Baju", "Celana", "Jaket / Outer", "Rok", "Dress"],
                    key="shopee_item_picker",
                )
            with c_sh3:
                st.write("")
                st.write("")
                cat_kw = {
                    "Semua Pakaian": "baju pakaian",
                    "Kemeja / Baju": "kemeja baju",
                    "Celana": "celana",
                    "Jaket / Outer": "jaket outer",
                    "Rok": "rok",
                    "Dress": "dress terusan",
                }.get(chosen_item, "baju")
                st.markdown(
                    shopee_button_html(
                        f"{cat_kw} {chosen_color}",
                        label="Buka di Shopee",
                        target_audience=target_audience,
                    ),
                    unsafe_allow_html=True,
                )

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

    # ------------------------------------------------------------------
    # Downloadable PDF Report Card
    # ------------------------------------------------------------------
    with st.container(key="pdf_export_card"):
        col_pdf_desc, col_pdf_act = st.columns([2.8, 1.2])
        with col_pdf_desc:
            st.markdown("### 📄 Unduh Laporan Analisis Warna (PDF)")
            st.caption(
                "Dapatkan berkas ringkasan resmi MyHue dalam format PDF siap cetak. "
                "Berisi hasil pengukuran skin tone, undertone, palet warna terbaik, "
                "warna yang dihindari, serta panduan bahan & perhiasan yang serasi."
            )
        with col_pdf_act:
            pdf_buffer = build_pdf_report(
                skin_result=result,
                palette=palette,
                logo_path=LOGO_PATH if LOGO_PATH.exists() else None,
            )
            st.download_button(
                label="📥 Unduh PDF Laporan",
                data=pdf_buffer.getvalue(),
                file_name=f"MyHue_Color_Report_{result.skin_tone}_{result.undertone}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

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
            indexed = list(enumerate(ranked.iterrows()))
            rows = [indexed[i : i + n_cols] for i in range(0, len(indexed), n_cols)]
            for row_chunk in rows:
                cols = st.columns(n_cols)
                for col, (idx, (_, item)) in zip(cols, row_chunk):
                    with col, st.container(key=f"catalog_card_{idx}"):
                        crop_path = item.get("crop_path", "")
                        if crop_path:
                            render_safe_image(crop_path, max_height=200)
                        source_tag = " · your closet" if item.get("source") == "user_upload" else ""
                        st.markdown(
                            f"""
                            <div style="display:flex; align-items:center; gap:6px;">
                                {swatch_dot(item['dominant_hex'])}
                                <b>{item['category'].title()}</b>
                            </div>
                            <p style="font-size:0.8rem; margin:2px 0 0 0; color:{THEME['ink_soft']};">
                                ΔE00 match: {item['match_distance']:.2f} ·
                                <code>{item['dominant_hex']}</code>{source_tag}
                            </p>
                            """,
                            unsafe_allow_html=True,
                        )
                        if enable_shopee:
                            cat_name = item['category']
                            closest_swatch = (
                                min(
                                    palette["recommended_swatches"],
                                    key=lambda s: abs(s["lab"][0] - item["lab_L"])
                                    + abs(s["lab"][1] - item["lab_a"])
                                    + abs(s["lab"][2] - item["lab_b"]),
                                )["name"]
                                if palette.get("recommended_swatches")
                                else ""
                            )
                            st.markdown(
                                shopee_button_html(
                                    f"{cat_name} {closest_swatch}",
                                    label="Beli di Shopee",
                                    target_audience=target_audience,
                                ),
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
    render_hero(
        "👕",
        "My Closet",
        "Upload photos of your own clothes. Each item is scored against "
        "your color palette, pairs are scored for outfit harmony (e.g. "
        "does this shirt work with these jeans?), and every valid "
        "combination across your uploads is ranked for you.",
        key="hero_b",
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

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛍️ Belanja di Shopee")
    enable_shopee_closet = st.sidebar.checkbox(
        "Tampilkan Link Belanja Shopee",
        value=True,
        key="shopee_closet_toggle",
        help="Tampilkan tombol pencarian pakaian di Shopee.",
    )
    target_audience_closet = "Semua"
    if enable_shopee_closet:
        target_audience_closet = st.sidebar.selectbox(
            "Kategori Pengguna",
            ["Semua", "Wanita", "Pria"],
            index=0,
            key="shopee_closet_audience",
            help="Menyesuaikan kata kunci pencarian pakaian di Shopee",
        )

    palette = palette_engine.get_palette(skin_tone, undertone)
    with st.container(key="profile_box"):
        cols_pb1, cols_pb2 = st.columns([3, 1.2])
        with cols_pb1:
            st.markdown(
                f"Scoring against: **{skin_tone} skin / {undertone} undertone** palette."
            )
        with cols_pb2:
            res_obj = st.session_state.skin_result or {
                "skin_tone": skin_tone,
                "undertone": undertone,
                "ita": 0.0,
                "lab": (0, 0, 0),
                "hex": "#C89D7C",
                "rgb": (200, 157, 124),
            }
            pdf_bytes = build_pdf_report(
                skin_result=res_obj,
                palette=palette,
                logo_path=LOGO_PATH if LOGO_PATH.exists() else None,
            ).getvalue()
            st.download_button(
                label="📄 Unduh PDF Profil",
                data=pdf_bytes,
                file_name=f"MyHue_Color_Report_{skin_tone}_{undertone}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

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
                render_safe_image(cropped_rgb, caption="Cropped preview")

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
    indexed = list(enumerate(closet_data))
    rows = [indexed[i : i + n_cols] for i in range(0, len(indexed), n_cols)]
    for row_chunk in rows:
        cols = st.columns(n_cols)
        for col, (idx, entry) in zip(cols, row_chunk):
            item = entry["item"]
            with col, st.container(key=f"item_card_{idx}"):
                crop_rgb = cv2.cvtColor(entry["crop_bgr"], cv2.COLOR_BGR2RGB)
                render_safe_image(crop_rgb)
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
                    if enable_shopee_closet:
                        rec_name = item.personal.closest_recommended["name"]
                        st.markdown(
                            shopee_button_html(
                                f"{item.category} {rec_name}",
                                label=f"Cari {rec_name.title()} di Shopee",
                                target_audience=target_audience_closet,
                            ),
                            unsafe_allow_html=True,
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
        for i, combo in enumerate(combos):
            piece_labels = " + ".join(f"{it.label} ({it.category})" for it in combo.items)
            with st.container(key=f"combo_card_{i}"):
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
