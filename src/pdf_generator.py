"""
src/pdf_generator.py
====================
Generates a downloadable, high-quality PDF report for MyHue Personal Color
and Fashion Styling analysis using ReportLab.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas:
    """Two-pass canvas to draw 'Page X of Y' footers and headers."""

    def __init__(self, *args, **kwargs):
        self._saved_page_states = []

    # ReportLab SimpleDocTemplate calls this callback with canvasmaker
    @classmethod
    def create_canvas(cls, *args, **kwargs):
        from reportlab.pdfgen import canvas

        class _CustomCanvas(canvas.Canvas):
            def __init__(self, *c_args, **c_kwargs):
                super().__init__(*c_args, **c_kwargs)
                self.pages = []

            def showPage(self):
                self.pages.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                num_pages = len(self.pages)
                for page in self.pages:
                    self.__dict__.update(page)
                    self.draw_page_decorations(num_pages)
                    super().showPage()
                super().save()

            def draw_page_decorations(self, total_pages: int):
                self.saveState()
                self.setFont("Helvetica", 8)
                self.setFillColor(colors.HexColor("#7E7267"))

                # Running footer
                footer_text = f"MyHue • Personal Color & Fashion Recommendation  |  Page {self._pageNumber} of {total_pages}"
                self.drawRightString(A4[0] - 36, 24, footer_text)
                self.drawString(36, 24, "Confidential & Personalized Style Report")

                # Running top subtle line on pages > 1
                if self._pageNumber > 1:
                    self.setStrokeColor(colors.HexColor("#EAE3DC"))
                    self.setLineWidth(0.5)
                    self.line(36, A4[1] - 30, A4[0] - 36, A4[1] - 30)
                    self.drawString(36, A4[1] - 25, "MyHue Analysis Report")

                self.restoreState()

        return _CustomCanvas(*args, **kwargs)


def _safe_hex(color_hex: str, default: str = "#CCCCCC") -> colors.HexColor:
    try:
        if not color_hex.startswith("#"):
            color_hex = f"#{color_hex}"
        return colors.HexColor(color_hex)
    except Exception:
        return colors.HexColor(default)


def _create_swatches_table(
    swatches: List[Dict[str, Any]],
    cols_per_row: int = 5,
    max_swatches: int = 15,
) -> Table:
    """Builds a grid of color chips with color block and hex label."""
    if not swatches:
        empty_data = [[Paragraph("<i>No swatches specified</i>", ParagraphStyle("em", fontSize=9))]]
        return Table(empty_data, colWidths=[520])

    display_swatches = swatches[:max_swatches]
    rows = []
    tstyles = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]

    chunk_size = cols_per_row
    chunks = [display_swatches[i : i + chunk_size] for i in range(0, len(display_swatches), chunk_size)]

    table_data = []
    row_idx = 0

    for chunk in chunks:
        # Row 1: colored blocks
        color_row = []
        # Row 2: labels (name + hex)
        label_row = []

        for c_idx, sw in enumerate(chunk):
            c_hex = sw.get("hex", "#CCCCCC")
            c_name = sw.get("name", "Color").title()

            # Empty block whose cell background will be colored
            color_row.append("")
            # Label below
            label_text = f"<b>{c_name}</b><br/><font size=6.5 color='#6E6259'>{c_hex}</font>"
            p = Paragraph(
                label_text,
                ParagraphStyle("sw_label", fontName="Helvetica", fontSize=7.5, leading=9, alignment=1),
            )
            label_row.append(p)

            # Set background color and border for the color box cell
            cell_pos = (c_idx, row_idx)
            tstyles.append(("BACKGROUND", cell_pos, cell_pos, _safe_hex(c_hex)))
            tstyles.append(("BOX", cell_pos, cell_pos, 0.5, colors.HexColor("#D0C7BF")))

        # If last row is shorter, pad with empty cells
        if len(chunk) < chunk_size:
            for _ in range(chunk_size - len(chunk)):
                color_row.append("")
                label_row.append("")

        table_data.append(color_row)
        table_data.append(label_row)

        row_idx += 2

    # Calculate column widths to fit A4 printable width (approx 520pt)
    col_w = 520.0 / chunk_size
    # Alternating heights: 30pt for color block, 26pt for label
    row_heights = []
    for _ in chunks:
        row_heights.extend([28, 24])

    t = Table(table_data, colWidths=[col_w] * chunk_size, rowHeights=row_heights)
    t.setStyle(TableStyle(tstyles))
    return t


def build_pdf_report(
    skin_result: Any,
    palette: Dict[str, Any],
    logo_path: Optional[str | Path] = None,
    client_name: str = "Client",
) -> io.BytesIO:
    """Generates a complete PDF personal color report.

    Parameters:
    - skin_result: SkinAnalysisResult instance or compatible object/dict.
    - palette: Dictionary returned by PaletteEngine.get_palette().
    - logo_path: Optional path to logo image (PNG).
    - client_name: Optional name for personalization.

    Returns:
    - io.BytesIO containing the generated PDF binary.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Base typography styles
    color_primary = colors.HexColor("#8B4020")      # Warm terracotta
    color_dark = colors.HexColor("#2E2823")         # Deep ink
    color_soft = colors.HexColor("#6B6055")         # Soft ink
    color_paper = colors.HexColor("#FBF7F2")        # Neutral card bg
    color_border = colors.HexColor("#EAE3DC")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=color_primary,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=color_soft,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=color_dark,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=color_dark,
    )
    meta_val_style = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        textColor=color_primary,
        alignment=1,
    )
    meta_lbl_style = ParagraphStyle(
        "MetaLbl",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9,
        textColor=color_soft,
        alignment=1,
    )

    story = []

    # ------------------------------------------------------------------
    # 1. Header with Logo and Meta
    # ------------------------------------------------------------------
    logo_img = None
    if logo_path and Path(logo_path).exists():
        try:
            logo_img = Image(str(logo_path), width=150, height=38)
        except Exception:
            logo_img = None

    date_str = datetime.now().strftime("%B %d, %Y")
    header_text = [
        Paragraph("<b>MYHUE PERSONAL COLOR REPORT</b>", title_style),
        Paragraph(f"Personal Color, Undertone & Fashion Guide  •  Issued on {date_str}", subtitle_style),
    ]

    if logo_img:
        hdr_table = Table([[header_text, logo_img]], colWidths=[370, 150])
        hdr_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ])
        )
        story.append(hdr_table)
    else:
        story.extend(header_text)

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=color_border, spaceBefore=0, spaceAfter=8))

    # ------------------------------------------------------------------
    # 2. Skin Analysis Profile Box
    # ------------------------------------------------------------------
    skin_tone = getattr(skin_result, "skin_tone", "Unknown") if hasattr(skin_result, "skin_tone") else skin_result.get("skin_tone", "Unknown")
    undertone = getattr(skin_result, "undertone", "Unknown") if hasattr(skin_result, "undertone") else skin_result.get("undertone", "Unknown")
    ita = getattr(skin_result, "ita", 0.0) if hasattr(skin_result, "ita") else skin_result.get("ita", 0.0)
    lab = getattr(skin_result, "lab", (0, 0, 0)) if hasattr(skin_result, "lab") else skin_result.get("lab", (0, 0, 0))
    hex_code = getattr(skin_result, "hex", "#C89D7C") if hasattr(skin_result, "hex") else skin_result.get("hex", "#C89D7C")
    rgb = getattr(skin_result, "rgb", (200, 157, 124)) if hasattr(skin_result, "rgb") else skin_result.get("rgb", (200, 157, 124))

    # Face thumbnail or annotated image if available
    annotated_bgr = getattr(skin_result, "annotated_bgr", None) if hasattr(skin_result, "annotated_bgr") else None
    thumb_flowable = None
    if annotated_bgr is not None and isinstance(annotated_bgr, np.ndarray):
        try:
            # Downsample to small thumbnail
            rgb_arr = annotated_bgr[:, :, ::-1]  # BGR to RGB
            pil_img = PILImage.fromarray(rgb_arr)
            pil_img.thumbnail((120, 120))
            thumb_io = io.BytesIO()
            pil_img.save(thumb_io, format="JPEG", quality=85)
            thumb_io.seek(0)
            thumb_flowable = Image(thumb_io, width=70, height=70)
        except Exception:
            thumb_flowable = None

    profile_card_data = [
        [
            thumb_flowable if thumb_flowable else "",
            [
                Paragraph("<b>Sampled Skin Tone</b>", meta_lbl_style),
                Paragraph(f"{skin_tone}", meta_val_style),
            ],
            [
                Paragraph("<b>Undertone Profile</b>", meta_lbl_style),
                Paragraph(f"{undertone}", meta_val_style),
            ],
            [
                Paragraph("<b>ITA Classification</b>", meta_lbl_style),
                Paragraph(f"{ita:.1f}°", meta_val_style),
            ],
            [
                Paragraph("<b>Measured CIELAB</b>", meta_lbl_style),
                Paragraph(f"L*{lab[0]:.0f} a*{lab[1]:.0f} b*{lab[2]:.0f}", meta_val_style),
            ],
            [
                Paragraph("<b>Skin Color Hex</b>", meta_lbl_style),
                Paragraph(f"<code>{hex_code}</code>", meta_val_style),
            ],
        ]
    ]

    widths = [75, 89, 89, 89, 89, 89]
    profile_table = Table(profile_card_data, colWidths=widths)
    profile_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), color_paper),
            ("BOX", (0, 0), (-1, -1), 1, color_border),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(profile_table)
    story.append(Spacer(1, 10))

    # ------------------------------------------------------------------
    # 3. Recommended Color Palette
    # ------------------------------------------------------------------
    rec_swatches = palette.get("recommended_swatches", [])
    story.append(Paragraph("<b>1. Your Best Clothing Colors (Recommended Palette)</b>", h1_style))
    story.append(
        Paragraph(
            "These shades harmonize with your skin tone and enhance your natural glow based on CIEDE2000 colorimetry.",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 4))
    story.append(_create_swatches_table(rec_swatches, cols_per_row=6, max_swatches=18))
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # 4. Colors to Avoid
    # ------------------------------------------------------------------
    avoid_swatches = palette.get("avoid_swatches", [])
    story.append(Paragraph("<b>2. Colors to Wear with Caution (Avoid Palette)</b>", h1_style))
    story.append(
        Paragraph(
            "These shades may wash out your complexion or clash with your underlying hue temperature.",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 4))
    story.append(_create_swatches_table(avoid_swatches, cols_per_row=6, max_swatches=12))
    story.append(Spacer(1, 10))

    # ------------------------------------------------------------------
    # 5. Full Styling & Fashion Guidance
    # ------------------------------------------------------------------
    style = palette.get("style", {})
    story.append(Paragraph("<b>3. Fashion & Styling Guidelines</b>", h1_style))

    style_rows = [
        [
            Paragraph("<b>Recommended Fitting:</b>", body_style),
            Paragraph(", ".join(style.get("fitting_style", [])) or "Standard classic fit", body_style),
            Paragraph("<b>Flattering Jewelry:</b>", body_style),
            Paragraph(", ".join(style.get("jewelry_metal", [])) or "Warm metals (Gold)", body_style),
        ],
        [
            Paragraph("<b>Best Fabrics & Nature:</b>", body_style),
            Paragraph(", ".join(style.get("fabric_nature", [])) or "Natural, textured fabrics", body_style),
            Paragraph("<b>Footwear Choices:</b>", body_style),
            Paragraph(", ".join(style.get("shoes", [])) or "Classic leather & neutral tones", body_style),
        ],
        [
            Paragraph("<b>Recommended Patterns:</b>", body_style),
            Paragraph(", ".join(style.get("patterns", [])) or "Subtle checks, solid basics", body_style),
            Paragraph("<b>Fabrics / Materials:</b>", body_style),
            Paragraph(", ".join(style.get("materials", [])) or "Cotton, linen, silk blends", body_style),
        ],
        [
            Paragraph("<b>Do Emphasize:</b>", body_style),
            Paragraph(", ".join(style.get("do_exaggerate", [])) or "Clean lines & cohesive accents", body_style),
            Paragraph("<b>Don't Exaggerate:</b>", body_style),
            Paragraph(", ".join(style.get("dont_exaggerate", [])) or "Overly fluorescent or harsh neon", body_style),
        ],
    ]

    style_table = Table(style_rows, colWidths=[120, 140, 120, 140])
    style_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), color_paper),
            ("BOX", (0, 0), (-1, -1), 0.5, color_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, color_border),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(style_table)
    story.append(Spacer(1, 10))

    # ------------------------------------------------------------------
    # 6. Quick Mix & Match Advice (My Closet Harmony)
    # ------------------------------------------------------------------
    guide_p = Paragraph(
        "<b>💡 Outfit Harmony Tip:</b> When pairing clothes in your closet, balance one statement color "
        "from your recommended palette with neutral bases (cream, navy, charcoal, or khaki). "
        "Analogous colors (adjacent on the color wheel) create effortless sophistication, while "
        "avoiding high-contrast clashing tones in the upper body near your face.",
        ParagraphStyle("Note", parent=body_style, fontSize=8, leading=11, textColor=color_soft),
    )
    note_box = Table([[guide_p]], colWidths=[520])
    note_box.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3EBE3")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DFCFC3")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(note_box)

    # Build PDF with custom canvas for page numbering
    doc.build(story, canvasmaker=NumberedCanvas.create_canvas)
    buffer.seek(0)
    return buffer
