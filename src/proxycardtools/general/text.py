"""Render a white proxy notice inside the final card boundary before upscaling.

TODO:
    I used Codex to make this file GitHub ready.  It appears to work but should go through a human
    review.
"""

import os
from math import isfinite
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from proxycardtools.general.custom_types import (
    CardProcessingOptions,
    ImageBGRF32,
    verify_card_image,
    verify_card_processing_options,
)


def add_proxy_text(
    image: ImageBGRF32,
    options: CardProcessingOptions,
    *,
    font_path: str | Path | None = None,
    font_size_pt: float = 3.0,
    bottom_margin_mm: float = 0.5,
) -> ImageBGRF32:
    """Centre ``PROXY CARD - NOT FOR SALE`` above the bottom trim edge at 600 DPI.

    The bottommost visible glyph is ``bottom_margin_mm`` above the trim edge. Bleed is
    excluded from the card area; spacing rounds to the nearest pixel at 600 DPI. The canvas
    must match the dimensions produced by :func:`~proxycardtools.inpaint.lama.outpaint_card_lama`.
    Only the text mask is rasterized to bytes; source artwork keeps its float32 precision.

    :param image: Outpainted normalized BGR float32 canvas, before 2x upscaling.
    :param options: Card dimensions, orientation, and bleed used for outpainting.
    :param font_path: Times New Roman TTF file. Defaults to ``PROXY_CARD_FONT`` or Windows'
        ``Fonts/times.ttf``. Other platforms must supply the font; no substitute is used.
    :param font_size_pt: Font size in typographic points, defaulting to 3 pt.
    :param bottom_margin_mm: Gap between visible lettering and the bottom card edge.
    :returns: Independent canvas with antialiased white text.
    :raises ValueError: Invalid geometry, spacing, font size, or a label that does not fit.
    :raises OSError: The requested font cannot be loaded.
    """
    verify_card_image(image)
    verify_card_processing_options(options, validate_trim=False)
    if not isfinite(font_size_pt) or font_size_pt <= 0:
        raise ValueError("font_size_pt must be finite and positive.")
    if not isfinite(bottom_margin_mm) or bottom_margin_mm < 0:
        raise ValueError("bottom_margin_mm must be finite and nonnegative.")
    height, width = image.shape[:2]
    landscape = options.orientation == "landscape" or (
        options.orientation == "auto" and width > height
    )
    short, long = options.card_short_side_mm, options.card_long_side_mm
    card_height, card_width = (short, long) if landscape else (long, short)
    pixels_per_mm = 600 / 25.4
    expected = tuple(
        round((length + 2 * options.bleed_mm) * pixels_per_mm)
        for length in (card_height, card_width)
    )
    if (height, width) != expected:
        raise ValueError("Text requires the 600 DPI canvas dimensions used for outpainting.")
    path = Path(
        font_path if font_path is not None else os.environ.get(
            "PROXY_CARD_FONT", str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/times.ttf")
        )
    ).expanduser()
    font = ImageFont.truetype(str(path), size=max(1, round(font_size_pt * 600 / 72)))
    label = "PROXY CARD - NOT FOR SALE"
    left, top, right, bottom = font.getbbox(label)
    text_width, text_height = right - left, bottom - top
    inset = round((options.bleed_mm + bottom_margin_mm) * pixels_per_mm)
    if text_width > width - 2 * inset or text_height > height - 2 * inset:
        raise ValueError("Proxy text does not fit inside the card margins.")
    # Use the glyph bounding box rather than the baseline to keep the visible bottom exact.
    glyphs = Image.new("L", (text_width, text_height))
    ImageDraw.Draw(glyphs).text((-left, -top), label, font=font, fill=255)
    visible = glyphs.getbbox()
    if visible is None:
        raise ValueError("The selected font did not render the proxy label.")
    glyphs = glyphs.crop(visible)
    alpha = Image.new("L", (width, height))
    alpha.paste(glyphs, ((width - glyphs.width) // 2, height - inset - glyphs.height))
    coverage = np.asarray(alpha, dtype=np.float32)[:, :, None] / np.float32(255)
    return np.ascontiguousarray(image + (np.float32(1) - image) * coverage)
