"""Locate an aligned card and mask its worn edges at 600 DPI.

Boundary detection assumes one approximately axis-aligned card, filling most of the scan, with a
reasonably uniform scanner surround. Colour alone cannot distinguish a card border from an identical
background; tightly cropped scans therefore use the image boundary when no supported transition is
found.

TODO:
    I used Codex to make this file GitHub ready.  It appears to work but should go through a human
    review.
"""

import numpy as np

from proxycardtools.general.custom_types import (
    ImageBGRF32,
    CardProcessingOptions,
    KeepMask,
    verify_card_image,
    verify_card_processing_options,
)

_PIXELS_PER_MM = 600 / 25.4


def _side_start(strip: ImageBGRF32) -> int:
    """Find the first supported transition from a uniform exterior strip.

    Rows run inward from one canvas edge. Only the outer 15 percent is searched, and the caller
    excludes corners. Requiring agreement across scan lines avoids mistaking lettering for the
    physical edge. The returned index is on the artwork side of the transition, so blur is
    discarded.
    """
    limit = max(1, int(strip.shape[0] * 0.15))
    exterior = np.median(strip[0], axis=0)
    # A textured outer line is evidence of an already cropped scan, not a
    # scanner background. Do not search inward for a decorative frame.
    initial_distance = np.max(np.abs(strip[0] - exterior), axis=1)
    if float(np.mean(initial_distance <= 0.04)) < 0.9:
        return 0
    distances = np.max(np.abs(strip[: limit + 2] - exterior), axis=2)
    support = np.mean(distances > 0.04, axis=1)
    for index in range(1, min(limit + 1, len(support) - 1)):
        if support[index] >= 0.85 and support[index + 1] >= 0.85:
            return index
    return 0


def trim_card_edges(image: ImageBGRF32, options: CardProcessingOptions) -> tuple[ImageBGRF32, KeepMask]:
    """Remove scanner margins and mark the inset rounded card as usable.

    The output spans the detected full card rectangle, preserving its centre for later outpainting.
    Unknown pixels are zeroed; retained pixels are copied exactly, without resizing. Pixel centres
    must lie at least ``edge_trim_mm`` inside the rounded physical boundary. Dimensions are allowed
    to differ by 20 percent from nominal to accommodate scan sizes; nominal dimensions never stretch
    or crop the artwork to a fixed size.

    Detection searches the outer 15 percent of each axis for a transition from a uniform surround.
    An undetectable boundary falls back to the corresponding image edge. Identical card/background
    colours and strong uniform decorative borders are inherently ambiguous under this model. Input
    must already be aligned; this function does not correct residual skew. Orientation and bleed are
    downstream settings and are not applied.

    :param image: Nonempty H x W x 3 finite BGR float32 array in [0, 1], at 600 DPI. The input is
        never modified.
    :param options: Nominal card dimensions, inset, and original corner radius in millimetres.
    :returns: Independent card-centred image and H x W boolean keep mask.
    :raises TypeError: The image is not a float32 NumPy array.
    :raises ValueError: Invalid image or geometry, implausible detected size, or an inset that
        leaves no usable pixels.
    """
    verify_card_image(image)
    verify_card_processing_options(options, validate_outpaint=False)
    short, long = options.card_short_side_mm, options.card_long_side_mm
    inset, radius = options.edge_trim_mm, options.corner_radius_mm

    height, width = image.shape[:2]
    # Central strips avoid rounded corners and rotation-padding wedges.
    vertical = image[:, width // 4 : width - width // 4]
    horizontal = np.transpose(image[height // 4 : height - height // 4], (1, 0, 2))
    top, bottom = _side_start(vertical), height - _side_start(vertical[::-1])
    left, right = _side_start(horizontal), width - _side_start(horizontal[::-1])
    card_height, card_width = bottom - top, right - left
    actual = sorted((card_height, card_width))
    nominal = (short * _PIXELS_PER_MM, long * _PIXELS_PER_MM)
    if any(
        not 0.8 * expected <= measured <= 1.2 * expected
        for measured, expected in zip(actual, nominal, strict=True)
    ):
        raise ValueError("Cannot locate a card of plausible 600 DPI dimensions.")

    inset_px = inset * _PIXELS_PER_MM
    radius_px = radius * _PIXELS_PER_MM
    if 2 * inset_px >= min(card_height, card_width) or 2 * radius_px > min(card_height, card_width):
        raise ValueError("Inset or corner radius is too large for the detected card.")
    # Signed distance to a rounded rectangle: erosion moves the straight
    # sides inward and reduces the radius, rather than retaining the original
    # radius on an already inset rectangle (which over-trims the corners).
    y, x = np.ogrid[:card_height, :card_width]
    dx = np.abs(x + 0.5 - card_width / 2) - (card_width / 2 - radius_px)
    dy = np.abs(y + 0.5 - card_height / 2) - (card_height / 2 - radius_px)
    distance = (
        np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
        + np.minimum(np.maximum(dx, dy), 0)
        - radius_px
    )
    keep_mask = distance <= -inset_px
    if not np.any(keep_mask):
        raise ValueError("Trimming leaves no usable pixels.")
    trimmed = image[top:bottom, left:right].copy()
    trimmed[~keep_mask] = 0
    return trimmed, keep_mask
