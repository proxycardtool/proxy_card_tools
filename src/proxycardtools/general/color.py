"""Restore scan contrast and adjust tones before home printing.

The default black/white points of 0.02/0.98 restore slightly faded scan endpoints before
a gamma exponent of 0.9 lifts midtones. These are starting points for test prints, not
a calibrated scanner, printer, or paper profile.

TODO:
    I used Codex to make this file GitHub ready.  It appears to work but should go through a human
    review.
"""

from math import isfinite

import numpy as np

from proxycardtools.general.custom_types import ImageBGRF32, verify_card_image


def adjust_gamma(
        image: ImageBGRF32,
        *,
        gamma: float = 0.9,
        black_point: float = 0.02,
        white_point: float = 0.98,
) -> ImageBGRF32:
    """Restore scan endpoints, then apply a gamma curve to each BGR channel.

    Compute ``clip((input - black_point) / (white_point - black_point), 0, 1) ** gamma``
    on encoded scan values, without linearizing them. By default, samples at or below
    0.02 become black and those at or above 0.98 become white; a 0.5 sample becomes
    approximately 0.536. Values outside the endpoint window lose detail through clipping.
    Set the points to 0 and 1 for gamma alone, and also gamma to 1 for an unchanged copy.

    The same endpoint window is used for every channel. Fixed, configurable points avoid
    forcing a dark or pale card's artwork to span the full range based on its histogram,
    or mistaking black rotation padding for the scanned black point. They do not correct
    scanner color casts. Tune the points to known black/white patches or test prints.

    :param image: Nonempty H x W x 3 BGR float32 image with finite values in [0, 1].
        The input is never modified.
    :param gamma: Finite positive exponent applied after endpoint restoration. Values
        below 1 brighten, 1 applies only endpoint restoration, and above 1 darken.
    :param black_point: Scan value mapped to black, in [0, white_point).
    :param white_point: Scan value mapped to white, in (black_point, 1].
    :return: Independent float32 image with unchanged shape and values in [0, 1].
    :raises TypeError: Image dtype or parameter type is unsupported; booleans are rejected.
    :raises ValueError: Invalid image, nonfinite parameters, nonpositive gamma, or invalid points.
    """
    verify_card_image(image)
    for name, value in (("gamma", gamma), ("black_point", black_point), ("white_point", white_point)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a real number, not a boolean.")
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if not 0 <= black_point < white_point <= 1:
        raise ValueError("Require 0 <= black_point < white_point <= 1.")
    if gamma == 1 and black_point == 0 and white_point == 1:
        return image.copy()

    # Clip before dividing so even an extremely narrow endpoint window cannot overflow.
    # Float64 also retains finite extreme gamma values; quantize only the final image.
    restored = np.clip(image.astype(np.float64), black_point, white_point)
    restored -= black_point
    restored /= white_point - black_point
    np.power(restored, gamma, out=restored)
    return restored.astype(np.float32)
