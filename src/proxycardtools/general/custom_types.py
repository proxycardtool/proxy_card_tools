"""Shared data contracts and non-mutating runtime verification.

Array aliases encode dtype, not shape, range, or channel order. Call the corresponding
verification function at processing boundaries. BGR channel order is a caller contract;
it cannot be inferred from pixel values. Image aliases name their color space and dtype;
their valid ranges still depend on the processing stage.

TODO:
    I used Codex to make this file GitHub ready.  It appears to work but should go through a human
    review.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type ImageBGRF32 = NDArray[np.float32]
type ImageBGRU8 = NDArray[np.uint8]
type ImageLabF32 = NDArray[np.float32]
type ImageLabU8 = NDArray[np.uint8]
type ImageGrayU8 = NDArray[np.uint8]
type KeepMask = NDArray[np.bool_]
type FloatValues = NDArray[np.float64]
type Orientation = Literal["auto", "portrait", "landscape"]


@dataclass(frozen=True)
class CardProcessingOptions:
    """Physical geometry and orientation for the processing stages.

    All lengths are in millimetres. Bleed is added to each of the four sides. Card dimensions
    describe the target trim size, not an exact input size; historical variations in scan dimensions
    must be accommodated. ``auto`` preserves the scan's portrait or landscape orientation. Input
    resolution is fixed at 600 DPI; output resolution is 1200 DPI. Processing boundaries validate
    dimensions and margins before use.
    """

    bleed_mm: float = 3.0
    orientation: Orientation = "auto"
    card_short_side_mm: float = 63.5
    card_long_side_mm: float = 88.0
    edge_trim_mm: float = 0.5
    corner_radius_mm: float = 3.5


@dataclass(frozen=True)
class RotationOptions:
    """Conservative decision thresholds, in degrees except dimensionless confidence scores.

    ``max_angle_deg`` is limited to 5 degrees; larger orientation changes are outside this API's
    purpose. Two eligible methods must agree within ``agreement_deg``. ``deadband_deg`` avoids
    unnecessary resampling. Detection is downsampled to at most ``analysis_max_side`` pixels;
    correction uses original pixels.
    """

    max_angle_deg: float = 3.0
    min_confidence: float = 0.55
    min_method_confidence: float = 0.35
    agreement_deg: float = 0.35
    deadband_deg: float = 0.05
    analysis_max_side: int = 1200

    def __post_init__(self) -> None:
        """Reject invalid thresholds before any image analysis or interpolation."""
        values = (
            self.max_angle_deg,
            self.min_confidence,
            self.min_method_confidence,
            self.agreement_deg,
            self.deadband_deg,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Rotation thresholds must be finite.")
        if not 0 < self.max_angle_deg <= 5:
            raise ValueError("max_angle_deg must be in (0, 5].")
        if not 0 < self.min_method_confidence <= self.min_confidence <= 1:
            raise ValueError("Confidence thresholds must satisfy 0 < method <= combined <= 1.")
        if not 0 < self.agreement_deg <= 1 or not 0 <= self.deadband_deg < self.max_angle_deg:
            raise ValueError(
                "Require agreement_deg in (0, 1] and deadband_deg in [0, max_angle_deg)."
            )
        if isinstance(self.analysis_max_side, bool) or not isinstance(self.analysis_max_side, int):
            raise TypeError("analysis_max_side must be an integer.")
        if not 256 <= self.analysis_max_side <= 2400:
            raise ValueError("analysis_max_side must be between 256 and 2400.")


@dataclass(frozen=True)
class MethodEstimate:
    """One method's corrective angle and evidence, including explicit abstention.

    ``angle_deg=None`` and zero confidence indicate no usable estimate. ``support`` counts
    supporting lines, contours, or text-like components; it is not comparable between methods.
    ``detail`` explains rejection or the measured evidence. Nonzero angles can exceed the allowed
    correction limit.
    """

    method: str
    angle_deg: float | None
    confidence: float
    support: int
    detail: str


@dataclass(frozen=True)
class RotationEstimate:
    """Consensus decision retaining every method's diagnostics.

    ``angle_deg`` is the estimated corrective angle, even when below the deadband or over the size
    limit. It is None when no eligible agreeing group exists. Only ``should_rotate=True`` authorizes
    interpolation. Confidence is a heuristic agreement/evidence score, not a success probability.
    """

    methods: tuple[MethodEstimate, ...]
    angle_deg: float | None
    confidence: float
    should_rotate: bool
    reason: str


@dataclass(frozen=True)
class RotationResult:
    """Corrected normalized BGR image, decision, source support, and input-to-output affine
    transform.

    ``valid_mask`` is H x W bool: False marks padding or pixels near the resampled source boundary.
    The canvas expands to retain all source content. New pixels are black placeholders, not card
    artwork; use the mask during future trimming/outpainting. No-op results are copies with an all-
    True mask.
    """

    image: ImageBGRF32
    estimate: RotationEstimate
    valid_mask: KeepMask
    transform: FloatValues


def verify_bgr_image(image: object, *, allow_uint8: bool = True) -> None:
    """Verify a nonempty H x W x 3 BGR image without coercing or copying it.

    :param image: Float32 pixels in [0, 1], or legacy uint8 pixels in [0, 255].
    :param allow_uint8: Whether the caller supports the legacy byte representation.
    :raises TypeError: The input is not an array with a supported dtype.
    :raises ValueError: Shape, finiteness, or normalized range is invalid.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")
    if image.dtype != np.float32 and not (allow_uint8 and image.dtype == np.uint8):
        raise TypeError("image must have dtype float32" + (" or uint8." if allow_uint8 else "."))
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must have nonempty H x W x 3 BGR shape.")
    if image.dtype == np.float32 and (
        not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1)
    ):
        raise ValueError("image values must be finite and in [0, 1].")


def verify_card_image(image: object) -> None:
    """Verify the normalized float32 BGR contract used by card processing stages.

    :raises TypeError: The input is not a float32 NumPy array.
    :raises ValueError: Shape, finiteness, or normalized range is invalid.
    """
    verify_bgr_image(image, allow_uint8=False)


def verify_lab_luminance(channel: object) -> None:
    """Verify a nonempty H x W Lab L plane, preserving fractional luminance.

    :param channel: Float32 luminance in [0, 100], or uint8 luminance in [0, 255].
    :raises TypeError: The input is not a float32 or uint8 NumPy array.
    :raises ValueError: Shape, finiteness, or luminance range is invalid.
    """
    if not isinstance(channel, np.ndarray) or channel.dtype not in (np.float32, np.uint8):
        raise TypeError("channel must be a float32 or uint8 NumPy array.")
    if channel.ndim != 2 or channel.size == 0:
        raise ValueError("channel must be a nonempty two-dimensional plane.")
    if channel.dtype == np.float32 and (
        not np.isfinite(channel).all() or np.any(channel < 0) or np.any(channel > 100)
    ):
        raise ValueError("float32 Lab luminance must be finite and in [0, 100].")


def verify_keep_mask(
    mask: object, *, shape: tuple[int, ...] | None = None, require_artwork: bool = False
) -> None:
    """Verify a boolean H x W mask; True means known, usable source pixels.

    :param shape: Expected image height and width, when alignment must be checked.
    :param require_artwork: Reject all-False masks when reconstruction needs source artwork.
    :raises TypeError: The input is not a boolean NumPy array.
    :raises ValueError: Empty/invalid shape, mismatched geometry, or missing required artwork.
    """
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("keep_mask must be a boolean NumPy array.")
    if mask.ndim != 2 or mask.size == 0 or (shape is not None and mask.shape != shape):
        raise ValueError("keep_mask must have nonempty H x W shape matching the image.")
    if require_artwork and not mask.any():
        raise ValueError("keep_mask must retain some artwork.")


def verify_card_processing_options(
    options: object, *, validate_trim: bool = True, validate_outpaint: bool = True
) -> None:
    """Verify physical geometry before image processing or canvas allocation.

    Stage-specific checks preserve standalone usage: outpainting does not use trim radius,
    and trimming does not use bleed or orientation. Full pipelines check both groups early.
    Pixel allocation limits and detected-card geometry remain checks at their usage sites.

    :param options: Card geometry in millimetres.
    :param validate_trim: Check edge inset and corner radius against nominal dimensions.
    :param validate_outpaint: Check bleed and orientation.
    :raises TypeError: Options or numeric field types are invalid; booleans are not lengths.
    :raises ValueError: Nonfinite values, invalid dimensions, margins, or orientation.
    """
    if not isinstance(options, CardProcessingOptions):
        raise TypeError("options must be CardProcessingOptions.")
    short, long = options.card_short_side_mm, options.card_long_side_mm
    values = [short, long]
    if validate_trim:
        values.extend((options.edge_trim_mm, options.corner_radius_mm))
    if validate_outpaint:
        values.append(options.bleed_mm)
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.floating)):
            raise TypeError("Card geometry must contain real numbers, not booleans.")
        if not isfinite(value):
            raise ValueError("Card geometry must be finite.")
    if not 0 < short <= long:
        raise ValueError("Require 0 < short <= long.")
    if validate_trim:
        if not 0 <= options.edge_trim_mm < short / 2:
            raise ValueError("edge_trim_mm must be nonnegative and leave usable artwork.")
        if not 0 <= options.corner_radius_mm <= short / 2:
            raise ValueError("corner_radius_mm must be in [0, short/2].")
    if validate_outpaint:
        if options.bleed_mm < 0:
            raise ValueError("bleed_mm must be nonnegative.")
        if options.orientation not in ("auto", "portrait", "landscape"):
            raise ValueError("orientation must be auto, portrait, or landscape.")
