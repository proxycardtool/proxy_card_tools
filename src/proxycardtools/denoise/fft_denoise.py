"""This module suppresses periodic noise using FFT notch filtering.

This module is used to suppress periodic noise using Fast Fourier Transform (FFT) notch filtering.
It was specifically created to remove the dots found on Lord of the Rings LCG card scans, but it may
eventually see use with other card types.

This module should be used by calling the ``fft_denoise`` function.  This function accepts both
``ImageBGRF32`` and ``ImageBGRU8`` data types; see the ``custom_types.py`` module for detailed
definitions of these data types. ``ImageBGRF32`` is recommended.

TODO:
    I used Codex to make this file GitHub ready.  It appears to work but should go through a human
    review.

TODO:
    This module is currently hard-coded for Lord of the Rings LCG card scans.  Associated variables
    should eventually be moved to parameters and the functions made more general.
"""

from typing import overload

import cv2
import numpy as np

from proxycardtools.general.custom_types import (
    ImageBGRF32,
    ImageBGRU8,
    ImageLabF32,
    ImageLabU8,
    verify_bgr_image,
    verify_lab_luminance,
)


@overload
def fft_denoise(img: ImageBGRF32) -> ImageBGRF32: ...

@overload
def fft_denoise(img: ImageBGRU8) -> ImageBGRU8: ...

def fft_denoise(img: ImageBGRF32 | ImageBGRU8) -> ImageBGRF32 | ImageBGRU8:
    """Remove prominent periodic luminance patterns while retaining the Lab colour channels.

    :param img: Nonempty H x W x 3 BGR image, either finite float32 in [0, 1] or uint8 in [0, 255].
        The input is not modified. Floating-point values in [0, 255] must be normalized by the
        caller.
    :returns: A new image with the same shape, dtype, and range convention. Lab processing uses
        float32; only uint8 output is rounded to byte values. Colour conversion can introduce small
        round-trip error.
    :raises TypeError: The input is not a NumPy array or has an unsupported dtype.
    :raises ValueError: The shape is invalid, or floating-point values are nonfinite or outside [0,
        1].
    """
    verify_bgr_image(img)
    img_float = img if img.dtype == np.float32 else img.astype(np.float32) / np.float32(255.0)

    lab_image = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)
    luminance, a_channel, b_channel = cv2.split(lab_image)
    filtered_luminance = _fft_denoise_channel(np.asarray(luminance, dtype=np.float32))
    filtered_lab = cv2.merge([filtered_luminance, a_channel, b_channel])
    filtered_bgr = np.clip(cv2.cvtColor(filtered_lab, cv2.COLOR_Lab2BGR), 0.0, 1.0).astype(
        np.float32
    )
    if img.dtype == np.float32:
        return filtered_bgr
    return np.rint(filtered_bgr * 255.0).astype(np.uint8)


@overload
def _fft_denoise_channel(channel: ImageLabF32) -> ImageLabF32: ...


@overload
def _fft_denoise_channel(channel: ImageLabU8) -> ImageLabU8: ...


def _fft_denoise_channel(channel: ImageLabF32 | ImageLabU8) -> ImageLabF32 | ImageLabU8:
    """Apply the original broad spectral mask to a Lab luminance plane.

    Detect peaks above a median spectral background, soften their suppression, and protect a
    10-pixel band on either side of each frequency axis. Axis-aligned periodic noise is
    intentionally left untouched. Conjugate symmetry is enforced for both odd and even image
    dimensions to retain a real-valued inverse.

    :param channel: Nonempty H x W float32 Lab L plane in [0, 100], or legacy uint8 L in [0, 255].
        The input is never modified. Float luminance is scaled by 2.55 before spectral detection to
        retain the selected experiment's log-magnitude threshold behavior without quantizing image
        samples.
    :returns: Same shape and dtype. Float output retains fractional values and any filtering
        overshoot; clipping is deferred until BGR reconstruction. Legacy byte output is rounded and
        clipped to [0, 255].
    :raises TypeError: The input is not a NumPy array with dtype float32 or uint8.
    :raises ValueError: Shape is invalid or float luminance is nonfinite or outside [0, 100].
    """
    verify_lab_luminance(channel)
    fft_input = channel * np.float32(2.55) if channel.dtype == np.float32 else channel
    spectrum = np.fft.fftshift(np.fft.fft2(fft_input))
    log_magnitude = np.log1p(np.abs(spectrum))
    magnitude_u8 = np.empty(channel.shape, dtype=np.uint8)
    cv2.normalize(log_magnitude, magnitude_u8, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    background = cv2.medianBlur(magnitude_u8, 31)
    residual = cv2.subtract(magnitude_u8, background)
    threshold = max(float(np.percentile(residual, 99.9)), 10.0)
    _, peaks = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)
    peaks = cv2.morphologyEx(peaks, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8))
    mask = (peaks > 0).astype(np.float32)

    height, width = channel.shape
    # In shifted coordinates the partner of index i is (2 * centre - i) modulo the axis length.
    # A simple flip is off by one for even dimensions and can introduce an
    # imaginary inverse component.
    partner_rows = (2 * (height // 2) - np.arange(height)) % height
    partner_columns = (2 * (width // 2) - np.arange(width)) % width
    mask = np.maximum(mask, mask[np.ix_(partner_rows, partner_columns)])

    protected_rows = np.abs(np.arange(height) - height // 2) <= 10
    protected_columns = np.abs(np.arange(width) - width // 2) <= 10
    mask[protected_rows, :] = 0.0
    mask[:, protected_columns] = 0.0

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    mask = np.asarray(cv2.dilate(mask, dilation_kernel), dtype=np.float32)
    soft_mask = np.asarray(cv2.GaussianBlur(mask, (0, 0), sigmaX=3.0, sigmaY=3.0), dtype=np.float32)

    # Border handling during dilation and blur can break symmetry. Restore it on the final filter.
    soft_mask = np.maximum(soft_mask, soft_mask[np.ix_(partner_rows, partner_columns)])
    # Expansion must not leak suppression back into the protected axes,
    # especially the DC (mean) component.
    soft_mask[protected_rows, :] = 0.0
    soft_mask[:, protected_columns] = 0.0
    maximum = float(soft_mask.max())
    if maximum > 0.0:
        soft_mask /= maximum
    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    filtered_spectrum = spectrum * (1.0 - soft_mask)
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)).real
    if channel.dtype == np.float32:
        return np.asarray(filtered.astype(np.float32) / np.float32(2.55), dtype=np.float32)
    return np.asarray(np.rint(np.clip(filtered, 0.0, 255.0)), dtype=np.uint8)
