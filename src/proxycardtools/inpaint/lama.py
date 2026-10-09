"""Run local Big-LaMa safetensors weights on a centred 600 DPI card canvas.

The model must accept RGB and hole-mask tensors and return an RGB tensor, all in NCHW layout.
Spandrel reconstructs the LaMa generator for the weights from michaelgold/big-lama. No model
downloads occur during inference. Set LAMA_MODEL or pass model_path to select a local weights file.

TODO:
    I used Codex to make this file GitHub ready.  It appears to work but should go through a human
    review.
"""

import os
from math import isfinite
from pathlib import Path

import numpy as np
import cv2
import torch
from spandrel import MaskedImageModelDescriptor, ModelLoader

from proxycardtools.general.custom_types import (
    ImageBGRF32,
    CardProcessingOptions,
    KeepMask,
    verify_card_image,
    verify_card_processing_options,
    verify_keep_mask,
)


def _prepare_canvas(
    image: ImageBGRF32, keep_mask: KeepMask, options: CardProcessingOptions
) -> tuple[ImageBGRF32, KeepMask]:
    """Validate input and translate pixels without scaling or discarding known artwork.

    Total physical dimensions are rounded once at 600 DPI. When source and destination parity
    differ, the extra pixel goes on the bottom/right, giving at most half a pixel centre error.
    Unknown source margins may extend beyond the destination; known pixels may not.
    """
    verify_card_image(image)
    verify_keep_mask(keep_mask, shape=image.shape[:2], require_artwork=True)
    verify_card_processing_options(options, validate_trim=False)
    short, long, bleed = options.card_short_side_mm, options.card_long_side_mm, options.bleed_mm
    landscape = options.orientation == "landscape" or (
        options.orientation == "auto" and image.shape[1] > image.shape[0]
    )
    dimensions = (short, long) if landscape else (long, short)
    physical = tuple((value + 2 * bleed) * (600 / 25.4) for value in dimensions)
    if not all(isfinite(value) and 1 <= value <= 32768 for value in physical):
        raise ValueError("Target dimensions must be between 1 and 32768 pixels at 600 DPI.")
    height, width = (round(value) for value in physical)
    if height * width > 64_000_000:
        raise ValueError("Target canvas exceeds the 64 megapixel allocation limit.")
    source_height, source_width = image.shape[:2]
    dy, dx = (height - source_height) // 2, (width - source_width) // 2
    sy, sx = max(0, -dy), max(0, -dx)
    ey, ex = min(source_height, height - dy), min(source_width, width - dx)
    retained = keep_mask[sy:ey, sx:ex]
    if np.count_nonzero(retained) != np.count_nonzero(keep_mask):
        raise ValueError("Target canvas would crop retained artwork; increase dimensions or bleed.")
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.bool_)
    destination = (slice(sy + dy, ey + dy), slice(sx + dx, ex + dx))
    canvas[destination] = image[sy:ey, sx:ex]
    mask[destination] = retained
    canvas[~mask] = 0
    return canvas, mask


def outpaint_card_lama(
    image: ImageBGRF32,
    keep_mask: KeepMask,
    options: CardProcessingOptions,
    *,
    model_path: str | Path | None = None,
    device: str = "auto",
) -> ImageBGRF32:
    """Reconstruct unknown regions and bleed with local Big-LaMa safetensors weights.

    Inference runs at full 600 DPI in float32, without resizing or quantizing artwork. Known pixels
    are restored exactly after prediction. Full-resolution inference can require substantial RAM
    or VRAM; allocation and model errors propagate rather than silently reducing resolution.
    The model is loaded per call and released afterwards, avoiding hidden GPU model caches.

    :param image: Nonempty finite H x W x 3 BGR float32 image in [0, 1]; never modified.
    :param keep_mask: Matching boolean mask, with True identifying pixels to preserve.
    :param options: Target card dimensions, orientation, and per-side bleed in millimetres.
    :param model_path: Big-LaMa safetensors file. Defaults to LAMA_MODEL, or
        ``external_data/models/big-lama.safetensors`` relative to the working directory.
    :param device: ``auto`` selects CUDA when available, otherwise CPU; explicit PyTorch device
        strings such as ``cpu`` and ``cuda:0`` are also accepted.
    :returns: Independent normalized float32 BGR canvas at the requested physical size.
    :raises TypeError: Unsupported image or mask dtype, or a model returning a non-tensor result.
    :raises ValueError: Invalid geometry, input, cropped known pixels, model type, or model output.
    :raises FileNotFoundError: The model file is missing and reconstruction is required.
    :raises RuntimeError: Model loading or inference fails, including insufficient device memory.
    """
    canvas, known = _prepare_canvas(image, keep_mask, options)
    return _inpaint_lama(canvas, known, model_path=model_path, device=device)


def _inpaint_lama(
    canvas: ImageBGRF32,
    known: KeepMask,
    *,
    model_path: str | Path | None,
    device: str,
) -> ImageBGRF32:
    """Infer missing pixels on an existing canvas, restoring known pixels exactly."""
    if known.all():
        return canvas.copy()
    path = Path(
        model_path
        if model_path is not None
        else os.environ.get("LAMA_MODEL", "external_data/models/big-lama.safetensors")
    ).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"LaMa safetensors model not found: {path}. Set LAMA_MODEL or pass model_path."
        )
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"LaMa model must be a .safetensors file, got: {path}")
    target_device = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    height, width = known.shape
    # The reference export expects dimensions divisible by eight. Symmetric padding also works
    # for one-pixel axes, unlike reflection padding. Remove it after inference.
    padding = ((0, (-height) % 8), (0, (-width) % 8))
    rgb = np.pad(canvas[:, :, ::-1], (*padding, (0, 0)), mode="symmetric")
    holes = np.pad((~known).astype(np.float32), padding, mode="symmetric")
    pixels = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])).to(target_device)
    mask = torch.from_numpy(np.ascontiguousarray(holes[None, None])).to(target_device)
    try:
        descriptor = ModelLoader(device=target_device).load_from_file(path)
    except Exception as exc:
        # Normalize parser, state-dict, and architecture detection failures into one public error.
        raise RuntimeError(f"Failed to load LaMa safetensors weights from {path}.") from exc
    if not isinstance(descriptor, MaskedImageModelDescriptor):
        raise ValueError("The safetensors file is not an inpainting model.")
    if descriptor.architecture.id != "LaMa":
        raise ValueError(
            f"Expected LaMa weights, but detected architecture {descriptor.architecture.id}."
        )
    model = descriptor.eval()
    with torch.inference_mode():
        prediction: object = model(pixels, mask)
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("LaMa must return an N x 3 x H x W tensor.")
    if prediction.shape != pixels.shape or not prediction.is_floating_point():
        raise ValueError("LaMa returned an invalid tensor shape or dtype.")
    output: ImageBGRF32 = (
        prediction[0, :, :height, :width]
        .permute(1, 2, 0)
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    if not np.isfinite(output).all():
        raise ValueError("LaMa returned nonfinite pixels.")
    result = np.ascontiguousarray(np.clip(output[:, :, ::-1], 0, 1))
    result[known] = canvas[known]
    return result


def detect_footer_text(image: ImageBGRF32, *, footer_height_mm: float = 2.2) -> KeepMask:
    """Find small white lettering in the bottom strip of an aligned 600 DPI scan.

    Detection uses neutral bright pixels, local contrast, and connected-component size.
    Nearby glyphs must form a horizontal group to reject isolated scan speckles. This is a
    geometric heuristic, not OCR: pale artwork in the footer can be ambiguous. Inspect the
    returned mask for unfamiliar card layouts and adjust ``footer_height_mm`` as needed.

    :param image: Normalized BGR float32 scan; never modified.
    :param footer_height_mm: Search depth measured upward from the scan's bottom edge.
    :returns: Boolean mask matching the image; True marks text-group boxes and their fringe.
    :raises ValueError: Invalid image or nonpositive/nonfinite search depth.
    """
    verify_card_image(image)
    if not isfinite(footer_height_mm) or footer_height_mm <= 0:
        raise ValueError("footer_height_mm must be finite and positive.")
    height, width = image.shape[:2]
    top = max(0, height - round(footer_height_mm * 600 / 25.4))
    strip = image[top:]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    # A top-hat rejects broad pale borders while retaining thin letter strokes.
    contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    neutral = np.max(strip, axis=2) - np.min(strip, axis=2) <= 0.16
    seeds = ((gray >= 0.55) & neutral & (contrast >= 0.12)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    glyphs = np.zeros_like(seeds)
    for index in range(1, count):
        x, y, component_width, component_height, area = (int(v) for v in stats[index])
        if 3 <= component_height <= 32 and component_width <= 40 and area >= 3:
            region = labels[y : y + component_height, x : x + component_width]
            glyphs[y : y + component_height, x : x + component_width][region == index] = 1
    # Joining adjacent strokes supplies line evidence. Complete line boxes also cover dim
    # antialiased strokes and shadows that cannot pass the white-pixel threshold themselves.
    joined = cv2.dilate(glyphs, np.ones((3, 19), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    supported = np.zeros_like(glyphs)
    for index in range(1, count):
        x, y, line_width, line_height, _ = (int(v) for v in stats[index])
        if (
            line_width >= 35 and line_height <= 36
            and x >= round(2 * 600 / 25.4)
            and x + line_width <= width - round(2 * 600 / 25.4)
        ):
            supported[y : y + line_height, x : x + line_width] = 1
    detected = cv2.dilate(supported, np.ones((5, 5), np.uint8))
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[top:] = detected.astype(np.bool_)
    return mask


def remove_footer_text_lama(
    image: ImageBGRF32,
    *,
    footer_height_mm: float = 2.2,
    model_path: str | Path | None = None,
    device: str = "auto",
) -> ImageBGRF32:
    """Detect and erase footer lettering with LaMa before edge trimming.

    Only masked pixels change. A bottom crop with 4 mm of additional context reduces inference
    memory without resizing the scan. No model is loaded when detection finds no text.

    :param image: Aligned, gamma-adjusted BGR float32 scan at 600 DPI; never modified.
    :param footer_height_mm: Footer search depth; see :func:`detect_footer_text`.
    :param model_path: Local LaMa safetensors weights, as in :func:`outpaint_card_lama`.
    :param device: PyTorch device or ``auto``.
    :returns: Independent image at the original size, with detected lettering reconstructed.
    """
    holes = detect_footer_text(image, footer_height_mm=footer_height_mm)
    result = image.copy()
    if not holes.any():
        return result
    first_row = int(np.flatnonzero(holes.any(axis=1))[0])
    top = max(0, first_row - round(4 * 600 / 25.4))
    result[top:] = _inpaint_lama(
        image[top:], ~holes[top:], model_path=model_path, device=device
    )
    return result
