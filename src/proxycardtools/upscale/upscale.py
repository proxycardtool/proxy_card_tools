"""Load local Spandrel models for float32, RGB-aware, exactly 2x card enlargement.

TODO:
    I used Codex to make this file GitHub ready.  It appears to work but should go through a human
    review.
"""

import os
from pathlib import Path

import numpy as np
import torch

# Spandrel ships annotations but no py.typed marker; model/output checks below guard its boundary.
from spandrel import ImageModelDescriptor, ModelLoader, ModelTiling  # type: ignore[import-untyped]

from proxycardtools.general.custom_types import ImageBGRF32, verify_card_image


def upscale_card_2x(
    image: ImageBGRF32,
    *,
    model_path: str | Path | None = None,
    device: str = "auto",
    tile_size: int = 256,
    tile_overlap: int = 32,
) -> ImageBGRF32:
    """Enlarge a 600 DPI card canvas to 1200 DPI with a local neural model.

    Each input pixel becomes a 2 x 2 output region, preserving physical dimensions and orientation.
    No file DPI metadata is written. Inference uses float32 throughout without byte quantization.
    Model weights are loaded once per call; no downloads or persistent device caches are created.

    Tiles include additional context on every available side. Only their central predictions are
    retained, preventing artificial tile boundaries from reaching the output for local models whose
    receptive radius fits inside the overlap. The default 32-pixel context covers the supplied SPAN
    model. Other architectures may need larger context or single-pass inference (tile_size=0).
    Memory or device failures propagate; no fallback changes resolution or model precision.

    :param image: Nonempty finite H x W x 3 BGR float32 image in [0, 1]; never modified.
    :param model_path: Trusted local Spandrel checkpoint. Defaults to UPSCALE_MODEL, or
        ``external_data/models/2x_ModernSpanimationV1.pth`` relative to the working directory.
    :param device: ``auto`` selects CUDA when available, otherwise CPU; explicit PyTorch device
        strings such as ``cpu`` or ``cuda:0`` are also accepted.
    :param tile_size: Maximum central tile side in input pixels; zero disables tiling.
    :param tile_overlap: Context in input pixels on each side of a tile; nonnegative integer.
    :returns: Independent contiguous float32 BGR array with exactly twice both input dimensions.
    :raises TypeError: Image dtype, tile parameter types, or model descriptor type is unsupported.
    :raises ValueError: Invalid image, tile parameters, model scale/channels, or prediction values.
    :raises FileNotFoundError: The local model file is missing.
    :raises RuntimeError: Loading or inference fails, including insufficient device memory.
    """
    verify_card_image(image)
    for name, value in (("tile_size", tile_size), ("tile_overlap", tile_overlap)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative.")
    path = Path(
        model_path
        if model_path is not None
        else os.environ.get("UPSCALE_MODEL", "external_data/models/2x_ModernSpanimationV1.pth")
    ).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Upscaling model not found: {path}. Set UPSCALE_MODEL or model_path."
        )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(device)
    model = ModelLoader().load_from_file(path)
    if not isinstance(model, ImageModelDescriptor):
        raise TypeError("Upscaling requires an image-to-image model.")
    if model.scale != 2 or model.input_channels != 3 or model.output_channels != 3:
        raise ValueError("Upscaling requires an exactly 2x model with three input/output channels.")
    height, width = image.shape[:2]
    step = tile_size or max(height, width)
    if step < max(height, width) and model.tiling != ModelTiling.SUPPORTED:
        raise ValueError("This model does not support reliable tiling; use tile_size=0.")
    model.to(device=target_device, dtype=torch.float32).eval()
    result = np.empty((2 * height, 2 * width, 3), dtype=np.float32)
    with torch.inference_mode():
        for top in range(0, height, step):
            bottom = min(top + step, height)
            for left in range(0, width, step):
                right = min(left + step, width)
                y0, y1 = max(0, top - tile_overlap), min(height, bottom + tile_overlap)
                x0, x1 = max(0, left - tile_overlap), min(width, right + tile_overlap)
                # BGR reversal creates negative strides; make an owned contiguous tensor buffer.
                rgb = np.ascontiguousarray(image[y0:y1, x0:x1, ::-1].transpose(2, 0, 1)[None])
                pixels = torch.from_numpy(rgb).to(target_device)
                prediction = model(pixels)
                if prediction.shape != (1, 3, 2 * (y1 - y0), 2 * (x1 - x0)):
                    raise ValueError("Upscaler returned an unexpected output shape.")
                if not prediction.is_floating_point() or not bool(torch.isfinite(prediction).all()):
                    raise ValueError("Upscaler returned nonfinite or non-floating pixels.")
                # Keep central predictions only: averaging context would mix less reliable tile
                # boundary predictions into good ones and can soften fine print.
                core = prediction[
                    0, :, 2 * (top - y0) : 2 * (bottom - y0), 2 * (left - x0) : 2 * (right - x0)
                ]
                output: ImageBGRF32 = (
                    core.permute(1, 2, 0).to(device="cpu", dtype=torch.float32).numpy()
                )
                result[2 * top : 2 * bottom, 2 * left : 2 * right] = np.clip(
                    output[:, :, ::-1], 0, 1
                )
    return result
