"""Prepare 600 DPI card scans for printing with 10 mm bleed at 1200 DPI.

Run ``python example/example.py`` using the project's environment.

TODO:
    I used Codex to make this file GitHub ready.  It appears to work but should go through a human
    review.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import tomllib
from typing import cast

from proxycardtools.general.custom_types import CardProcessingOptions, ImageBGRF32
from proxycardtools.denoise.fft_denoise import fft_denoise
from proxycardtools.general.color import adjust_gamma
from proxycardtools.general.text import add_proxy_text
from proxycardtools.general.trim_card_edges import trim_card_edges
from proxycardtools.inpaint.lama import outpaint_card_lama, remove_footer_text_lama
from proxycardtools.rotation.scan_rotation import correct_scan_rotation
from proxycardtools.upscale.upscale import upscale_card_2x


def main() -> None:
    """Process the scans in ``example/input`` and save PNGs in ``example/output``."""
    example_dir = Path(__file__).resolve().parent
    project_dir = example_dir.parent
    input_dir = example_dir / "input"
    output_dir = example_dir / "output"

    with (project_dir / "config" / "settings.toml").open("rb") as config_file:
        settings = tomllib.load(config_file)

    # Configured paths are relative to the project root.
    lama_model = project_dir / cast(str, settings["inpainting"]["model_path_rel"])
    upscale_model = project_dir / cast(str, settings["upscaling"]["model_path_rel"])

    output_dir.mkdir(exist_ok=True)

    options = CardProcessingOptions()

    for input_path in input_dir.iterdir():
        print(f"Processing {input_path.name}", flush=True)

        # The processing functions use BGR float32 pixels in [0, 1].
        with Image.open(input_path) as scan:
            rgb = np.asarray(scan.convert("RGB"), dtype=np.float32) / np.float32(255.0)
        image: ImageBGRF32 = np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.float32)

        denoised = fft_denoise(image)
        aligned = correct_scan_rotation(denoised).image
        print_adjusted = adjust_gamma(aligned)
        cleaned = remove_footer_text_lama(print_adjusted, model_path=lama_model)
        trimmed, keep_mask = trim_card_edges(cleaned, options)
        outpainted = outpaint_card_lama(trimmed, keep_mask, options, model_path=lama_model)
        labelled = add_proxy_text(outpainted, options)
        upscaled = upscale_card_2x(labelled, model_path=upscale_model)

        # Convert back to RGB bytes and record the final print resolution.
        output_rgb = np.rint(upscaled[:, :, ::-1] * 255.0).astype(np.uint8)
        output_path = output_dir / f"{input_path.stem}.png"
        Image.fromarray(output_rgb).save(output_path, dpi=(1200, 1200))
        print(f"Saved {output_path.name}", flush=True)


if __name__ == "__main__":
    main()
