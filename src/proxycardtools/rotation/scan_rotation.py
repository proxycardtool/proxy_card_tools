"""Estimate small card skew using Hough lines, rectangular contours, and text projection profiles.

All reported angles are corrective OpenCV angles in degrees: positive means counterclockwise
rotation. Methods measure deviations from the nearest image axis, never portrait/landscape
orientation. Confidence values are heuristic evidence scores in [0, 1], not probabilities.

TODO:
    I used Codex to make this file GitHub ready.  It appears to work but should go through a human
    review.
"""

from math import ceil

import cv2
import numpy as np

from proxycardtools.general.custom_types import (
    ImageBGRF32,
    FloatValues,
    ImageGrayU8,
    MethodEstimate,
    RotationEstimate,
    RotationOptions,
    RotationResult,
    verify_card_image,
)


def _axis_angle(dx: float, dy: float) -> float:
    """Map an image-space line slope onto its signed deviation from the nearest axis."""
    return float((np.degrees(np.arctan2(dy, dx)) + 45) % 90 - 45)


def _weighted_median(values: FloatValues, weights: FloatValues) -> float:
    """Locate the weighted median of a nonempty finite vector with positive total weight."""
    order = np.argsort(values)
    index = np.searchsorted(np.cumsum(weights[order]), weights.sum() * 0.5)
    return float(values[order[min(int(index), len(order) - 1)]])


def _hough_estimate(gray: ImageGrayU8, search_limit: float) -> MethodEstimate:
    """Estimate correction from robust, length-weighted agreement among long Hough line segments."""
    height, width = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0.8), 50, 140)
    # Image boundaries are scanner/crop coordinates, not evidence about the card's orientation.
    edges[:4, :] = edges[-4:, :] = 0
    edges[:, :4] = edges[:, -4:] = 0
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=35,
        minLineLength=max(30, min(height, width) * 0.25),
        maxLineGap=12,
    )
    if lines is None:
        return MethodEstimate("hough_lines", None, 0.0, 0, "No long straight segments.")
    angles: list[float] = []
    lengths: list[float] = []
    positions: list[tuple[int, int]] = []
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        dx, dy = float(x1 - x0), float(y1 - y0)
        angle = _axis_angle(dx, dy)
        if abs(angle) <= search_limit:
            angles.append(angle)
            lengths.append(float(np.hypot(dx, dy)))
            vertical = abs(dy) > abs(dx)
            coordinate = (x0 + x1) / 2 / width if vertical else (y0 + y1) / 2 / height
            positions.append((int(vertical), int(coordinate * 12)))
    if len(angles) < 2:
        return MethodEstimate(
            "hough_lines", None, 0.0, len(angles), "Insufficient near-axis segments."
        )
    values = np.asarray(angles)
    segment_lengths = np.asarray(lengths)
    # Longer frame edges carry more angular precision than short fragments of lettering or artwork.
    weights = segment_lengths**2
    # Select a mode before averaging; decorative diagonals should not drag the
    # correction toward them.
    masses = np.array([weights[np.abs(values - value) <= 0.3].sum() for value in values])
    inliers = np.abs(values - values[int(np.argmax(masses))]) <= 0.3
    angle = _weighted_median(values[inliers], weights[inliers])
    spread = float(np.average(np.abs(values[inliers] - angle), weights=weights[inliers]))
    coverage = len({positions[i] for i in np.flatnonzero(inliers)})
    support = int(inliers.sum())
    dominance = float(weights[inliers].sum() / weights.sum())
    confidence = np.sqrt(dominance) * np.exp(-spread / 0.5) * min(1.0, coverage / 3)
    confidence *= min(1.0, float(segment_lengths[inliers].sum()) / (height + width))
    return MethodEstimate(
        "hough_lines",
        angle,
        float(np.clip(confidence, 0, 1)),
        support,
        f"{coverage} spatial bands; dominance={dominance:.2f}; deviation={spread:.3f} deg.",
    )


def _contour_estimate(gray: ImageGrayU8, search_limit: float) -> MethodEstimate:
    """Fit a minimum-area rectangle only to a large, closed, card-shaped contour clear of image
    boundaries."""
    height, width = gray.shape
    _, binary = cv2.threshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    candidates: list[tuple[float, float]] = []
    for mask in (binary, cv2.bitwise_not(binary)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < height * width * 0.4:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
                continue
            rectangle = cv2.minAreaRect(contour)
            rw, rh = rectangle[1]
            if min(rw, rh) <= 0:
                continue
            aspect = min(rw, rh) / max(rw, rh)
            rectangularity = area / (rw * rh)
            if not 0.60 <= aspect <= 0.85 or rectangularity < 0.88:
                continue
            box = cv2.boxPoints(rectangle)
            dx, dy = box[1] - box[0]
            angle = _axis_angle(float(dx), float(dy))
            if abs(angle) > search_limit:
                continue
            confidence = min(1.0, area / (height * width * 0.65))
            confidence *= min(1.0, (rectangularity - 0.85) / 0.12)
            confidence *= float(np.exp(-abs(aspect - 63.5 / 88) / 0.15))
            candidates.append((float(confidence), angle))
    if not candidates:
        return MethodEstimate(
            "card_contour", None, 0.0, 0, "No complete card-shaped contour clear of crop edges."
        )
    confidence, angle = max(candidates)
    return MethodEstimate(
        "card_contour", angle, confidence, 1, "Large closed card-shaped rectangular contour."
    )


def _projection_estimate(gray: ImageGrayU8, search_limit: float) -> MethodEstimate:
    """Find the correction maximizing row-profile sharpness of small dark, text-like connected
    components.

    A coarse-to-fine shear search avoids repeatedly interpolating image pixels. Each component
    centroid casts a linearly split vote to rows after correction. Component count, horizontal
    extent, and peak prominence determine confidence. This is not OCR and can abstain on
    illustration-only cards.
    """
    height, width = gray.shape
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    margin_y, margin_x = max(4, height // 15), max(4, width // 15)
    binary[:margin_y, :] = binary[-margin_y:, :] = 0
    binary[:, :margin_x] = binary[:, -margin_x:] = 0
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    accepted = np.zeros(count, dtype=np.bool_)
    for index in range(1, count):
        _, _, w, h, area = stats[index]
        accepted[index] = 3 <= h <= height * 0.035 and 1 <= w <= width * 0.06 and 5 <= area <= 500
    xs, ys = centroids[accepted].T
    components = int(accepted.sum())
    if components < 20 or np.ptp(xs) < width * 0.25:
        return MethodEstimate(
            "text_projection", None, 0.0, components, "Insufficient distributed text-like features."
        )
    # Subpixel centroids avoid the systematic zero-angle preference of
    # foreground-pixel row projections. Capping component weight prevents a few
    # large artwork blobs from dominating the text alignment estimate.
    vote_weights = np.minimum(stats[accepted, cv2.CC_STAT_AREA], 100).astype(np.float64)
    x = xs.astype(np.float64) - width / 2
    y = ys.astype(np.float64) + width

    def score(angle: float) -> float:
        """Measure squared row-profile differences after a small-angle vertical shear."""
        coordinates = y - np.tan(np.radians(angle)) * x
        lower = coordinates.astype(np.int64)
        fraction = coordinates - lower
        profile = np.asarray(
            np.bincount(lower, weights=(1 - fraction) * vote_weights, minlength=height + 2 * width),
            dtype=np.float64,
        )
        profile += np.bincount(
            lower + 1, weights=fraction * vote_weights, minlength=height + 2 * width
        )
        # Smoothing limits integer-row binning bias, which otherwise creates an
        # artificial peak at zero skew.
        profile = np.convolve(profile, [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16], mode="same")
        return float(np.square(np.diff(profile)).sum())

    coarse = np.linspace(-search_limit, search_limit, ceil(2 * search_limit / 0.1) + 1)
    scores = np.array([score(float(angle)) for angle in coarse])
    best = int(np.argmax(scores))
    if best == 0 or best == len(coarse) - 1:
        return MethodEstimate(
            "text_projection", float(coarse[best]), 0.0, components, "Peak at search boundary."
        )
    fine = np.linspace(coarse[best] - 0.1, coarse[best] + 0.1, 21)
    fine_scores = np.array([score(float(angle)) for angle in fine])
    angle = float(fine[int(np.argmax(fine_scores))])
    peak = float(fine_scores.max())
    baseline = float(np.median(scores))
    outside = scores[np.abs(coarse - angle) > 0.4]
    rival = float(outside.max()) if len(outside) else baseline
    prominence = (peak - baseline) / max(peak, 1e-12)
    separation = (peak - rival) / max(peak, 1e-12)
    confidence = min(1.0, prominence / 0.5) * min(1.0, max(0.0, separation) / 0.12)
    confidence *= min(1.0, components / 80)
    return MethodEstimate(
        "text_projection",
        angle,
        float(np.clip(confidence, 0, 1)),
        components,
        f"Profile prominence={prominence:.2f}; separated-peak margin={separation:.2f}.",
    )


def _combine(methods: tuple[MethodEstimate, ...], options: RotationOptions) -> RotationEstimate:
    """Require mutually agreeing methods and penalize both angular spread and competing evidence."""
    eligible = [
        method
        for method in methods
        if method.angle_deg is not None and method.confidence >= options.min_method_confidence
    ]
    if len(eligible) < 2:
        confidence = max((method.confidence for method in eligible), default=0.0) * 0.5
        return RotationEstimate(
            methods, None, confidence, False, "Fewer than two confident methods."
        )
    groups: list[list[MethodEstimate]] = []
    for method in eligible:
        assert method.angle_deg is not None
        group = [
            other
            for other in eligible
            if other.angle_deg is not None
            and method.angle_deg <= other.angle_deg <= method.angle_deg + options.agreement_deg
        ]
        if len(group) >= 2:
            groups.append(group)
    if not groups:
        return RotationEstimate(methods, None, 0.0, False, "Confident methods disagree.")
    group = max(groups, key=lambda members: sum(member.confidence for member in members))
    values = np.array([method.angle_deg for method in group], dtype=np.float64)
    weights = np.array([method.confidence for method in group])
    angle = float(np.average(values, weights=weights))
    deviation = float(np.sqrt(np.average((values - angle) ** 2, weights=weights)))
    agreement = float(np.exp(-0.5 * (deviation / (options.agreement_deg / 2)) ** 2))
    support = float(weights.sum()) / sum(method.confidence for method in eligible)
    confidence = float(weights.mean()) * agreement * support
    if abs(angle) > options.max_angle_deg:
        return RotationEstimate(
            methods, angle, confidence, False, "Estimated correction exceeds small-angle limit."
        )
    if confidence < options.min_confidence:
        return RotationEstimate(
            methods, angle, confidence, False, "Consensus confidence below threshold."
        )
    if abs(angle) <= options.deadband_deg:
        return RotationEstimate(
            methods, angle, confidence, False, "Already aligned within the deadband."
        )
    return RotationEstimate(
        methods, angle, confidence, True, "Confident methods agree on a small correction."
    )


def estimate_rotation(
    image: ImageBGRF32, *, options: RotationOptions | None = None
) -> RotationEstimate:
    """Measure small scan skew and report per-method and combined confidence without changing
    pixels.

    :param image: Nonempty finite H x W x 3 BGR float32 image in [0, 1], preferably already
        denoised.
    :param options: Detection/decision thresholds; defaults require two agreeing methods and at most
        3 degrees.
    :returns: All three method estimates and the conservative correction decision.
    :raises TypeError: Image type/dtype is unsupported.
    :raises ValueError: Image shape or values are invalid.
    """
    verify_card_image(image)
    settings = options if options is not None else RotationOptions()
    gray = np.asarray(
        cv2.cvtColor(np.rint(image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), dtype=np.uint8
    )
    height, width = gray.shape
    scale = min(1.0, settings.analysis_max_side / max(height, width))
    if min(height, width) * scale < 64:
        methods = tuple(
            MethodEstimate(name, None, 0.0, 0, "Image too small for reliable skew detection.")
            for name in ("hough_lines", "card_contour", "text_projection")
        )
        return RotationEstimate(
            methods, None, 0.0, False, "Image too small for reliable skew detection."
        )
    if scale < 1:
        gray = np.asarray(
            cv2.resize(
                gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
            ),
            dtype=np.uint8,
        )
    # Search beyond the allowed correction so an out-of-range peak is rejected
    # rather than clamped to a limit.
    search_limit = settings.max_angle_deg + 2.0
    methods = (
        _hough_estimate(gray, search_limit),
        _contour_estimate(gray, search_limit),
        _projection_estimate(gray, search_limit),
    )
    return _combine(methods, settings)


def correct_scan_rotation(
    image: ImageBGRF32, *, options: RotationOptions | None = None
) -> RotationResult:
    """Apply a confident small correction once, expanding the canvas instead of clipping source
    content.

    :param image: Finite normalized float32 BGR input, never modified. DPI is preserved by a unit-
        scale transform.
    :param options: Optional confidence, angle-limit, and no-op thresholds.
    :returns: Corrected image, boolean source-support mask, affine transform, and complete decision
        diagnostics. Low-confidence, conflicting, oversized, or already-aligned estimates return an
        exact pixel copy.
    :raises TypeError: Image type/dtype is unsupported.
    :raises ValueError: Image shape or values are invalid.
    """
    estimate = estimate_rotation(image, options=options)
    height, width = image.shape[:2]
    if not estimate.should_rotate:
        return RotationResult(
            image.copy(),
            estimate,
            np.ones((height, width), dtype=np.bool_),
            np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64),
        )
    assert estimate.angle_deg is not None
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), estimate.angle_deg, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_width = ceil(width * cosine + height * sine)
    out_height = ceil(height * cosine + width * sine)
    matrix[0, 2] += (out_width - width) / 2
    matrix[1, 2] += (out_height - height) / 2
    rotated = cv2.warpAffine(
        image,
        matrix,
        (out_width, out_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    # Mark interpolation near a source boundary invalid because cubic sampling
    # needs neighbouring source pixels.
    support = np.ones((height, width), dtype=np.uint8)
    support[:2, :] = support[-2:, :] = 0
    support[:, :2] = support[:, -2:] = 0
    valid = cv2.warpAffine(
        support.astype(np.float32), matrix, (out_width, out_height), flags=cv2.INTER_LINEAR
    )
    return RotationResult(
        np.clip(rotated, 0, 1).astype(np.float32),
        estimate,
        valid >= 0.999999,
        np.asarray(matrix, dtype=np.float64),
    )
