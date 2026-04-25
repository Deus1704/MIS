from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def circular_roi_mask(
    shape: Tuple[int, int],
    center_rc: Tuple[float, float],
    radius_px: float,
) -> np.ndarray:
    """Create a circular binary ROI mask."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    r0, c0 = center_rc
    dist2 = (rows - r0) ** 2 + (cols - c0) ** 2
    return dist2 <= radius_px**2


def lesion_mask_from_points(
    shape: Tuple[int, int],
    points_rc: Iterable[Sequence[float]],
    radius_px: float,
) -> np.ndarray:
    """Union of circular ROIs centered on lesion mark points."""
    mask = np.zeros(shape, dtype=bool)
    for point in points_rc:
        if len(point) != 2:
            raise ValueError("Each lesion point must have row and col")
        mask |= circular_roi_mask(shape=shape, center_rc=(float(point[0]), float(point[1])), radius_px=radius_px)
    return mask


def inverse_mask(mask: np.ndarray) -> np.ndarray:
    """Return logical inverse mask."""
    return np.logical_not(mask.astype(bool))
