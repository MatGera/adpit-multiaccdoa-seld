"""Homography mapping: pixel coordinates → BIM world coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import structlog

if TYPE_CHECKING:
    from vision_fusion.config import Settings

logger = structlog.get_logger(__name__)


class HomographyMapper:
    """Maps camera pixel coordinates to BIM 2D ground-plane coordinates.

    Uses pre-calibrated 3×3 homography matrices stored per camera.
    The mapping assumes a planar ground surface (z ≈ 0).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # camera_id → 3×3 homography matrix
        self._matrices: dict[str, np.ndarray] = {}

    def set_homography(self, camera_id: str, matrix: np.ndarray) -> None:
        """Set the homography matrix for a camera.

        Args:
            camera_id: Camera identifier.
            matrix: 3×3 homography matrix mapping pixel → BIM ground plane.
        """
        assert matrix.shape == (3, 3), f"Expected 3×3, got {matrix.shape}"
        self._matrices[camera_id] = matrix.astype(np.float64)
        logger.info("homography_set", camera_id=camera_id)

    def set_homography_from_points(
        self,
        camera_id: str,
        pixel_points: np.ndarray,
        bim_points: np.ndarray,
    ) -> np.ndarray:
        """Compute homography from corresponding point pairs.

        Args:
            camera_id: Camera identifier.
            pixel_points: Nx2 array of pixel coordinates.
            bim_points: Nx2 array of BIM ground-plane coordinates.

        Returns:
            Computed 3×3 homography matrix.
        """
        import cv2

        assert pixel_points.shape[0] >= 4, "Need at least 4 point pairs"
        assert pixel_points.shape == bim_points.shape

        H, mask = cv2.findHomography(pixel_points, bim_points, cv2.RANSAC, 5.0)

        if H is None:
            raise ValueError("Failed to compute homography")

        self._matrices[camera_id] = H
        inliers = int(mask.sum()) if mask is not None else pixel_points.shape[0]
        logger.info("homography_computed",
                     camera_id=camera_id,
                     inliers=inliers,
                     total_points=pixel_points.shape[0])

        return H

    def pixel_to_bim(
        self,
        camera_id: str,
        pixel_coord: np.ndarray,
    ) -> tuple[float, float] | None:
        """Map a pixel coordinate to BIM ground-plane coordinate.

        Args:
            camera_id: Camera identifier.
            pixel_coord: 2-element array [px, py].

        Returns:
            (bim_x, bim_y) tuple, or None if no homography available.
        """
        H = self._matrices.get(camera_id)
        if H is None:
            return None

        # Homogeneous coordinates
        pt = np.array([pixel_coord[0], pixel_coord[1], 1.0])
        result = H @ pt

        # Normalize
        if abs(result[2]) < 1e-10:
            return None

        bim_x = result[0] / result[2]
        bim_y = result[1] / result[2]

        return (float(bim_x), float(bim_y))

    def has_homography(self, camera_id: str) -> bool:
        """Check if a camera has a configured homography."""
        return camera_id in self._matrices
