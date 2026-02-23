"""Roto-translation matrix utilities.

Manages the transformation between device-local coordinate frames
and the BIM world coordinate frame.
"""

from __future__ import annotations

import numpy as np

import structlog

logger = structlog.get_logger(__name__)


class RotoTranslationMatrix:
    """4×4 homogeneous transformation matrix for coordinate frame mapping.

    Converts a DOA vector from device-local coordinates to BIM world
    coordinates using the device's calibrated pose.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """Initialize from a 4×4 homogeneous transformation matrix.

        Args:
            matrix: 4×4 numpy array [R|t; 0 0 0 1]
        """
        assert matrix.shape == (4, 4), f"Expected 4×4 matrix, got {matrix.shape}"
        self._matrix = matrix.astype(np.float64)

    @classmethod
    def from_rotation_translation(
        cls,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> RotoTranslationMatrix:
        """Create from separate rotation matrix and translation vector.

        Args:
            rotation: 3×3 rotation matrix.
            translation: 3-element translation vector.
        """
        mat = np.eye(4)
        mat[:3, :3] = rotation
        mat[:3, 3] = translation
        return cls(mat)

    @classmethod
    def from_euler_xyz(
        cls,
        roll: float,
        pitch: float,
        yaw: float,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> RotoTranslationMatrix:
        """Create from Euler angles (X-Y-Z convention, radians) + translation.

        Args:
            roll: Rotation around X axis (radians).
            pitch: Rotation around Y axis (radians).
            yaw: Rotation around Z axis (radians).
            translation: (tx, ty, tz) translation.
        """
        cx, sx = np.cos(roll), np.sin(roll)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cz, sz = np.cos(yaw), np.sin(yaw)

        # R = Rz * Ry * Rx
        r = np.array([
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy,     cy * sx,                 cy * cx               ],
        ])

        return cls.from_rotation_translation(r, np.array(translation))

    @classmethod
    def identity(cls) -> RotoTranslationMatrix:
        """Return the identity transformation (no rotation, no translation)."""
        return cls(np.eye(4))

    @property
    def rotation(self) -> np.ndarray:
        """3×3 rotation component."""
        return self._matrix[:3, :3].copy()

    @property
    def translation(self) -> np.ndarray:
        """3-element translation vector."""
        return self._matrix[:3, 3].copy()

    @property
    def matrix(self) -> np.ndarray:
        """Full 4×4 matrix (read-only copy)."""
        return self._matrix.copy()

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a 3D point from local to world coordinates.

        Args:
            point: 3-element array [x, y, z].

        Returns:
            Transformed 3-element array.
        """
        p_homo = np.append(point, 1.0)
        return (self._matrix @ p_homo)[:3]

    def transform_direction(self, direction: np.ndarray) -> np.ndarray:
        """Transform a direction vector (rotation only, no translation).

        Args:
            direction: 3-element unit vector [x, y, z].

        Returns:
            Rotated 3-element vector (normalized).
        """
        rotated = self._matrix[:3, :3] @ direction
        norm = np.linalg.norm(rotated)
        if norm > 1e-10:
            rotated /= norm
        return rotated

    def inverse(self) -> RotoTranslationMatrix:
        """Compute the inverse transformation."""
        return RotoTranslationMatrix(np.linalg.inv(self._matrix))

    def compose(self, other: RotoTranslationMatrix) -> RotoTranslationMatrix:
        """Compose with another transformation: self * other."""
        return RotoTranslationMatrix(self._matrix @ other._matrix)

    def to_flat_list(self) -> list[float]:
        """Serialize to a flat 16-element list (row-major)."""
        return self._matrix.flatten().tolist()

    @classmethod
    def from_flat_list(cls, values: list[float]) -> RotoTranslationMatrix:
        """Deserialize from a flat 16-element list."""
        assert len(values) == 16, f"Expected 16 values, got {len(values)}"
        return cls(np.array(values).reshape(4, 4))
