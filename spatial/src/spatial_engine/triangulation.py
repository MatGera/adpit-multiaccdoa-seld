"""Multi-sensor triangulation: estimate 3D sound source position.

Given DOA observations from multiple devices at known positions,
find the 3D point that minimizes the angular error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

import structlog

from spatial_engine.roto_translation import RotoTranslationMatrix

logger = structlog.get_logger(__name__)


@dataclass
class Observation:
    """A single DOA observation from a device."""
    device_id: str
    origin: np.ndarray       # 3D position of device in world coords
    direction: np.ndarray    # Unit DOA vector in world coords
    confidence: float = 1.0  # Weight for this observation


@dataclass
class TriangulationResult:
    """Result of multi-sensor triangulation."""
    estimated_point: np.ndarray
    residual_error: float
    contributing_devices: list[str]
    converged: bool


class Triangulator:
    """Estimates 3D source position from multiple DOA observations.

    Uses least-squares optimization to find the point that minimizes
    the sum of squared angular distances from all observation rays.
    """

    def __init__(
        self,
        min_observations: int = 2,
        max_residual: float = 5.0,
    ) -> None:
        self._min_observations = min_observations
        self._max_residual = max_residual

    def triangulate(self, observations: list[Observation]) -> TriangulationResult | None:
        """Estimate 3D source position from DOA observations.

        Args:
            observations: List of DOA observations from different devices.

        Returns:
            TriangulationResult or None if insufficient observations.
        """
        if len(observations) < self._min_observations:
            logger.warning("insufficient_observations",
                           count=len(observations),
                           required=self._min_observations)
            return None

        origins = np.array([o.origin for o in observations])
        directions = np.array([o.direction for o in observations])
        weights = np.array([o.confidence for o in observations])

        # Initial guess: weighted midpoint of all device positions
        x0 = np.average(origins, axis=0, weights=weights)

        # Offset initial guess along average direction
        avg_dir = np.average(directions, axis=0, weights=weights)
        avg_dir_norm = np.linalg.norm(avg_dir)
        if avg_dir_norm > 1e-10:
            avg_dir /= avg_dir_norm
            x0 += avg_dir * 5.0  # offset 5m along average direction

        def residuals(point: np.ndarray) -> np.ndarray:
            """Compute weighted angular residuals for all observations."""
            res = []
            for i, obs in enumerate(observations):
                # Vector from device to estimated point
                to_point = point - obs.origin
                dist = np.linalg.norm(to_point)

                if dist < 1e-10:
                    res.append(weights[i] * 1.0)
                    continue

                to_point_normalized = to_point / dist

                # Angular error: 1 - cos(angle)
                cos_angle = np.clip(np.dot(to_point_normalized, obs.direction), -1.0, 1.0)
                angular_error = 1.0 - cos_angle

                res.append(weights[i] * angular_error)

            return np.array(res)

        result = least_squares(residuals, x0, method="lm")

        residual_error = float(np.sqrt(np.mean(result.fun ** 2)))
        converged = result.success and residual_error < self._max_residual

        device_ids = [o.device_id for o in observations]

        logger.info("triangulation_result",
                     point=result.x.tolist(),
                     residual=residual_error,
                     converged=converged,
                     devices=device_ids)

        return TriangulationResult(
            estimated_point=result.x,
            residual_error=residual_error,
            contributing_devices=device_ids,
            converged=converged,
        )

    def prepare_observation(
        self,
        device_id: str,
        doa_local: tuple[float, float, float],
        transform: RotoTranslationMatrix,
        confidence: float = 1.0,
    ) -> Observation:
        """Create an Observation from device-local DOA and its transform.

        Args:
            device_id: Device identifier.
            doa_local: DOA vector in device-local frame.
            transform: Device-to-world roto-translation.
            confidence: Observation weight.

        Returns:
            Observation in world coordinates.
        """
        origin = transform.transform_point(np.zeros(3))
        direction = transform.transform_direction(np.array(doa_local))

        return Observation(
            device_id=device_id,
            origin=origin,
            direction=direction,
            confidence=confidence,
        )
