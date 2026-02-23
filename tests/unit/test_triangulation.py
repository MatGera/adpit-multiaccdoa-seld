"""Unit tests for multi-sensor triangulation."""

from __future__ import annotations

import numpy as np
import pytest

from spatial_engine.triangulation import Triangulator, Observation


class TestTriangulator:
    def setup_method(self):
        self.tri = Triangulator(min_observations=2, max_residual=5.0)

    def test_insufficient_observations(self):
        obs = [
            Observation(
                device_id="d1",
                origin=np.array([0, 0, 0]),
                direction=np.array([1, 0, 0]),
            )
        ]
        result = self.tri.triangulate(obs)
        assert result is None

    def test_two_perpendicular_rays(self):
        """Two perpendicular rays should converge to a known point."""
        target = np.array([5.0, 5.0, 0.0])

        obs1_origin = np.array([0.0, 5.0, 0.0])
        obs1_dir = (target - obs1_origin)
        obs1_dir = obs1_dir / np.linalg.norm(obs1_dir)

        obs2_origin = np.array([5.0, 0.0, 0.0])
        obs2_dir = (target - obs2_origin)
        obs2_dir = obs2_dir / np.linalg.norm(obs2_dir)

        obs = [
            Observation(device_id="d1", origin=obs1_origin, direction=obs1_dir),
            Observation(device_id="d2", origin=obs2_origin, direction=obs2_dir),
        ]

        result = self.tri.triangulate(obs)
        assert result is not None
        assert result.converged
        np.testing.assert_allclose(result.estimated_point, target, atol=0.5)

    def test_three_device_triangulation(self):
        """Three devices with rays converging on the same point."""
        target = np.array([3.0, 4.0, 2.0])

        origins = [
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 0.0]),
            np.array([5.0, 10.0, 0.0]),
        ]

        obs = []
        for i, origin in enumerate(origins):
            d = target - origin
            d = d / np.linalg.norm(d)
            obs.append(Observation(device_id=f"d{i}", origin=origin, direction=d))

        result = self.tri.triangulate(obs)
        assert result is not None
        assert result.converged
        assert len(result.contributing_devices) == 3
        np.testing.assert_allclose(result.estimated_point, target, atol=0.5)
