"""Unit tests for roto-translation matrix utilities."""

from __future__ import annotations

import numpy as np
import pytest

from spatial_engine.roto_translation import RotoTranslationMatrix


class TestRotoTranslationMatrix:
    def test_identity(self):
        T = RotoTranslationMatrix.identity()
        point = np.array([1.0, 2.0, 3.0])
        result = T.transform_point(point)
        np.testing.assert_allclose(result, point)

    def test_pure_translation(self):
        T = RotoTranslationMatrix.from_rotation_translation(
            np.eye(3), np.array([10.0, 20.0, 30.0])
        )
        point = np.array([1.0, 1.0, 1.0])
        result = T.transform_point(point)
        np.testing.assert_allclose(result, [11.0, 21.0, 31.0])

    def test_direction_not_translated(self):
        T = RotoTranslationMatrix.from_rotation_translation(
            np.eye(3), np.array([100.0, 200.0, 300.0])
        )
        direction = np.array([0.0, 0.0, 1.0])
        result = T.transform_direction(direction)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0])

    def test_from_euler_xyz(self):
        T = RotoTranslationMatrix.from_euler_xyz(0, 0, 0, (5.0, 5.0, 5.0))
        point = np.array([0.0, 0.0, 0.0])
        result = T.transform_point(point)
        np.testing.assert_allclose(result, [5.0, 5.0, 5.0])

    def test_90_degree_rotation(self):
        # 90° rotation around Z axis
        T = RotoTranslationMatrix.from_euler_xyz(0, 0, np.pi / 2)
        direction = np.array([1.0, 0.0, 0.0])
        result = T.transform_direction(direction)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)

    def test_inverse(self):
        T = RotoTranslationMatrix.from_euler_xyz(0.5, -0.3, 1.2, (10, -5, 3))
        T_inv = T.inverse()
        composed = T.compose(T_inv)
        np.testing.assert_allclose(composed.matrix, np.eye(4), atol=1e-10)

    def test_serialization_roundtrip(self):
        T = RotoTranslationMatrix.from_euler_xyz(0.1, 0.2, 0.3, (1, 2, 3))
        flat = T.to_flat_list()
        assert len(flat) == 16
        T2 = RotoTranslationMatrix.from_flat_list(flat)
        np.testing.assert_allclose(T.matrix, T2.matrix)
