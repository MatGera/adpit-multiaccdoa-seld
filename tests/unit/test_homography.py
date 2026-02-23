"""Unit tests for homography mapping."""

from __future__ import annotations

import numpy as np
import pytest

from vision_fusion.homography import HomographyMapper
from vision_fusion.config import Settings


class TestHomographyMapper:
    def setup_method(self):
        self.mapper = HomographyMapper(Settings())

    def test_no_homography_returns_none(self):
        result = self.mapper.pixel_to_bim("nonexistent", np.array([100, 200]))
        assert result is None

    def test_identity_homography(self):
        self.mapper.set_homography("cam-001", np.eye(3))
        result = self.mapper.pixel_to_bim("cam-001", np.array([5.0, 10.0]))
        assert result is not None
        np.testing.assert_allclose(result, (5.0, 10.0), atol=1e-10)

    def test_scaling_homography(self):
        # 2x scale
        H = np.diag([2.0, 2.0, 1.0])
        self.mapper.set_homography("cam-002", H)
        result = self.mapper.pixel_to_bim("cam-002", np.array([10.0, 20.0]))
        assert result is not None
        np.testing.assert_allclose(result, (20.0, 40.0), atol=1e-10)

    def test_has_homography(self):
        assert not self.mapper.has_homography("cam-001")
        self.mapper.set_homography("cam-001", np.eye(3))
        assert self.mapper.has_homography("cam-001")
