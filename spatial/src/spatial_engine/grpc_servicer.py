"""Spatial Engine gRPC servicer — handles spatial query RPCs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import text

import structlog

from seld_common.db import DatabaseManager
from spatial_engine.ifc_parser import IFCParser
from spatial_engine.raycaster import Raycaster
from spatial_engine.triangulation import Triangulator
from spatial_engine.roto_translation import RotoTranslationMatrix

if TYPE_CHECKING:
    from spatial_engine.config import Settings

logger = structlog.get_logger(__name__)


class SpatialServicer:
    """gRPC servicer for spatial query operations.

    Implements SpatialService RPCs:
    - Raycast: DOA vector → BIM asset hits
    - Triangulate: multiple DOA observations → 3D point
    - GetCalibration: device calibration matrix
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db = DatabaseManager(settings.database_url)
        self._parser = IFCParser(settings.bim_storage_path)
        self._raycaster = Raycaster(max_distance=settings.max_ray_distance)
        self._triangulator = Triangulator(
            min_observations=settings.min_observations,
            max_residual=settings.max_residual_error,
        )
        self._loaded_models: set[str] = set()

    async def _ensure_model_loaded(self, bim_model_id: str) -> bool:
        """Ensure a BIM model is loaded for raycasting."""
        if bim_model_id in self._loaded_models:
            return True

        async with self._db.session() as session:
            result = await session.execute(
                text("SELECT ifc_file_path FROM bim_models WHERE id = :id AND status = 'ready'"),
                {"id": bim_model_id},
            )
            row = result.fetchone()
            if not row:
                logger.warning("bim_model_not_found", model_id=bim_model_id)
                return False

        parsed = self._parser.parse(row.ifc_file_path, bim_model_id)
        assets = [
            {
                "global_id": a.global_id,
                "name": a.name,
                "ifc_type": a.ifc_type,
                "mesh": a.mesh,
            }
            for a in parsed.assets
        ]
        self._raycaster.load_model(assets)
        self._loaded_models.add(bim_model_id)
        return True

    async def _get_device_transform(self, device_id: str) -> RotoTranslationMatrix | None:
        """Fetch calibration matrix for a device."""
        async with self._db.session() as session:
            result = await session.execute(
                text("""
                    SELECT matrix_values FROM calibration_matrices
                    WHERE device_id = :device_id AND is_active = true
                    ORDER BY calibrated_at DESC LIMIT 1
                """),
                {"device_id": device_id},
            )
            row = result.fetchone()
            if not row:
                return None

        return RotoTranslationMatrix.from_flat_list(row.matrix_values)

    async def raycast(
        self,
        device_id: str,
        doa_vector: tuple[float, float, float],
        bim_model_id: str,
    ) -> dict:
        """Perform DOA→BIM asset raycasting.

        Args:
            device_id: Source device.
            doa_vector: DOA unit vector in device-local frame.
            bim_model_id: Target BIM model.

        Returns:
            Dict with ray info and sorted hits.
        """
        loaded = await self._ensure_model_loaded(bim_model_id)
        if not loaded:
            return {"error": "BIM model not found"}

        transform = await self._get_device_transform(device_id)
        if transform is None:
            return {"error": f"No calibration for device {device_id}"}

        hits = self._raycaster.cast_doa(device_id, doa_vector, transform)

        origin = transform.transform_point(np.zeros(3))
        direction = transform.transform_direction(np.array(doa_vector))

        return {
            "ray_origin": origin.tolist(),
            "ray_direction": direction.tolist(),
            "hits": [
                {
                    "asset_id": h.asset_id,
                    "asset_name": h.asset_name,
                    "ifc_type": h.ifc_type,
                    "hit_point": h.hit_point.tolist(),
                    "distance": h.distance,
                }
                for h in hits
            ],
        }

    async def triangulate(
        self,
        observations: list[dict],
        bim_model_id: str,
    ) -> dict:
        """Multi-sensor triangulation.

        Args:
            observations: List of {device_id, direction: {x,y,z}, confidence}.
            bim_model_id: BIM model for nearest-asset lookup.

        Returns:
            Dict with estimated point and nearest asset.
        """
        obs_list = []
        for obs_data in observations:
            device_id = obs_data["device_id"]
            transform = await self._get_device_transform(device_id)
            if transform is None:
                logger.warning("skip_uncalibrated_device", device_id=device_id)
                continue

            doa = (obs_data["direction"]["x"],
                   obs_data["direction"]["y"],
                   obs_data["direction"]["z"])
            confidence = obs_data.get("confidence", 1.0)

            obs = self._triangulator.prepare_observation(
                device_id, doa, transform, confidence,
            )
            obs_list.append(obs)

        result = self._triangulator.triangulate(obs_list)
        if result is None:
            return {"error": "Insufficient valid observations"}

        return {
            "estimated_point": result.estimated_point.tolist(),
            "residual_error": result.residual_error,
            "converged": result.converged,
            "contributing_devices": result.contributing_devices,
        }
