"""Parametric raycaster: DOA vector → BIM asset intersections.

Uses a BVH-accelerated ray-mesh intersection via trimesh to find
which BIM elements a sound DOA vector passes through.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

import structlog

from spatial_engine.roto_translation import RotoTranslationMatrix

logger = structlog.get_logger(__name__)


@dataclass
class RayHit:
    """A single ray-mesh intersection result."""
    asset_id: str
    asset_name: str
    ifc_type: str
    hit_point: np.ndarray
    distance: float
    face_index: int


class Raycaster:
    """BVH-accelerated raycaster for DOA-to-BIM-asset mapping.

    Given a device position and a DOA direction vector (in world coords),
    casts a ray through the BIM mesh and returns sorted intersections.
    """

    def __init__(self, max_distance: float = 200.0) -> None:
        self._max_distance = max_distance
        self._meshes: dict[str, trimesh.Trimesh] = {}
        self._asset_map: dict[str, dict] = {}  # face_range → asset info
        self._combined_mesh: trimesh.Trimesh | None = None
        self._face_to_asset: list[str] = []

    def load_model(
        self,
        assets: list[dict],
    ) -> None:
        """Load BIM assets for raycasting.

        Args:
            assets: List of dicts with keys: global_id, name, ifc_type, mesh (trimesh.Trimesh)
        """
        meshes = []
        face_to_asset = []

        for asset in assets:
            mesh = asset["mesh"]
            if mesh is None or len(mesh.faces) == 0:
                continue

            # Track which faces belong to which asset
            face_to_asset.extend([asset["global_id"]] * len(mesh.faces))

            self._asset_map[asset["global_id"]] = {
                "name": asset["name"],
                "ifc_type": asset["ifc_type"],
            }
            meshes.append(mesh)

        if meshes:
            self._combined_mesh = trimesh.util.concatenate(meshes)
            self._face_to_asset = face_to_asset
            logger.info("raycaster_model_loaded",
                         assets=len(assets),
                         faces=len(self._face_to_asset))
        else:
            logger.warning("raycaster_no_meshes")

    def cast_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> list[RayHit]:
        """Cast a single ray and return sorted intersections.

        Args:
            origin: 3D ray origin in world coordinates.
            direction: 3D unit direction vector in world coordinates.

        Returns:
            List of RayHit sorted by ascending distance.
        """
        if self._combined_mesh is None:
            return []

        # Normalize direction
        d = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm < 1e-10:
            return []
        d = d / norm

        origin = np.asarray(origin, dtype=np.float64)

        # Ray-mesh intersection
        locations, ray_indices, face_indices = self._combined_mesh.ray.intersects_location(
            ray_origins=origin.reshape(1, 3),
            ray_directions=d.reshape(1, 3),
        )

        if len(locations) == 0:
            return []

        # Build hits
        hits = []
        for loc, face_idx in zip(locations, face_indices):
            distance = float(np.linalg.norm(loc - origin))

            if distance > self._max_distance:
                continue

            asset_id = self._face_to_asset[face_idx] if face_idx < len(self._face_to_asset) else "unknown"
            asset_info = self._asset_map.get(asset_id, {"name": "Unknown", "ifc_type": "Unknown"})

            hits.append(RayHit(
                asset_id=asset_id,
                asset_name=asset_info["name"],
                ifc_type=asset_info["ifc_type"],
                hit_point=loc,
                distance=distance,
                face_index=int(face_idx),
            ))

        # Sort by distance
        hits.sort(key=lambda h: h.distance)
        return hits

    def cast_doa(
        self,
        device_id: str,
        doa_vector: tuple[float, float, float],
        device_transform: RotoTranslationMatrix,
    ) -> list[RayHit]:
        """Cast a DOA ray from a device, transforming to world coordinates.

        Args:
            device_id: Device identifier (for logging).
            doa_vector: DOA unit vector in device-local frame.
            device_transform: Device-to-world roto-translation matrix.

        Returns:
            List of RayHit sorted by distance.
        """
        # Transform origin (device position) and direction to world coords
        origin_world = device_transform.transform_point(np.zeros(3))
        direction_world = device_transform.transform_direction(np.array(doa_vector))

        logger.debug("cast_doa",
                      device_id=device_id,
                      origin=origin_world.tolist(),
                      direction=direction_world.tolist())

        return self.cast_ray(origin_world, direction_world)
