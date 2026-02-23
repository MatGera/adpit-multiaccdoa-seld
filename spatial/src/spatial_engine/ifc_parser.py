"""IFC model parser: extracts geometry and metadata using IfcOpenShell.

Converts IFC elements to trimesh objects for raycasting and exports
glTF/GLB for the frontend viewer.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import ifcopenshell
import ifcopenshell.geom
import trimesh

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BIMAsset:
    """Represents a parsed BIM element with geometry and metadata."""
    global_id: str
    name: str
    ifc_type: str
    mesh: trimesh.Trimesh | None = None
    properties: dict = field(default_factory=dict)


@dataclass
class ParsedModel:
    """Result of parsing an IFC file."""
    model_id: str
    file_path: str
    assets: list[BIMAsset]
    combined_mesh: trimesh.Trimesh | None = None
    bounds_min: np.ndarray | None = None
    bounds_max: np.ndarray | None = None


class IFCParser:
    """Parses IFC files into triangulated meshes for spatial queries."""

    def __init__(self, storage_path: str = "/data/bim_models") -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # IfcOpenShell geometry settings for triangulation
        self._geom_settings = ifcopenshell.geom.settings()
        self._geom_settings.set(self._geom_settings.USE_WORLD_COORDS, True)

    def parse(self, ifc_path: str, model_id: str) -> ParsedModel:
        """Parse an IFC file and extract all elements with geometry.

        Args:
            ifc_path: Path to the .ifc file.
            model_id: Unique identifier for this BIM model.

        Returns:
            ParsedModel with all assets and combined mesh.
        """
        logger.info("ifc_parse_start", path=ifc_path, model_id=model_id)

        ifc_file = ifcopenshell.open(ifc_path)
        assets: list[BIMAsset] = []
        meshes: list[trimesh.Trimesh] = []

        # Iterate over all products with geometry
        products = ifc_file.by_type("IfcProduct")

        for product in products:
            try:
                shape = ifcopenshell.geom.create_shape(self._geom_settings, product)
            except Exception:
                continue  # No geometry for this element

            # Extract triangulated mesh
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            faces = np.array(shape.geometry.faces).reshape(-1, 3)

            if len(verts) == 0 or len(faces) == 0:
                continue

            mesh = trimesh.Trimesh(vertices=verts, faces=faces)

            # Extract properties
            props = {}
            if hasattr(product, "Name") and product.Name:
                props["name"] = product.Name
            if hasattr(product, "Description") and product.Description:
                props["description"] = product.Description

            asset = BIMAsset(
                global_id=product.GlobalId,
                name=product.Name or product.is_a(),
                ifc_type=product.is_a(),
                mesh=mesh,
                properties=props,
            )
            assets.append(asset)
            meshes.append(mesh)

        # Combine all meshes into a single scene mesh
        combined = trimesh.util.concatenate(meshes) if meshes else None

        bounds_min = combined.bounds[0] if combined else None
        bounds_max = combined.bounds[1] if combined else None

        logger.info("ifc_parse_complete",
                     model_id=model_id,
                     assets=len(assets),
                     vertices=combined.vertices.shape[0] if combined else 0)

        return ParsedModel(
            model_id=model_id,
            file_path=ifc_path,
            assets=assets,
            combined_mesh=combined,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )

    def export_glb(self, parsed: ParsedModel, output_path: str | None = None) -> str:
        """Export parsed model to glTF Binary (.glb) for frontend viewer.

        Args:
            parsed: Parsed IFC model.
            output_path: Optional output path. Defaults to storage_path/model_id.glb.

        Returns:
            Path to the exported GLB file.
        """
        if output_path is None:
            output_path = str(self._storage_path / f"{parsed.model_id}.glb")

        scene = trimesh.Scene()
        for asset in parsed.assets:
            if asset.mesh is not None:
                scene.add_geometry(asset.mesh, node_name=asset.global_id)

        scene.export(output_path, file_type="glb")
        logger.info("glb_exported", path=output_path, assets=len(parsed.assets))

        return output_path
