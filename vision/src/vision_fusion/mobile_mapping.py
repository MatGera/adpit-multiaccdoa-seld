"""Mobile mapping: LiDAR point cloud processing and BIM-Lite generation.

Ingests point clouds from mobile mapping platforms (e.g., Leica BLK2GO),
performs 3D semantic segmentation, and generates lightweight BIM models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import structlog

if TYPE_CHECKING:
    from vision_fusion.config import Settings

logger = structlog.get_logger(__name__)


@dataclass
class SegmentedElement:
    """A semantically segmented element from a point cloud."""
    class_name: str
    points: np.ndarray  # Nx3
    confidence: float
    bounding_box: tuple[np.ndarray, np.ndarray]  # min, max corners


class MobileMappingPipeline:
    """Processes LiDAR point clouds for BIM-Lite generation.

    Pipeline:
    1. Ingest raw point cloud (LAS/LAZ/PLY/PCD)
    2. Preprocess: downsample, denoise, align
    3. 3D semantic segmentation (PointNet++/Point Transformer)
    4. Extract structured elements
    5. Generate BIM-Lite model
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._storage = Path(settings.point_cloud_storage)
        self._storage.mkdir(parents=True, exist_ok=True)

    def ingest_point_cloud(self, file_path: str) -> dict:
        """Ingest a point cloud file.

        Args:
            file_path: Path to LAS/LAZ/PLY/PCD file.

        Returns:
            Dict with ingestion results.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in (".las", ".laz"):
            return self._ingest_las(file_path)
        elif suffix in (".ply", ".pcd"):
            return self._ingest_open3d(file_path)
        else:
            return {"error": f"Unsupported format: {suffix}"}

    def _ingest_las(self, file_path: str) -> dict:
        """Ingest a LAS/LAZ file."""
        try:
            import laspy

            las = laspy.read(file_path)
            points = np.vstack([las.x, las.y, las.z]).T

            logger.info("las_ingested",
                         file=file_path,
                         points=points.shape[0])

            return {
                "status": "ingested",
                "format": "LAS",
                "num_points": points.shape[0],
                "bounds_min": points.min(axis=0).tolist(),
                "bounds_max": points.max(axis=0).tolist(),
            }
        except ImportError:
            return {"error": "laspy not installed. Install with: pip install laspy"}

    def _ingest_open3d(self, file_path: str) -> dict:
        """Ingest PLY/PCD via Open3D."""
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)

            logger.info("pointcloud_ingested",
                         file=file_path,
                         points=points.shape[0])

            return {
                "status": "ingested",
                "format": Path(file_path).suffix.upper().strip("."),
                "num_points": points.shape[0],
                "bounds_min": points.min(axis=0).tolist(),
                "bounds_max": points.max(axis=0).tolist(),
            }
        except ImportError:
            return {"error": "open3d not installed. Install with: pip install open3d"}

    def preprocess(
        self,
        points: np.ndarray,
        voxel_size: float = 0.05,
    ) -> np.ndarray:
        """Preprocess point cloud: voxel downsample and statistical outlier removal.

        Args:
            points: Nx3 numpy array.
            voxel_size: Voxel size for downsampling.

        Returns:
            Processed Nx3 array.
        """
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Voxel downsample
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

            # Statistical outlier removal
            pcd_clean, _ = pcd_down.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )

            result = np.asarray(pcd_clean.points)
            logger.info("preprocessed",
                         original=points.shape[0],
                         result=result.shape[0])
            return result

        except ImportError:
            logger.warning("open3d_not_available, skipping preprocessing")
            return points

    def segment_3d(self, points: np.ndarray) -> list[SegmentedElement]:
        """Perform 3D semantic segmentation on a point cloud.

        Uses PointNet++ or Point Transformer model.
        Currently returns a placeholder — actual model loading and
        inference would be integrated here.

        Args:
            points: Nx3 preprocessed point cloud.

        Returns:
            List of segmented elements.
        """
        # TODO: Integrate actual 3D segmentation model
        # (PointNet++ / Point Transformer v3)
        logger.info("3d_segmentation_placeholder", points=points.shape[0])

        return [
            SegmentedElement(
                class_name="floor",
                points=points[:100] if len(points) > 100 else points,
                confidence=0.85,
                bounding_box=(
                    points.min(axis=0),
                    points.max(axis=0),
                ),
            )
        ]
