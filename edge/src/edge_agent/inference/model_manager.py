"""Model version management and OTA model swap."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class ModelManager:
    """Manages model versions on the edge device.

    Supports:
    - Loading specific model versions
    - Atomic swap of models during OTA updates
    - SHA256 verification of downloaded models
    """

    def __init__(self, models_dir: str = "/models") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.active_model_path: Path | None = None

    def get_active_model(self) -> Path | None:
        """Get the path to the currently active TensorRT engine."""
        # Look for the symlink or the default engine
        active_link = self.models_dir / "active.engine"
        if active_link.is_symlink() or active_link.exists():
            return active_link.resolve()

        # Fallback: find any .engine file
        engines = list(self.models_dir.glob("*.engine"))
        if engines:
            return engines[0]

        return None

    def swap_model(self, new_model_path: Path, expected_sha256: str | None = None) -> bool:
        """Atomically swap the active model.

        Args:
            new_model_path: Path to the new model file.
            expected_sha256: Expected SHA256 hash for verification.

        Returns:
            True if swap was successful.
        """
        if not new_model_path.exists():
            logger.error("model_swap_failed", reason="file_not_found", path=str(new_model_path))
            return False

        # Verify checksum
        if expected_sha256:
            actual_hash = self._compute_sha256(new_model_path)
            if actual_hash != expected_sha256:
                logger.error(
                    "model_swap_failed",
                    reason="checksum_mismatch",
                    expected=expected_sha256,
                    actual=actual_hash,
                )
                return False

        # Copy to models directory
        dest = self.models_dir / new_model_path.name
        shutil.copy2(new_model_path, dest)

        # Update active symlink atomically
        active_link = self.models_dir / "active.engine"
        temp_link = self.models_dir / "active.engine.tmp"

        try:
            temp_link.symlink_to(dest)
            temp_link.rename(active_link)
        except OSError:
            # On systems without symlink support, just copy
            shutil.copy2(dest, active_link)

        self.active_model_path = dest
        logger.info("model_swapped", new_model=new_model_path.name)
        return True

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
