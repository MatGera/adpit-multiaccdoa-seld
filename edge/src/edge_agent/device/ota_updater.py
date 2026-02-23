"""OTA (Over-The-Air) update manager for model and configuration updates."""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from edge_agent.config import EdgeConfig
from edge_agent.inference.model_manager import ModelManager
from edge_agent.transport.https_client import CloudHTTPSClient

logger = structlog.get_logger(__name__)


class OTAUpdater:
    """Periodically checks for and applies OTA updates.

    Update types:
    - Model update: new TensorRT engine
    - Config update: threshold, class names, etc.
    """

    def __init__(
        self,
        config: EdgeConfig,
        https_client: CloudHTTPSClient,
        model_manager: ModelManager,
    ) -> None:
        self.config = config
        self.https_client = https_client
        self.model_manager = model_manager
        self._running = False

    async def start(self) -> None:
        """Start the OTA check loop."""
        self._running = True
        logger.info("ota_updater_started", interval_s=self.config.ota_check_interval_s)

        while self._running:
            await asyncio.sleep(self.config.ota_check_interval_s)
            await self._check_and_apply()

    async def stop(self) -> None:
        """Stop the OTA check loop."""
        self._running = False

    async def _check_and_apply(self) -> None:
        """Check for updates and apply if available."""
        try:
            update_info = await self.https_client.check_ota_update()
            if not update_info:
                return

            update_type = update_info.get("type")
            logger.info("ota_update_available", type=update_type)

            if update_type == "model":
                await self._apply_model_update(update_info)
            elif update_type == "config":
                await self._apply_config_update(update_info)

        except Exception:
            logger.exception("ota_check_error")

    async def _apply_model_update(self, info: dict) -> None:
        """Download and swap to a new model."""
        download_url = info.get("url")
        expected_sha256 = info.get("sha256")

        if not download_url:
            logger.error("ota_model_update_missing_url")
            return

        temp_path = Path("/tmp/ota_model.engine")

        success = await self.https_client.download_model(download_url, temp_path)
        if not success:
            return

        swapped = self.model_manager.swap_model(temp_path, expected_sha256)
        if swapped:
            logger.info("ota_model_update_applied", version=info.get("version"))
        else:
            logger.error("ota_model_swap_failed")

        # Clean up temp file
        temp_path.unlink(missing_ok=True)

    async def _apply_config_update(self, info: dict) -> None:
        """Apply a configuration update."""
        new_config = info.get("config", {})
        logger.info("ota_config_update_applied", keys=list(new_config.keys()))
        # Config application would require restarting components
        # For now, just log it
