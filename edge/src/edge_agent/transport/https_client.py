"""HTTPS client for config sync, OTA updates, and health reporting."""

from __future__ import annotations

from pathlib import Path

import httpx
import structlog

from edge_agent.config import EdgeConfig

logger = structlog.get_logger(__name__)


class CloudHTTPSClient:
    """HTTPS client for edge-to-cloud management communication."""

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self.base_url = config.cloud_api_url
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            headers={"X-Device-ID": config.device_id},
        )

    async def fetch_config(self) -> dict | None:
        """Fetch latest device configuration from cloud."""
        try:
            response = await self._client.get(
                f"/api/v1/devices/{self.config.device_id}"
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            logger.exception("config_fetch_failed")
            return None

    async def check_ota_update(self) -> dict | None:
        """Check if an OTA update is available.

        Returns:
            OTA update metadata dict or None if no update available.
        """
        try:
            response = await self._client.get(
                f"/api/v1/devices/{self.config.device_id}/ota/check",
                params={"current_version": self.config.class_names},
            )
            if response.status_code == 200:
                return response.json()
            return None
        except httpx.HTTPError:
            logger.exception("ota_check_failed")
            return None

    async def download_model(self, url: str, dest: Path) -> bool:
        """Download a model file from the cloud.

        Args:
            url: Download URL for the model.
            dest: Local destination path.

        Returns:
            True if download succeeded.
        """
        try:
            async with self._client.stream("GET", url) as response:
                response.raise_for_status()
                with open(dest, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
            logger.info("model_downloaded", dest=str(dest))
            return True
        except (httpx.HTTPError, OSError):
            logger.exception("model_download_failed")
            return False

    async def report_health(self, health_data: dict) -> None:
        """Report device health to cloud."""
        try:
            await self._client.post(
                f"/api/v1/devices/{self.config.device_id}/health",
                json=health_data,
            )
        except httpx.HTTPError:
            logger.debug("health_report_failed")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
