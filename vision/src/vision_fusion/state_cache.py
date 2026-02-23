"""Redis state cache for vision fusion tracking state."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import redis.asyncio as redis

import structlog

if TYPE_CHECKING:
    from vision_fusion.config import Settings

logger = structlog.get_logger(__name__)


class StateCache:
    """Redis-backed state cache for real-time track data.

    Stores current object positions and track history for
    cross-service querying (e.g., API Gateway, Spatial Engine).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        self._client = redis.from_url(
            self._settings.redis_url,
            decode_responses=True,
        )
        await self._client.ping()
        logger.info("redis_connected", url=self._settings.redis_url)

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()

    async def update_track(
        self,
        camera_id: str,
        track_id: int,
        class_name: str,
        confidence: float,
        pixel_coords: tuple[float, float],
        bim_coords: tuple[float, float] | None = None,
    ) -> None:
        """Update a track in the cache.

        Stores current position and appends to recent trajectory.
        TTL ensures stale tracks are automatically cleaned up.
        """
        assert self._client is not None

        key = f"vision:track:{camera_id}:{track_id}"
        data = {
            "camera_id": camera_id,
            "track_id": track_id,
            "class_name": class_name,
            "confidence": confidence,
            "pixel_x": pixel_coords[0],
            "pixel_y": pixel_coords[1],
            "bim_x": bim_coords[0] if bim_coords else None,
            "bim_y": bim_coords[1] if bim_coords else None,
            "timestamp": time.time(),
        }

        await self._client.set(key, json.dumps(data), ex=60)

        # Also maintain a set of active tracks per camera
        camera_key = f"vision:active_tracks:{camera_id}"
        await self._client.sadd(camera_key, str(track_id))
        await self._client.expire(camera_key, 60)

    async def get_tracks(self, camera_id: str | None = None) -> list[dict]:
        """Get all active tracks, optionally filtered by camera.

        Returns:
            List of track data dicts.
        """
        assert self._client is not None

        if camera_id:
            pattern = f"vision:track:{camera_id}:*"
        else:
            pattern = "vision:track:*"

        tracks = []
        async for key in self._client.scan_iter(pattern):
            data = await self._client.get(key)
            if data:
                tracks.append(json.loads(data))

        return tracks
