"""Vision fusion entry point — multi-camera tracking pipeline."""

from __future__ import annotations

import asyncio
import signal

import structlog

from vision_fusion.config import Settings
from vision_fusion.cctv_pipeline import CCTVPipeline

logger = structlog.get_logger(__name__)


async def main() -> None:
    settings = Settings()

    camera_sources = [
        s.strip() for s in settings.camera_sources.split(",")
        if s.strip()
    ]

    if not camera_sources:
        logger.warning("no_camera_sources_configured")
        logger.info("vision_fusion_idle, waiting for camera configuration")
        # Keep alive for health checks
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass
        await stop.wait()
        return

    pipeline = CCTVPipeline(settings, camera_sources)

    logger.info("vision_fusion_starting", cameras=len(camera_sources))
    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())
