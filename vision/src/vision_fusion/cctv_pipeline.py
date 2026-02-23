"""CCTV-to-BIM pipeline: YOLO detection + multi-object tracking + homography.

Processes RTSP camera feeds, detects objects with YOLO, tracks them with
BoT-SORT/ByteTrack, and maps pixel coordinates to BIM coordinates via
homography matrices.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
from ultralytics import YOLO

import structlog

from vision_fusion.tracker import TrackerManager
from vision_fusion.homography import HomographyMapper
from vision_fusion.state_cache import StateCache

if TYPE_CHECKING:
    from vision_fusion.config import Settings

logger = structlog.get_logger(__name__)


class CCTVPipeline:
    """Multi-camera CCTV processing pipeline."""

    def __init__(self, settings: Settings, camera_sources: list[str]) -> None:
        self._settings = settings
        self._camera_sources = camera_sources
        self._model: YOLO | None = None
        self._tracker = TrackerManager(settings)
        self._homography = HomographyMapper(settings)
        self._state_cache = StateCache(settings)
        self._running = False

    def _load_model(self) -> YOLO:
        """Load YOLO model for object detection."""
        if self._model is None:
            logger.info("loading_yolo_model", model=self._settings.yolo_model)
            self._model = YOLO(self._settings.yolo_model)
        return self._model

    async def start(self) -> None:
        """Start processing all camera feeds concurrently."""
        self._running = True
        await self._state_cache.connect()

        tasks = [
            asyncio.create_task(self._process_camera(i, source))
            for i, source in enumerate(self._camera_sources)
        ]

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _stop():
            self._running = False
            stop_event.set()

        for sig in (asyncio.coroutines._get_running_loop,):
            pass

        import signal
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _stop)
            except NotImplementedError:
                pass

        await stop_event.wait()

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        await self._state_cache.disconnect()

    async def _process_camera(self, camera_idx: int, source: str) -> None:
        """Process a single camera feed."""
        camera_id = f"cam-{camera_idx:03d}"
        logger.info("camera_pipeline_starting", camera_id=camera_id, source=source)

        model = self._load_model()
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error("camera_open_failed", camera_id=camera_id, source=source)
            return

        frame_interval = 1.0 / self._settings.fps_limit

        try:
            while self._running:
                t_start = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("camera_frame_read_failed", camera_id=camera_id)
                    await asyncio.sleep(1.0)
                    continue

                # Run YOLO detection + tracking
                results = model.track(
                    frame,
                    conf=self._settings.yolo_confidence,
                    iou=self._settings.yolo_iou,
                    tracker=f"{self._settings.tracker_type}.yaml",
                    persist=True,
                    verbose=False,
                )

                # Process detections
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes

                    for box in boxes:
                        if box.id is None:
                            continue

                        track_id = int(box.id.item())
                        class_id = int(box.cls.item())
                        class_name = model.names[class_id]
                        confidence = float(box.conf.item())
                        bbox = box.xyxy[0].cpu().numpy()

                        # Compute centroid in pixel space
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2

                        # Map to BIM coordinates via homography
                        bim_coords = self._homography.pixel_to_bim(
                            camera_id, np.array([cx, cy])
                        )

                        # Update state cache
                        await self._state_cache.update_track(
                            camera_id=camera_id,
                            track_id=track_id,
                            class_name=class_name,
                            confidence=confidence,
                            pixel_coords=(float(cx), float(cy)),
                            bim_coords=bim_coords,
                        )

                # Rate limit
                elapsed = time.monotonic() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        finally:
            cap.release()
            logger.info("camera_pipeline_stopped", camera_id=camera_id)
