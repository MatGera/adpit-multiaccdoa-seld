"""Edge agent main entry point — orchestrates the audio → inference → publish pipeline."""

from __future__ import annotations

import asyncio
import gc
import signal
import time

import numpy as np
import structlog

from edge_agent.audio.buffer_manager import SecureBufferManager
from edge_agent.audio.pdm_capture import AudioCapture
from edge_agent.audio.foa_encoder import FOAEncoder
from edge_agent.config import EdgeConfig
from edge_agent.features.feature_pipeline import FeaturePipeline
from edge_agent.inference.postprocess import decode_multi_accdoa
from edge_agent.inference.tensorrt_engine import InferenceEngine
from edge_agent.transport.mqtt_publisher import MQTTPublisher
from edge_agent.device.health import HealthMonitor

logger = structlog.get_logger(__name__)


class EdgeAgent:
    """Main edge agent orchestrating the SELD pipeline."""

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self.running = False

        # Pipeline components
        self.capture = AudioCapture(config)
        self.foa_encoder = FOAEncoder(config.num_channels)
        self.features = FeaturePipeline(config)
        self.engine = InferenceEngine(config)
        self.publisher = MQTTPublisher(config)
        self.health = HealthMonitor(config)
        self.buffer_manager = SecureBufferManager()

    async def start(self) -> None:
        """Initialize all components and start the pipeline loop."""
        logger.info("edge_agent_starting", device_id=self.config.device_id)

        # Connect MQTT
        await self.publisher.connect()

        # Load TensorRT engine
        self.engine.load()

        self.running = True
        logger.info("edge_agent_started", device_id=self.config.device_id)

        # Start pipeline loop
        try:
            await self._pipeline_loop()
        except asyncio.CancelledError:
            logger.info("edge_agent_cancelled")
        finally:
            await self.shutdown()

    async def _pipeline_loop(self) -> None:
        """Continuous audio capture → feature extraction → inference → publish loop."""
        frame_idx = 0

        while self.running:
            t_start = time.perf_counter()

            try:
                # 1. Capture audio frame
                raw_audio = self.capture.read_frame()
                self.buffer_manager.register(raw_audio)

                # 2. FOA encoding
                foa = self.foa_encoder.encode(raw_audio)
                self.buffer_manager.register(foa)

                # 3. Feature extraction (Log-Mel + SALSA + IV)
                features = self.features.extract(foa)
                self.buffer_manager.register(features)

                # 4. TensorRT inference
                t_infer_start = time.perf_counter()
                output = self.engine.infer(features)
                inference_ms = (time.perf_counter() - t_infer_start) * 1000

                # 5. Post-process Multi-ACCDOA output
                predictions = decode_multi_accdoa(
                    output,
                    class_names=self.config.class_names,
                    threshold=self.config.confidence_threshold,
                )

                # 6. Publish predictions via MQTT
                if predictions:
                    await self.publisher.publish_predictions(
                        predictions=predictions,
                        frame_idx=frame_idx,
                        inference_ms=inference_ms,
                        health=self.health.snapshot(),
                    )

                # 7. Destroy audio buffers (privacy-by-design)
                self.buffer_manager.destroy_all()
                gc.collect()

                frame_idx += 1

                total_ms = (time.perf_counter() - t_start) * 1000
                logger.debug(
                    "frame_processed",
                    frame_idx=frame_idx,
                    inference_ms=round(inference_ms, 1),
                    total_ms=round(total_ms, 1),
                    predictions=len(predictions),
                )

            except Exception:
                logger.exception("pipeline_error", frame_idx=frame_idx)
                self.buffer_manager.destroy_all()
                gc.collect()
                await asyncio.sleep(0.1)

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        self.running = False
        self.buffer_manager.destroy_all()
        await self.publisher.disconnect()
        self.engine.unload()
        logger.info("edge_agent_shutdown")


async def main() -> None:
    """Entry point."""
    config = EdgeConfig()
    agent = EdgeAgent(config)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(agent.shutdown()))

    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
