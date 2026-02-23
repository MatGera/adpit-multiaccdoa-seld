"""Spatial engine entry point — gRPC server."""

from __future__ import annotations

import asyncio
import signal

import grpc
import structlog

from spatial_engine.config import Settings

logger = structlog.get_logger(__name__)


async def serve() -> None:
    settings = Settings()
    server = grpc.aio.server()

    # Import and register servicers
    from spatial_engine.grpc_servicer import SpatialServicer

    servicer = SpatialServicer(settings)
    # spatial_pb2_grpc.add_SpatialServiceServicer_to_server(servicer, server)

    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    logger.info("spatial_engine_starting", addr=listen_addr)
    await server.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    await stop_event.wait()
    await server.stop(grace=5)
    logger.info("spatial_engine_stopped")


if __name__ == "__main__":
    asyncio.run(serve())
