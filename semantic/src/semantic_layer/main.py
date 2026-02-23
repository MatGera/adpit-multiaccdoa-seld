"""Semantic layer entry point — gRPC server."""

from __future__ import annotations

import asyncio
import signal

import grpc
import structlog

from semantic_layer.config import Settings

logger = structlog.get_logger(__name__)


async def serve() -> None:
    settings = Settings()
    server = grpc.aio.server()

    from semantic_layer.grpc_servicer import SemanticServicer

    servicer = SemanticServicer(settings)
    # semantic_pb2_grpc.add_SemanticServiceServicer_to_server(servicer, server)

    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    logger.info("semantic_layer_starting", addr=listen_addr)
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
    logger.info("semantic_layer_stopped")


if __name__ == "__main__":
    asyncio.run(serve())
