"""Device service entry point — gRPC server."""

from __future__ import annotations

import asyncio
import signal

import grpc
import structlog

from device_service.config import Settings
from device_service.grpc_server import DeviceServicer

logger = structlog.get_logger(__name__)


async def serve() -> None:
    settings = Settings()
    server = grpc.aio.server()

    servicer = DeviceServicer(settings)
    # Register the servicer with generated proto stubs
    # device_pb2_grpc.add_DeviceServiceServicer_to_server(servicer, server)

    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"

    if settings.mtls_enabled:
        try:
            with open(settings.server_cert_path, "rb") as f:
                server_cert = f.read()
            with open(settings.server_key_path, "rb") as f:
                server_key = f.read()
            with open(settings.ca_cert_path, "rb") as f:
                ca_cert = f.read()

            credentials = grpc.ssl_server_credentials(
                [(server_key, server_cert)],
                root_certificates=ca_cert,
                require_client_auth=True,
            )
            server.add_secure_port(listen_addr, credentials)
            logger.info("grpc_server_starting", addr=listen_addr, mtls=True)
        except FileNotFoundError:
            logger.warning("mtls_certs_not_found, falling back to insecure")
            server.add_insecure_port(listen_addr)
    else:
        server.add_insecure_port(listen_addr)
        logger.info("grpc_server_starting", addr=listen_addr, mtls=False)

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
    logger.info("grpc_server_stopped")


if __name__ == "__main__":
    asyncio.run(serve())
