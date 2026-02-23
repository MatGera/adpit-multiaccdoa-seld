"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import structlog

from api_gateway.config import Settings
from api_gateway.deps import get_db_manager
from api_gateway.routers import predictions, devices, spatial, semantic, bim, vision
from api_gateway.websocket.live_feed import router as ws_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize and cleanup resources."""
    settings = Settings()
    db = get_db_manager(settings)

    logger.info("api_gateway_starting", host=settings.host, port=settings.port)
    yield
    await db.close()
    logger.info("api_gateway_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = Settings()

    app = FastAPI(
        title="SELD Digital Twin API",
        description="Semantic Acoustic Digital Twin — REST/WebSocket Gateway",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
    app.include_router(devices.router, prefix="/api/v1", tags=["devices"])
    app.include_router(spatial.router, prefix="/api/v1", tags=["spatial"])
    app.include_router(semantic.router, prefix="/api/v1", tags=["semantic"])
    app.include_router(bim.router, prefix="/api/v1", tags=["bim"])
    app.include_router(vision.router, prefix="/api/v1", tags=["vision"])
    app.include_router(ws_router, prefix="/api/v1", tags=["websocket"])

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "service": "api-gateway"}

    return app


app = create_app()
