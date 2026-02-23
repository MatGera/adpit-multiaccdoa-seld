"""WebSocket endpoint for real-time prediction streaming."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import structlog

logger = structlog.get_logger(__name__)

router = APIRouter()

# Connected WebSocket clients
_clients: set[WebSocket] = set()


@router.websocket("/predictions/live")
async def prediction_live_feed(websocket: WebSocket):
    """WebSocket: stream real-time predictions to frontend clients.

    Clients connect and receive a continuous stream of prediction events
    as they are ingested from edge devices via Kafka.
    """
    await websocket.accept()
    _clients.add(websocket)
    logger.info("ws_client_connected", total=len(_clients))

    try:
        while True:
            # Keep connection alive; actual data pushed via broadcast_prediction()
            data = await websocket.receive_text()
            # Clients can send filter preferences
            try:
                filters = json.loads(data)
                logger.debug("ws_filter_updated", filters=filters)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        _clients.discard(websocket)
        logger.info("ws_client_disconnected", total=len(_clients))


async def broadcast_prediction(prediction: dict) -> None:
    """Broadcast a prediction event to all connected WebSocket clients."""
    if not _clients:
        return

    message = json.dumps(prediction)
    disconnected = set()

    for client in _clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)

    _clients.difference_update(disconnected)
