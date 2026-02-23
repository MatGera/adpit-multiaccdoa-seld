"""Spatial router — proxy to spatial engine for DOA→asset hit queries."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class SpatialQueryRequest(BaseModel):
    device_id: str
    direction: dict  # {x, y, z}
    bim_model_id: str
    max_distance: float = 100.0


class TriangulationRequest(BaseModel):
    observations: list[dict]  # [{device_id, direction: {x,y,z}, confidence}]
    bim_model_id: str


@router.post("/spatial/query")
async def spatial_query(request: SpatialQueryRequest):
    """Query spatial engine: DOA vector → ranked BIM asset hits."""
    # TODO: Forward to spatial engine gRPC service
    return {
        "ray_origin": {"x": 0, "y": 0, "z": 0},
        "ray_direction": request.direction,
        "hits": [],
        "message": "Spatial engine not connected yet",
    }


@router.post("/spatial/triangulate")
async def triangulate(request: TriangulationRequest):
    """Multi-sensor triangulation: multiple DOA observations → 3D point."""
    if len(request.observations) < 2:
        raise HTTPException(status_code=400, detail="At least 2 observations required")

    # TODO: Forward to spatial engine gRPC service
    return {
        "estimated_point": {"x": 0, "y": 0, "z": 0},
        "residual_error": 0.0,
        "nearest_asset": None,
        "contributing_devices": [o["device_id"] for o in request.observations],
    }
