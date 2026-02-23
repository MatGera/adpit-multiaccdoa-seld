"""Vision fusion router — camera tracking and homography endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HomographyRequest(BaseModel):
    camera_id: str
    bim_model_id: str
    matrix: list[float]  # 3x3 homography matrix (9 elements)


@router.post("/vision/homography")
async def set_homography(request: HomographyRequest):
    """Set camera-to-BIM homography matrix for CCTV pipeline."""
    if len(request.matrix) != 9:
        return {"error": "Homography matrix must have exactly 9 elements (3x3)"}

    # TODO: Store in Redis/DB and notify vision fusion service
    return {"status": "updated", "camera_id": request.camera_id}


@router.get("/vision/tracks")
async def get_tracks():
    """Get current vehicle/object tracks from vision fusion."""
    # TODO: Query vision fusion service / Redis state cache
    return {"tracks": [], "count": 0}
