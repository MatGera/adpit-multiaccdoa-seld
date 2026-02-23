"""Predictions router — query historical and live prediction data."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.deps import get_db_session

router = APIRouter()


@router.get("/predictions")
async def query_predictions(
    device_id: str | None = Query(None),
    class_name: str | None = Query(None),
    start_time: datetime | None = Query(None),
    end_time: datetime | None = Query(None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db_session),
):
    """Query historical predictions with time range, device, and class filters."""
    conditions = ["confidence >= :min_confidence"]
    params: dict = {"min_confidence": min_confidence, "limit": limit, "offset": offset}

    if device_id:
        conditions.append("device_id = :device_id")
        params["device_id"] = device_id
    if class_name:
        conditions.append("class_name = :class_name")
        params["class_name"] = class_name
    if start_time:
        conditions.append("time >= :start_time")
        params["start_time"] = start_time
    if end_time:
        conditions.append("time <= :end_time")
        params["end_time"] = end_time

    where_clause = " AND ".join(conditions)

    query = text(f"""
        SELECT time, device_id, frame_idx, class_name, confidence,
               vector_x, vector_y, vector_z
        FROM predictions
        WHERE {where_clause}
        ORDER BY time DESC
        LIMIT :limit OFFSET :offset
    """)

    result = await session.execute(query, params)
    rows = result.fetchall()

    return {
        "predictions": [
            {
                "time": row.time.isoformat(),
                "device_id": row.device_id,
                "frame_idx": row.frame_idx,
                "class": row.class_name,
                "confidence": row.confidence,
                "vector": [row.vector_x, row.vector_y, row.vector_z],
            }
            for row in rows
        ],
        "count": len(rows),
        "limit": limit,
        "offset": offset,
    }
