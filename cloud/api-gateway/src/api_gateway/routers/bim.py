"""BIM model management router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.deps import get_db_session

router = APIRouter()


@router.post("/bim/upload")
async def upload_bim_model(
    file: UploadFile = File(...),
    name: str = "",
    session: AsyncSession = Depends(get_db_session),
):
    """Upload an IFC model for spatial engine processing."""
    if not file.filename or not file.filename.lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="Only .ifc files accepted")

    content = await file.read()

    # TODO: Save file to BIM storage, trigger IFC parsing pipeline
    return {
        "status": "processing",
        "file_name": file.filename,
        "file_size": len(content),
        "model_id": "pending",
    }


@router.get("/bim/{model_id}/glb")
async def download_glb(
    model_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Download the glTF/GLB representation of a BIM model for the 3D viewer."""
    result = await session.execute(
        text("SELECT glb_file_path FROM bim_models WHERE id = :id AND status = 'ready'"),
        {"id": model_id},
    )
    row = result.fetchone()

    if not row or not row.glb_file_path:
        raise HTTPException(status_code=404, detail="GLB model not found or not ready")

    return FileResponse(
        row.glb_file_path,
        media_type="model/gltf-binary",
        filename=f"{model_id}.glb",
    )
