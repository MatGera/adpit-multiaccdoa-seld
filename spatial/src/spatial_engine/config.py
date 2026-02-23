"""Spatial engine configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://seld:seld_dev_password@localhost:5432/seld_db"

    # gRPC
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051

    # BIM storage
    bim_storage_path: str = "/data/bim_models"
    glb_cache_path: str = "/data/glb_cache"

    # Raycasting
    max_ray_distance: float = 200.0
    bvh_leaf_size: int = 32

    # Triangulation
    min_observations: int = 2
    max_residual_error: float = 5.0
