"""Dependency injection for database, Redis, gRPC stubs."""

from __future__ import annotations

from functools import lru_cache

from seld_common.db import DatabaseManager
from api_gateway.config import Settings


@lru_cache
def get_settings() -> Settings:
    return Settings()


_db_manager: DatabaseManager | None = None


def get_db_manager(settings: Settings | None = None) -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        s = settings or get_settings()
        _db_manager = DatabaseManager(s.database_url)
    return _db_manager


async def get_db_session():
    """FastAPI dependency: yield an async database session."""
    db = get_db_manager()
    async with db.session() as session:
        yield session
