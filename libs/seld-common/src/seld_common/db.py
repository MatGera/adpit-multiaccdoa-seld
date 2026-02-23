"""SQLAlchemy + asyncpg database session factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

import structlog

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""
    pass


class DatabaseManager:
    """Manages async database connections and sessions."""

    def __init__(
        self,
        database_url: str,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
    ) -> None:
        """Initialize the database manager.

        Args:
            database_url: PostgreSQL connection URL
                (e.g. postgresql+asyncpg://user:pass@host:5432/dbname)
            echo: Whether to echo SQL statements.
            pool_size: Connection pool size.
            max_overflow: Max overflow connections above pool_size.
        """
        self._engine: AsyncEngine = create_async_engine(
            database_url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a transactional async session scope."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_tables(self) -> None:
        """Create all registered tables (for dev/testing)."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_tables_created")

    async def close(self) -> None:
        """Close the engine and release connections."""
        await self._engine.dispose()
        logger.info("database_closed")
