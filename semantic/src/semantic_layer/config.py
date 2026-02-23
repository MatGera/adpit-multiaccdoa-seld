"""Semantic layer configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://seld:seld_dev_password@localhost:5432/seld_db"

    # gRPC
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50052

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # LLM
    llm_provider: str = "anthropic"  # anthropic | ollama
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.3:latest"

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 8
    similarity_threshold: float = 0.5

    # Hybrid search
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    # Guardrails
    max_tokens: int = 2048
    temperature: float = 0.2
