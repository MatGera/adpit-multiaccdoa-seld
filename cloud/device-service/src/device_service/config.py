"""Device service configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://seld:seld_dev_password@localhost:5432/seld_db"

    # gRPC
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50053

    # Certificate authority
    ca_cert_path: str = "/etc/seld/certs/ca.pem"
    ca_key_path: str = "/etc/seld/certs/ca-key.pem"
    cert_validity_days: int = 365

    # mTLS
    mtls_enabled: bool = False
    server_cert_path: str = "/etc/seld/certs/server.pem"
    server_key_path: str = "/etc/seld/certs/server-key.pem"
