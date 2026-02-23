-- Initialize PostgreSQL extensions for SELD Digital Twin
-- This script runs on first database creation via docker-entrypoint-initdb.d

CREATE EXTENSION IF NOT EXISTS vector;           -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS timescaledb;      -- TimescaleDB for time-series
CREATE EXTENSION IF NOT EXISTS pg_trgm;          -- Trigram similarity for text search
CREATE EXTENSION IF NOT EXISTS btree_gin;        -- GIN indexes for composite search
