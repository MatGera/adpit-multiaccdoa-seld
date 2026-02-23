# Production Dockerfile for Jetson Orin (aarch64 / L4T)
# Base image: NVIDIA L4T with JetPack 6.x + TensorRT
FROM nvcr.io/nvidia/l4t-tensorrt:r36.4.0 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libasound2-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install NanoMQ
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -sSL https://www.emqx.com/en/downloads/nanomq/v0.21.0/nanomq-0.21.0-linux-aarch64.deb -o nanomq.deb \
    && dpkg -i nanomq.deb \
    && rm nanomq.deb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy shared library
COPY libs/seld-common /app/libs/seld-common

# Copy edge agent
COPY edge/pyproject.toml /app/
COPY edge/src /app/src

# Install Python dependencies
RUN python3.12 -m pip install --no-cache-dir .

# Copy NanoMQ config
COPY edge/nanomq.conf /etc/nanomq.conf

# Create model directory
RUN mkdir -p /app/models

EXPOSE 1883

# Run both NanoMQ broker and edge agent
CMD ["python3.12", "-m", "edge_agent.main"]
