# Training Dockerfile — CUDA 12.x + PyTorch 2.x
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy shared library
COPY libs/seld-common /app/libs/seld-common

# Copy training package
COPY training/pyproject.toml /app/
COPY training/src /app/src

# Install dependencies
RUN pip install --no-cache-dir ".[augmentation]"

# Copy config files
COPY training/src/seld_training/config /app/config

CMD ["python", "-m", "seld_training.train"]
