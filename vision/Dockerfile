FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY libs/seld-common /app/libs/seld-common
COPY vision/pyproject.toml /app/
COPY vision/src /app/src

RUN pip install --no-cache-dir .

CMD ["python", "-m", "vision_fusion.main"]
