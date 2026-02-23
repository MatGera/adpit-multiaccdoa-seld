FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# IfcOpenShell may need build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglu1-mesa && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY libs/seld-common /app/libs/seld-common
COPY spatial/pyproject.toml /app/
COPY spatial/src /app/src

RUN pip install --no-cache-dir .

EXPOSE 50051

CMD ["python", "-m", "spatial_engine.main"]
