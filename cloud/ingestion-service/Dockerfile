FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY libs/seld-common /app/libs/seld-common
COPY cloud/ingestion-service/pyproject.toml /app/
COPY cloud/ingestion-service/src /app/src

RUN pip install --no-cache-dir .

CMD ["python", "-m", "ingestion_service.main"]
