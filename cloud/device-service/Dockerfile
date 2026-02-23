FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY libs/seld-common /app/libs/seld-common
COPY cloud/device-service/pyproject.toml /app/
COPY cloud/device-service/src /app/src

RUN pip install --no-cache-dir .

EXPOSE 50053

CMD ["python", "-m", "device_service.main"]
