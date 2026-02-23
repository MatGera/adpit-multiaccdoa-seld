FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY libs/seld-common /app/libs/seld-common
COPY cloud/api-gateway/pyproject.toml /app/
COPY cloud/api-gateway/src /app/src

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
