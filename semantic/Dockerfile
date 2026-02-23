FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY libs/seld-common /app/libs/seld-common
COPY semantic/pyproject.toml /app/
COPY semantic/src /app/src

RUN pip install --no-cache-dir .

EXPOSE 50052

CMD ["python", "-m", "semantic_layer.main"]
