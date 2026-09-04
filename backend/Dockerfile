FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --upgrade pip && pip install .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
