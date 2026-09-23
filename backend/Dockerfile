FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# ffmpeg provides Decoder/ffprobe and the libass-based `ass`/`subtitles`
# filters used for real caption rendering validation. fonts-noto-core is a
# legally distributable Arabic+Latin font family (of which "Noto Sans Arabic"
# is the default CLIPFACTORY_VISUAL_CAPTION_FONT_FAMILY); no proprietary font
# is bundled.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-noto-core && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --upgrade pip && pip install .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
