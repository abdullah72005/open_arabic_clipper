"""Configuration values that materially affect transcription output."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TranscriptionOptions:
    """Stable options used to decide whether an existing transcript is reusable."""

    model: str
    device: str
    compute_type: str
    beam_size: int
    language: str | None = None
    word_timestamps: bool = True
    cpu_compute_type: str = "int8"
    cuda_compute_type: str = "float16"
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    condition_on_previous_text: bool = True
    vad_filter: bool = False
    initial_prompt: str | None = None
    hotwords: str | None = None

    def fingerprint(self, audio_hash: str) -> str:
        """Return the deterministic cache key for audio and output-affecting options."""

        payload = json.dumps(
            {"audio_hash": audio_hash, "options": asdict(self)},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
