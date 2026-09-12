"""Targeted local ASR over a bounded refinement window.

This module reuses the existing :class:`WhisperEngine` rather than adding a
second Whisper implementation. It acquires the shared heavy-model lease around
the actual transcription call, converts clip-relative word timestamps into
source time, and rejects any word that falls outside the requested window.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from app.core.enums import RefinementPriority
from app.refinement.fingerprints import component_fingerprint
from app.refinement.types import TargetASRResult, WordTimestamp
from app.runtime.heavy_model_lease import NoopHeavyModelLeaseFactory
from app.transcription.engine import TranscriptionResult, WhisperEngine
from app.transcription.service import TranscriptionOptions

_TIMESTAMP_EPSILON = 1e-6


class TargetedASRError(RuntimeError):
    """Targeted local ASR could not produce valid bounded evidence."""


class TargetedASREngine:
    """Run one bounded local faster-whisper pass under the heavy-model lease.

    ``options_for`` is supplied by the caller so model and beam settings stay in
    settings; this module never imports settings. ``context_terms`` are assumed
    already source-evidenced by the executor. They are folded in as an additive
    hotword hint on a copied, frozen options object (the caller's instance is
    never mutated) and are never used to invent missing English.
    """

    def __init__(
        self,
        *,
        options_for: Callable[[RefinementPriority], TranscriptionOptions],
        engine: WhisperEngine,
        lease_factory: object | None = None,
    ) -> None:
        self._options_for = options_for
        self._engine = engine
        self._lease_factory: Any = (
            lease_factory if lease_factory is not None else NoopHeavyModelLeaseFactory()
        )

    def transcribe(
        self,
        audio_path: Path,
        *,
        context_start: float,
        context_end: float,
        priority: RefinementPriority,
        cancel_event: threading.Event | None = None,
        context_terms: Sequence[str] = (),
    ) -> TargetASRResult:
        """Transcribe ``audio_path`` and return window-validated source-time words."""

        if (
            not (math.isfinite(context_start) and math.isfinite(context_end))
            or context_start < 0
            or context_end <= context_start
        ):
            raise TargetedASRError("context bounds are invalid for targeted ASR")

        options = self._options_for(priority)
        effective_options = self._with_context_terms(options, context_terms)

        ownership_lost = False
        with self._lease_factory.acquire(purpose="targeted-asr") as lease:
            result = self._engine.transcribe(
                audio_path, effective_options, cancel_event=cancel_event
            )
            ownership_lost = bool(getattr(lease, "ownership_lost", False))
        if ownership_lost:
            raise TargetedASRError("heavy-model lease was lost during targeted ASR")

        words, rejected_reason = self._source_time_words(
            result, context_start=context_start, context_end=context_end
        )
        transcript = result.raw_text or ""
        if not transcript.strip() and rejected_reason is None:
            rejected_reason = "empty transcript"

        probabilities = [word.probability for word in words if word.probability is not None]
        confidence = sum(probabilities) / len(probabilities) if probabilities else 0.0
        transcript_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
        fingerprint = component_fingerprint(
            "local-asr",
            {
                "model": effective_options.model,
                "device": effective_options.device,
                "compute_type": effective_options.compute_type,
                "beam_size": effective_options.beam_size,
                "language": result.language,
                "context_start": round(float(context_start), 6),
                "context_end": round(float(context_end), 6),
                "context_terms": tuple(context_terms),
                "transcript_hash": transcript_hash,
            },
        )
        settings: dict[str, object] = {
            **asdict(effective_options),
            "context_start": round(float(context_start), 6),
            "context_end": round(float(context_end), 6),
        }
        return TargetASRResult(
            provider="faster-whisper",
            model=effective_options.model,
            language=result.language,
            language_probability=result.language_probability,
            transcript=transcript,
            word_timestamps=tuple(words),
            confidence=confidence,
            runtime_identity={
                "model": effective_options.model,
                "device": effective_options.device,
                "compute_type": effective_options.compute_type,
                "beam_size": effective_options.beam_size,
            },
            fingerprint=fingerprint,
            settings=settings,
            rejected_reason=rejected_reason,
        )

    def _with_context_terms(
        self, options: TranscriptionOptions, context_terms: Sequence[str]
    ) -> TranscriptionOptions:
        hint = " ".join(term.strip() for term in context_terms if term and term.strip())
        if not hint:
            return options
        try:
            return replace(options, hotwords=hint)
        except (TypeError, ValueError):
            return options

    def _source_time_words(
        self,
        result: TranscriptionResult,
        *,
        context_start: float,
        context_end: float,
    ) -> tuple[list[WordTimestamp], str | None]:
        words: list[WordTimestamp] = []
        rejected_reason: str | None = None
        for raw in getattr(result, "word_segments", None) or []:
            try:
                start = float(raw["start"]) + context_start
                end = float(raw["end"]) + context_start
                text = str(raw.get("word", ""))
                probability = raw.get("probability")
            except (KeyError, TypeError, ValueError):
                rejected_reason = "malformed word timestamp"
                continue
            if not (math.isfinite(start) and math.isfinite(end)):
                rejected_reason = "non-finite word timestamp"
                continue
            if not (context_start <= start < end <= context_end + _TIMESTAMP_EPSILON):
                rejected_reason = "word timestamp outside requested window"
                continue
            words.append(
                WordTimestamp(
                    text=text,
                    start=start,
                    end=end,
                    probability=float(probability) if probability is not None else None,
                )
            )
        return words, rejected_reason
