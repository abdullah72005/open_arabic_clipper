"""Shared Stage 5.0 test helpers: managed media, fake prober, seeded contract.

Every test is hermetic: the ffprobe seam is always a fake, no provider is ever
constructed, and no rendering/caption/TTS path is exercised.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stage43_support import (
    SelectionFixture,
    install_selection_settings,
    make_result_spec,
    seed_selection_fixture,
)

from app.core.enums import (
    RefinementPriority,
    RefinementStatus,
    TransformationSelectionStatus,
)
from app.core.settings import get_settings
from app.media.ffprobe import MediaMetadata, ProbeExecutionError
from app.models import CandidateRefinement
from app.render.fingerprints import build_caption_source_payload as _caption_payload
from app.render.fingerprints import caption_source_fingerprint
from app.render.types import ClipWord, FinalClipEvidence
from app.services.storage import StorageService
from app.transformation.selection.service import select_transformation_plan

SAMPLE_METADATA = MediaMetadata(
    duration_seconds=90.0,
    video_codec="h264",
    width=1920,
    height=1080,
    frames_per_second=30.0,
    audio_codec="aac",
    audio_sample_rate=48_000,
)


class FakeProber:
    """Injectable ffprobe seam that never invokes a real binary."""

    def __init__(
        self,
        metadata: MediaMetadata | None = None,
        error: Exception | None = None,
    ) -> None:
        self.metadata = metadata or SAMPLE_METADATA
        self.error = error
        self.calls = 0

    def probe(self, path: Path) -> MediaMetadata:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.metadata


def no_video_prober() -> FakeProber:
    from app.media.ffprobe import ProbeParseError

    return FakeProber(error=ProbeParseError("no video stream"))


def corrupt_prober() -> FakeProber:
    return FakeProber(error=ProbeExecutionError("probe failed"))


@dataclass
class Stage50Fixture:
    selection: SelectionFixture
    storage: StorageService
    prober: FakeProber
    source_path: Path
    settings: Any


def managed_source(source: Any, *, size: int = 2048) -> tuple[StorageService, Path]:
    settings = get_settings()
    storage = StorageService(settings.storage_root)
    directory = storage.source_directory(source.id)
    path = directory / "source.mp4"
    path.write_bytes(b"\x00" * size)
    source.source_uri = str(path)
    return storage, path


def seed_stage50(
    session: Any,
    *,
    result_specs: list[dict[str, object]] | None = None,
    narration: str = "NONE",
    final_transcript: str | None = None,
    refinement_status: RefinementStatus = RefinementStatus.CANDIDATE_REFINED,
    clear_hook_payoff: bool = False,
    create_media: bool = True,
    prober: FakeProber | None = None,
    planning_on_final: bool = True,
    settings: Any = None,
    source_only: bool = False,
) -> Stage50Fixture:
    specs = result_specs or [make_result_spec()]
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=specs,
        planning_on_final=planning_on_final,
        plan_narration=narration,
        refinement_status=refinement_status,
        final_transcript=final_transcript,
        source_only=source_only,
    )
    if clear_hook_payoff:
        for plan in fixture.plans:
            plan.hook_payoff_evidence = {"hook_index": None, "payoff_index": None}
        session.flush()
    storage = StorageService(get_settings().storage_root)
    source_path = Path()
    if create_media:
        storage, source_path = managed_source(fixture.source)
        session.flush()
    view = select_transformation_plan(session, fixture.candidate.id)
    assert view is not None
    return Stage50Fixture(
        selection=fixture,
        storage=storage,
        prober=prober or FakeProber(),
        source_path=source_path,
        settings=fixture.settings,
    )


def selected_status(view: Any) -> str:
    return view.row.status.value


def assert_selected(view: Any) -> None:
    assert view.row.status.value in {
        TransformationSelectionStatus.PLAN_SELECTED.value,
        TransformationSelectionStatus.PLAN_SELECTED_WITH_CAUTION.value,
    }


def clip_words(
    text: str,
    *,
    start: float = 21.0,
    end: float = 44.0,
) -> tuple[ClipWord, ...]:
    tokens = text.split()
    if not tokens:
        return ()
    step = (end - start) / len(tokens)
    return tuple(
        ClipWord(
            index=index,
            text=token,
            start=round(start + index * step, 3),
            end=round(start + (index + 1) * step, 3),
            probability=0.9,
        )
        for index, token in enumerate(tokens)
    )


def make_final(
    text: str,
    *,
    words: tuple[ClipWord, ...] | None = None,
    refined_start: float = 20.5,
    refined_end: float = 44.5,
    unresolved_spans: tuple[dict[str, object], ...] = (),
    code_switch_evidence: dict[str, object] | None = None,
    dialect_profile: str | None = "EGYPTIAN",
    output_fingerprint: str = "final-fp",
) -> FinalClipEvidence:
    return FinalClipEvidence(
        refinement_id="final-id",
        priority=RefinementPriority.FINAL_CLIP.value,
        quality_level="FINAL_CLIP",
        status=RefinementStatus.FINAL_TRANSCRIPT_READY.value,
        final_transcript=text,
        refined_start=refined_start,
        refined_end=refined_end,
        words=words if words is not None else clip_words(text),
        unresolved_spans=unresolved_spans,
        code_switch_evidence=code_switch_evidence or {"suspected": False, "tokens": []},
        dialect_profile=dialect_profile,
        dialect_confidence=0.9,
        output_fingerprint=output_fingerprint,
    )


def source_block(
    text: str,
    *,
    index: int = 0,
    source_start: float = 20.5,
    source_end: float = 44.5,
    word_start_index: int | None = None,
    word_end_index: int | None = None,
    role: str = "HERO",
) -> dict[str, object]:
    return {
        "index": index,
        "block_type": "SOURCE_EXCERPT",
        "purpose": "Hero source moment",
        "estimated_duration": round(source_end - source_start, 3),
        "placement": "",
        "interrupts_source": False,
        "preservation_constraints": [],
        "dependency_ids": [],
        "source_role": role,
        "word_start_index": word_start_index,
        "word_end_index": word_end_index,
        "source_start": source_start,
        "source_end": source_end,
        "source_text": text,
    }


def caption_fingerprint_for(final: FinalClipEvidence) -> str:
    return caption_source_fingerprint(
        _caption_payload(
            final_transcript=final.final_transcript,
            word_timestamps=[word.as_dict() for word in final.words],
            dialect_profile=final.dialect_profile,
            dialect_confidence=final.dialect_confidence,
            code_switch_evidence=final.code_switch_evidence,
            final_refinement_output_fingerprint=final.output_fingerprint,
        )
    )


def add_final(
    session: Any,
    fixture: Stage50Fixture,
    *,
    transcript: str,
    output_fingerprint: str = "added-final-fp",
    status: RefinementStatus = RefinementStatus.FINAL_TRANSCRIPT_READY,
    refined_start: float | None = None,
    refined_end: float | None = None,
) -> Any:
    """Add a FINAL_CLIP refinement row for a candidate-grade planning fixture."""

    base = fixture.selection.refinement
    row = CandidateRefinement(
        source_video_id=base.source_video_id,
        clip_candidate_id=base.clip_candidate_id,
        priority=RefinementPriority.FINAL_CLIP,
        status=status,
        coarse_start=base.coarse_start,
        coarse_end=base.coarse_end,
        context_start=base.context_start,
        context_end=base.context_end,
        refined_start=base.refined_start if refined_start is None else refined_start,
        refined_end=base.refined_end if refined_end is None else refined_end,
        automatic_transcript=transcript,
        final_transcript=transcript,
        word_timestamps=list(base.word_timestamps or []),
        confidence=0.95,
        quality_level="FINAL_CLIP",
        dialect_profile=base.dialect_profile,
        dialect_confidence=base.dialect_confidence,
        output_fingerprint=output_fingerprint,
    )
    session.add(row)
    session.flush()
    return row


__all__ = [
    "FakeProber",
    "SAMPLE_METADATA",
    "Stage50Fixture",
    "add_final",
    "assert_selected",
    "caption_fingerprint_for",
    "clip_words",
    "corrupt_prober",
    "install_selection_settings",
    "make_final",
    "managed_source",
    "no_video_prober",
    "seed_stage50",
    "selected_status",
    "source_block",
]
