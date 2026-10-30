"""Focused operator candidate registration and persistence-ownership tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from registration_support import (
    EXPECTED_KEY,
    OPERATOR_REASON,
    SELECTED_END,
    SELECTED_INDEXES,
    SELECTED_START,
    SOURCE_ID,
    build_segments,
    install_analysis,
    install_source,
    make_candidate,
    make_draft,
    make_outcome,
)
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from app.candidates.executor import CandidateAnalysisExecutor
from app.candidates.fingerprints import candidate_key
from app.candidates.policy import DEFAULT_CONFIG
from app.candidates.registration import (
    CandidateRegistrationConflict,
    CandidateRegistrationError,
    register_operator_candidate,
)
from app.cli import app
from app.core.enums import (
    CandidateDisposition,
    PipelineStage,
    RefinementPriority,
)
from app.core.settings import get_settings
from app.db.base import Base
from app.models import (
    CandidateAnalysis,
    CandidateRefinement,
    ClipCandidate,
    ProcessingJob,
)
from app.refinement.audio_window import CandidateAudioWindowService
from app.refinement.policy import DEFAULT_CONFIG as REFINEMENT_CONFIG
from app.refinement.queue import validate_candidate_for_refinement
from app.services.storage import StorageService


@pytest.fixture  # type: ignore[untyped-decorator]
def engine(tmp_path: Path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'registration.sqlite3'}")
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture  # type: ignore[untyped-decorator]
def factory(engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(factory: sessionmaker[Session]) -> Iterator[Session]:
    with factory() as session:
        yield session


def _count(session: Session, model: object) -> int:
    return int(session.scalar(select(func.count()).select_from(model)) or 0)


def test_register_operator_window_persists_exact_binding_and_marker(
    session: Session,
) -> None:
    source, transcript = install_source(session)
    analysis = install_analysis(session, source.id)
    other = make_candidate(
        session,
        source_id=source.id,
        candidate_key="existing-automatic-key",
        start_time=365.72,
        end_time=401.46,
        start_segment_index=220,
        end_segment_index=239,
        analysis_id=analysis.id,
        transcript_excerpt="existing automatic candidate",
    )
    other_snapshot = (other.id, other.candidate_key, other.is_current, other.start_time)
    lifecycle_before = source.lifecycle_state
    analysis_before = analysis.output_fingerprint
    refinement_count = _count(session, CandidateRefinement)
    job_count = _count(session, ProcessingJob)

    result = register_operator_candidate(
        session,
        source.id,
        SELECTED_START,
        SELECTED_END,
        OPERATOR_REASON,
    )
    session.commit()
    row = result.candidate

    assert result.created is True
    assert result.ownership == "OPERATOR"
    assert row.id is not None
    assert row.candidate_key == EXPECTED_KEY
    assert row.candidate_key == candidate_key(source.id, 230, 245, span_start=230, span_end=245)
    assert row.source_video_id == source.id
    assert row.start_time == pytest.approx(SELECTED_START)
    assert row.end_time == pytest.approx(SELECTED_END)
    assert row.start_segment_index == 230
    assert row.end_segment_index == 245
    assert row.segment_indexes == SELECTED_INDEXES
    assert row.is_current is True
    assert row.disposition is CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT
    assert row.candidate_analysis_id is None
    assert row.provider_evidence == {}
    assert row.provider_input_fingerprint == ""
    assert row.analysis_fingerprint
    assert row.transcript_excerpt
    assert row.primary_content_type is not None
    assert row.evidence_snapshot["start_segment_index"] == 230
    assert row.evidence_snapshot["end_segment_index"] == 245

    marker = row.provenance_snapshot["operator_selection"]
    assert marker["version"] == "operator-candidate-selection-v1"
    assert marker["origin"] == "OPERATOR"
    assert marker["source_id"] == str(source.id)
    assert marker["transcript_id"] == str(transcript.id)
    assert marker["transcript_input_fingerprint"] == transcript.input_fingerprint
    assert marker["transcription_revision"] == transcript.transcription_revision
    assert marker["start_time"] == pytest.approx(SELECTED_START)
    assert marker["end_time"] == pytest.approx(SELECTED_END)
    assert marker["start_segment_index"] == 230
    assert marker["end_segment_index"] == 245
    assert marker["segment_indexes"] == SELECTED_INDEXES
    assert marker["reason"] == OPERATOR_REASON
    assert marker["audio_artifact_content_hash"] == "operator-audio-content-hash"
    assert marker["audio_artifact_source_content_hash"] == "operator-audio-source-content-hash"
    assert marker["raw_span_fingerprint"]

    session.refresh(other)
    assert (other.id, other.candidate_key, other.is_current, other.start_time) == other_snapshot
    assert source.lifecycle_state is lifecycle_before
    assert analysis.output_fingerprint == analysis_before
    assert _count(session, CandidateRefinement) == refinement_count
    assert _count(session, ProcessingJob) == job_count
    assert _count(session, ClipCandidate) == 2


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("start", "end", "code"),
    [
        (SELECTED_START + 0.001, SELECTED_END, "start_bound_not_unique"),
        (SELECTED_START, SELECTED_END - 0.001, "end_bound_not_unique"),
        (SELECTED_END, SELECTED_START, "invalid_bounds"),
        (float("nan"), SELECTED_END, "invalid_bounds"),
        (SELECTED_START, float("inf"), "invalid_bounds"),
    ],
)
def test_registration_rejects_invalid_or_nonaligned_bounds(
    session: Session,
    start: float,
    end: float,
    code: str,
) -> None:
    install_source(session)
    before = _count(session, ClipCandidate)

    with pytest.raises(CandidateRegistrationError) as captured:
        register_operator_candidate(session, SOURCE_ID, start, end, OPERATOR_REASON)
    session.rollback()

    assert captured.value.code == code
    assert _count(session, ClipCandidate) == before


def test_registration_rejects_ambiguous_start_bound(session: Session) -> None:
    segments = build_segments()
    segments[231]["start"] = SELECTED_START
    install_source(session, segments=segments)
    before = _count(session, ClipCandidate)

    with pytest.raises(CandidateRegistrationError) as captured:
        register_operator_candidate(
            session, SOURCE_ID, SELECTED_START, SELECTED_END, OPERATOR_REASON
        )
    session.rollback()

    assert captured.value.code == "start_bound_not_unique"
    assert _count(session, ClipCandidate) == before


def test_registration_rejects_noncontiguous_segment_index_metadata(
    session: Session,
) -> None:
    segments = build_segments(with_index=True)
    segments[240]["index"] = 999
    install_source(session, segments=segments)

    with pytest.raises(CandidateRegistrationError) as captured:
        register_operator_candidate(
            session, SOURCE_ID, SELECTED_START, SELECTED_END, OPERATOR_REASON
        )
    session.rollback()

    assert captured.value.code == "segment_index_mismatch"


def test_registration_rejects_edge_straddling_word(session: Session) -> None:
    segments = build_segments()
    words = segments[245]["words"]
    assert isinstance(words, list)
    words[-1]["end"] = SELECTED_END + 0.001
    install_source(session, segments=segments)

    with pytest.raises(CandidateRegistrationError) as captured:
        register_operator_candidate(
            session, SOURCE_ID, SELECTED_START, SELECTED_END, OPERATOR_REASON
        )
    session.rollback()

    assert captured.value.code == "edge_straddling_word"


def test_registration_rejects_out_of_source_and_invalid_reason(
    session: Session,
) -> None:
    install_source(session, duration=400.0)

    with pytest.raises(CandidateRegistrationError) as captured:
        register_operator_candidate(
            session, SOURCE_ID, SELECTED_START, SELECTED_END, OPERATOR_REASON
        )
    assert captured.value.code == "invalid_bounds"
    session.rollback()

    with pytest.raises(CandidateRegistrationError) as reason_error:
        register_operator_candidate(session, SOURCE_ID, SELECTED_START, SELECTED_END, " ")
    assert reason_error.value.code == "invalid_reason"


def test_registration_accepts_valid_silent_gaps(session: Session) -> None:
    source, _ = install_source(session)
    result = register_operator_candidate(
        session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
    )
    session.commit()
    assert result.created is True


def test_registration_conflicts_on_same_key_different_window(session: Session) -> None:
    source, _ = install_source(session)
    existing = make_candidate(
        session,
        source_id=source.id,
        candidate_key=EXPECTED_KEY,
        start_time=365.72,
        end_time=401.46,
        start_segment_index=220,
        end_segment_index=239,
        transcript_excerpt="different automatic window",
    )
    before = (
        existing.start_time,
        existing.end_time,
        existing.start_segment_index,
        existing.end_segment_index,
        list(existing.segment_indexes),
        existing.transcript_excerpt,
    )

    with pytest.raises(CandidateRegistrationConflict) as captured:
        register_operator_candidate(
            session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
        )
    session.rollback()

    assert captured.value.code == "candidate_identity_conflict"
    session.refresh(existing)
    assert (
        existing.start_time,
        existing.end_time,
        existing.start_segment_index,
        existing.end_segment_index,
        list(existing.segment_indexes),
        existing.transcript_excerpt,
    ) == before
    assert _count(session, ClipCandidate) == 1


def test_repeat_registration_is_idempotent_from_fresh_session(
    factory: sessionmaker[Session],
) -> None:
    with factory() as session:
        source, _ = install_source(session)
        first = register_operator_candidate(
            session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
        )
        session.commit()
        first_id = first.candidate.id
        marker_before = dict(first.candidate.provenance_snapshot["operator_selection"])
        fingerprint_before = first.candidate.analysis_fingerprint

    with factory() as session:
        second = register_operator_candidate(
            session, SOURCE_ID, SELECTED_START, SELECTED_END, OPERATOR_REASON
        )
        session.commit()
        assert second.created is False
        assert second.candidate.id == first_id
        assert second.ownership == "OPERATOR"
        assert dict(second.candidate.provenance_snapshot["operator_selection"]) == marker_before
        assert second.candidate.analysis_fingerprint == fingerprint_before
        assert _count(session, ClipCandidate) == 1


def test_stale_operator_row_conflicts_without_reactivation(session: Session) -> None:
    source, _ = install_source(session)
    result = register_operator_candidate(
        session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
    )
    session.commit()
    result.candidate.is_current = False
    session.commit()

    with pytest.raises(CandidateRegistrationConflict) as captured:
        register_operator_candidate(
            session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
        )
    session.rollback()

    assert captured.value.code == "operator_candidate_stale"
    session.refresh(result.candidate)
    assert result.candidate.is_current is False


def test_changed_raw_evidence_conflicts_with_operator_row(session: Session) -> None:
    source, transcript = install_source(session)
    result = register_operator_candidate(
        session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
    )
    session.commit()
    marker_before = dict(result.candidate.provenance_snapshot["operator_selection"])
    segments = [dict(segment) for segment in transcript.segments]
    segments[230]["raw_text"] = "changed raw text"
    transcript.segments = segments
    session.commit()

    with pytest.raises(CandidateRegistrationConflict) as captured:
        register_operator_candidate(
            session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
        )
    session.rollback()

    assert captured.value.code == "operator_raw_identity_conflict"
    session.refresh(result.candidate)
    assert result.candidate.is_current is True
    assert dict(result.candidate.provenance_snapshot["operator_selection"]) == marker_before


def test_exact_current_automatic_row_is_reused_without_operator_pin(
    session: Session,
) -> None:
    source, transcript = install_source(session)
    analysis = install_analysis(session, source.id)
    excerpt = " ".join(
        str(transcript.segments[index]["raw_text"]) for index in SELECTED_INDEXES
    ).strip()
    automatic = make_candidate(
        session,
        source_id=source.id,
        candidate_key=EXPECTED_KEY,
        start_time=SELECTED_START,
        end_time=SELECTED_END,
        start_segment_index=230,
        end_segment_index=245,
        analysis_id=analysis.id,
        transcript_excerpt=excerpt,
    )

    result = register_operator_candidate(
        session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
    )
    session.commit()

    assert result.created is False
    assert result.ownership == "AUTOMATIC"
    assert result.candidate.id == automatic.id
    assert result.candidate.candidate_analysis_id == analysis.id
    assert "operator_selection" not in result.candidate.provenance_snapshot
    assert _count(session, ClipCandidate) == 1


def test_cli_registers_and_rejects_without_partial_writes(tmp_path: Path) -> None:
    get_settings.cache_clear()
    engine = create_engine(get_settings().database_url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        source, _ = install_source(session)
        source_id = source.id

    runner = CliRunner()
    success = runner.invoke(
        app,
        [
            "candidate-register-window",
            str(source_id),
            "--start",
            "384.280",
            "--end",
            "408.660",
            "--reason",
            OPERATOR_REASON,
        ],
    )
    assert success.exit_code == 0, success.output
    payload = json.loads(success.stdout)
    assert payload["candidate_key"] == EXPECTED_KEY
    assert payload["created"] is True
    assert payload["ownership"] == "OPERATOR"
    assert payload["start_segment_index"] == 230
    assert payload["end_segment_index"] == 245
    assert payload["segment_indexes"] == SELECTED_INDEXES
    assert payload["disposition"] == CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT.value

    failure = runner.invoke(
        app,
        [
            "candidate-register-window",
            str(source_id),
            "--start",
            "384.281",
            "--end",
            "408.660",
            "--reason",
            OPERATOR_REASON,
        ],
    )
    assert failure.exit_code != 0
    assert "start_bound_not_unique" in failure.output

    with factory() as session:
        assert _count(session, ClipCandidate) == 1
    engine.dispose()


def test_registration_admits_refinement_and_whole_window_context(
    session: Session, tmp_path: Path
) -> None:
    source, transcript = install_source(session)
    result = register_operator_candidate(
        session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
    )
    session.commit()

    admitted = validate_candidate_for_refinement(session, result.candidate.id)
    assert admitted.id == result.candidate.id
    assert admitted.start_time == pytest.approx(SELECTED_START)
    assert admitted.end_time == pytest.approx(SELECTED_END)

    service = CandidateAudioWindowService(
        storage=StorageService(tmp_path / "storage"),
        session=session,
        config=REFINEMENT_CONFIG,
        ffmpeg_binary="ffmpeg",
        command_runner=lambda _args: None,
    )
    context_start, context_end = service.context_bounds(
        coarse_start=admitted.start_time,
        coarse_end=admitted.end_time,
        source_duration=transcript.duration,
        priority=RefinementPriority.FINAL_CLIP,
    )
    assert context_start == pytest.approx(SELECTED_START - 8.0)
    assert context_end == pytest.approx(SELECTED_END + 8.0)


def test_persist_preserves_nonemitted_operator_and_stales_automatic(
    session: Session,
) -> None:
    source, transcript = install_source(session)
    result = register_operator_candidate(
        session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
    )
    session.commit()
    operator_id = result.candidate.id
    marker_before = dict(result.candidate.provenance_snapshot["operator_selection"])
    automatic = make_candidate(
        session,
        source_id=source.id,
        candidate_key="nonemitted-automatic-key",
        start_time=100.0,
        end_time=110.0,
        start_segment_index=100,
        end_segment_index=110,
        transcript_excerpt="non-emitted automatic",
    )

    executor = CandidateAnalysisExecutor(session=session, config=DEFAULT_CONFIG)
    executor._persist(source, transcript, make_outcome(), "input-fp", 0.1)

    session.refresh(result.candidate)
    session.refresh(automatic)
    assert result.candidate.id == operator_id
    assert result.candidate.is_current is True
    assert dict(result.candidate.provenance_snapshot["operator_selection"]) == marker_before
    assert automatic.is_current is False
    analysis = session.scalar(
        select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
    )
    assert analysis is not None
    assert analysis.metrics["operator_rows_preserved"] == 1
    assert analysis.metrics["automatic_candidates_applied"] == 0


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "same_scope", [True, False]
)
def test_persist_suppresses_same_key_operator_draft_and_accounts(
    session: Session, same_scope: bool
) -> None:
    source, transcript = install_source(session)
    result = register_operator_candidate(
        session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
    )
    session.commit()
    operator_id = result.candidate.id
    marker_before = dict(result.candidate.provenance_snapshot["operator_selection"])
    if same_scope:
        draft = make_draft(
            candidate_key=EXPECTED_KEY,
            start_segment_index=230,
            end_segment_index=245,
            start_time=SELECTED_START,
            end_time=SELECTED_END,
            text="same scope automatic draft",
        )
    else:
        draft = make_draft(
            candidate_key=EXPECTED_KEY,
            start_segment_index=220,
            end_segment_index=239,
            start_time=365.72,
            end_time=401.46,
            text="colliding automatic draft",
        )

    executor = CandidateAnalysisExecutor(session=session, config=DEFAULT_CONFIG)
    executor._persist(source, transcript, make_outcome(draft), "input-fp", 0.1)

    session.refresh(result.candidate)
    assert result.candidate.id == operator_id
    assert result.candidate.is_current is True
    assert result.candidate.start_time == pytest.approx(SELECTED_START)
    assert result.candidate.end_time == pytest.approx(SELECTED_END)
    assert result.candidate.start_segment_index == 230
    assert result.candidate.end_segment_index == 245
    assert result.candidate.transcript_excerpt != "same scope automatic draft"
    assert result.candidate.transcript_excerpt != "colliding automatic draft"
    assert dict(result.candidate.provenance_snapshot["operator_selection"]) == marker_before
    assert _count(session, ClipCandidate) == 1

    analysis = session.scalar(
        select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
    )
    assert analysis is not None
    assert analysis.metrics["operator_rows_preserved"] == 1
    assert analysis.metrics["automatic_candidates_applied"] == 0
    assert analysis.metrics["automatic_drafts_suppressed"] == 1
    if same_scope:
        assert analysis.metrics["operator_same_key_drafts_matched"] == 1
        assert analysis.metrics["operator_same_key_drafts_collided"] == 0
    else:
        assert analysis.metrics["operator_same_key_drafts_matched"] == 0
        assert analysis.metrics["operator_same_key_drafts_collided"] == 1


def test_persist_stales_operator_on_raw_change_but_not_normalization(
    session: Session,
) -> None:
    source, transcript = install_source(session)
    result = register_operator_candidate(
        session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON
    )
    session.commit()
    executor = CandidateAnalysisExecutor(session=session, config=DEFAULT_CONFIG)

    segments = [dict(segment) for segment in transcript.segments]
    segments[240]["normalized_text"] = "derived normalization change only"
    segments[240]["corrected_text"] = "derived correction change only"
    segments[240]["final_text"] = "derived final change only"
    transcript.segments = segments
    session.commit()
    executor._persist(source, transcript, make_outcome(), "input-fp", 0.1)
    session.refresh(result.candidate)
    assert result.candidate.is_current is True

    segments = [dict(segment) for segment in transcript.segments]
    segments[240]["raw_text"] = "changed immutable raw text"
    transcript.segments = segments
    session.commit()
    executor._persist(source, transcript, make_outcome(), "input-fp", 0.1)
    session.refresh(result.candidate)
    assert result.candidate.is_current is False
    analysis = session.scalar(
        select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
    )
    assert analysis is not None
    assert analysis.metrics["operator_rows_staled_raw_identity"] == 1


def test_registration_leaves_other_rows_and_lifecycle_untouched(session: Session) -> None:
    source, _ = install_source(session, lifecycle=PipelineStage.READY_FOR_REFINEMENT)
    lifecycle_before = source.lifecycle_state
    other = make_candidate(
        session,
        source_id=source.id,
        candidate_key="other-key",
        start_time=10.0,
        end_time=20.0,
        start_segment_index=10,
        end_segment_index=20,
        transcript_excerpt="other",
    )
    snapshot = (other.id, other.start_time, other.end_time, other.is_current)
    register_operator_candidate(session, source.id, SELECTED_START, SELECTED_END, OPERATOR_REASON)
    session.commit()
    session.refresh(other)
    session.refresh(source)
    assert (other.id, other.start_time, other.end_time, other.is_current) == snapshot
    assert source.lifecycle_state is lifecycle_before
