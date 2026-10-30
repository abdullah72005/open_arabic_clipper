"""Pure FINAL_CLIP context-versus-selected-evidence regressions.

The captured 107-record decoder response is used as an offline fixture only.
These tests never load a model or call a provider; the real targeted ASR adapter
is exercised over a fake Whisper engine so the observation shape is the one the
production producer persists.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.core.enums import RefinementPriority
from app.refinement.asr import TargetedASREngine
from app.refinement.policy import FINAL_CLIP_SCOPE_POLICY_VERSION
from app.refinement.scope import (
    SCOPE_REASON_CONTEXT_ESCAPE,
    SCOPE_REASON_EDGE_WORD,
    SCOPE_REASON_MALFORMED,
    SCOPE_REASON_NON_FINITE,
    SCOPE_REASON_OBSERVATIONS_MISSING,
    SCOPE_REASON_REVERSED,
    SCOPE_REASON_ZERO_INSIDE,
    FinalClipView,
    ScopeRefusal,
    derive_final_clip_view,
    raw_observation_fingerprint,
)
from app.refinement.types import TargetASRResult
from app.transcription.engine import TranscriptionResult
from app.transcription.service import TranscriptionOptions

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "final_clip_scope"
_CONTEXT_START = 376.28
_CONTEXT_END = 416.66
_CLIP_START = 384.29999999999995
_CLIP_END = 408.67999999999995


class _CapturedWhisper:
    def __init__(self, result: TranscriptionResult) -> None:
        self._result = result

    def transcribe(self, audio_path, options, cancel_event=None):  # type: ignore[no-untyped-def]
        del audio_path, options, cancel_event
        return self._result


def _payload(name: str) -> dict[str, object]:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _captured_result() -> TranscriptionResult:
    response = _payload("decoder-result.json")["response"]
    return TranscriptionResult(
        language=response["language"],
        language_probability=response["language_probability"],
        raw_text=response["raw_text"],
        duration=response["duration"],
        segments=response["segments"],
        word_segments=response["word_segments"],
    )


def _targeted() -> TargetASRResult:
    return TargetedASREngine(
        options_for=lambda _priority: TranscriptionOptions(
            model="large-v3-turbo",
            device="cpu",
            compute_type="int8",
            beam_size=8,
        ),
        engine=_CapturedWhisper(_captured_result()),
    ).transcribe(
        Path("window.wav"),
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        priority=RefinementPriority.FINAL_CLIP,
    )


def _derive(target: TargetASRResult, *, clip_start=_CLIP_START, clip_end=_CLIP_END):
    return derive_final_clip_view(
        raw_observations=list(target.raw_word_observations),
        record_count=target.raw_word_record_count,
        observations_complete=target.raw_word_observations_complete,
        observation_fingerprint=target.raw_word_observations_fingerprint,
        transcript=target.transcript,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        clip_start=clip_start,
        clip_end=clip_end,
    )


def _observations(target: TargetASRResult) -> list[dict[str, object]]:
    return copy.deepcopy(list(target.raw_word_observations))


def _fingerprint(observations: list[dict[str, object]], record_count: int) -> str:
    return raw_observation_fingerprint(
        raw_observations=observations,
        record_count=record_count,
        observations_complete=True,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
    )


def _set_point(observation: dict[str, object], point: float) -> None:
    observation["relative_start"] = point - _CONTEXT_START
    observation["relative_end"] = point - _CONTEXT_START
    observation["source_start"] = point
    observation["source_end"] = point


def test_captured_context_rejects_but_selected_view_is_valid() -> None:
    target = _targeted()

    assert target.rejected_reason == "nonpositive word duration"
    assert len(target.word_timestamps) == 106
    assert target.raw_word_record_count == 107
    assert target.raw_word_observations_complete is True
    assert len(target.raw_word_observations) == 107

    view = _derive(target)
    assert isinstance(view, FinalClipView)
    assert view.policy_version == FINAL_CLIP_SCOPE_POLICY_VERSION
    assert view.state == "ACCEPTED"
    assert (view.clip_start, view.clip_end) == (_CLIP_START, _CLIP_END)
    assert view.context_only_invalid_record_indexes == (105,)
    assert len(view.words) == 67
    assert [word.parent_record_index for word in view.words] == list(range(17, 84))
    assert len(view.text.split()) == 67

    actual = _payload("result.json")["selected_words"]
    assert len(actual) == 67
    for word, expected in zip(view.words, actual, strict=True):
        assert word.text == expected["text"]
        assert word.start == expected["start"]
        assert word.end == expected["end"]
        assert word.probability == expected["probability"]

    denial = [word for word in view.words if 402.4 <= word.start < 403.35]
    assert [word.text for word in denial] == [" لا", " مش", " هاسبه"]
    assert [round(word.start, 3) for word in denial] == [402.42, 402.72, 402.96]
    assert [round(word.end, 3) for word in denial] == [402.72, 402.96, 403.36]


def test_positive_context_words_are_preserved_but_never_selected() -> None:
    target = _targeted()
    view = _derive(target)
    assert isinstance(view, FinalClipView)

    context_only = [item for item in target.raw_word_observations if item["record_index"] == 0]
    assert context_only and context_only[0]["text"].strip() == "معي"
    assert "معي" not in view.text.split()
    assert 0 not in [word.parent_record_index for word in view.words]


@pytest.mark.parametrize("point", [_CLIP_START, _CLIP_END, 404.0])
def test_zero_duration_inside_or_on_clip_refuses(point: float) -> None:
    target = _targeted()
    observations = _observations(target)
    _set_point(observations[105], point)

    result = derive_final_clip_view(
        raw_observations=observations,
        record_count=target.raw_word_record_count,
        observations_complete=True,
        observation_fingerprint=_fingerprint(observations, target.raw_word_record_count),
        transcript=target.transcript,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        clip_start=_CLIP_START,
        clip_end=_CLIP_END,
    )

    assert isinstance(result, ScopeRefusal)
    assert result.reason == SCOPE_REASON_ZERO_INSIDE
    assert {item["record_index"] for item in result.violations} == {105}


def test_reversed_non_finite_malformed_and_escape_records_refuse() -> None:
    target = _targeted()
    cases: list[tuple[str, dict[str, object], str]] = []

    reversed_record = copy.deepcopy(target.raw_word_observations[17])
    reversed_record["source_end"] = float(reversed_record["source_start"]) - 0.1
    reversed_record["relative_end"] = float(reversed_record["relative_start"]) - 0.1
    cases.append(("reversed", reversed_record, SCOPE_REASON_REVERSED))

    non_finite_record = copy.deepcopy(target.raw_word_observations[17])
    non_finite_record["source_start"] = "nan"
    cases.append(("non_finite", non_finite_record, SCOPE_REASON_NON_FINITE))

    malformed_record = copy.deepcopy(target.raw_word_observations[17])
    malformed_record["relative_start"] = "not-a-number"
    cases.append(("malformed", malformed_record, SCOPE_REASON_MALFORMED))

    escape_record = copy.deepcopy(target.raw_word_observations[17])
    escape_record["source_start"] = 420.0
    escape_record["source_end"] = 420.4
    escape_record["relative_start"] = 420.0 - _CONTEXT_START
    escape_record["relative_end"] = 420.4 - _CONTEXT_START
    cases.append(("escape", escape_record, SCOPE_REASON_CONTEXT_ESCAPE))

    for name, replacement, expected_reason in cases:
        observations = _observations(target)
        observations[17] = replacement
        result = derive_final_clip_view(
            raw_observations=observations,
            record_count=target.raw_word_record_count,
            observations_complete=True,
            observation_fingerprint=_fingerprint(observations, target.raw_word_record_count),
            transcript=target.transcript,
            context_start=_CONTEXT_START,
            context_end=_CONTEXT_END,
            clip_start=_CLIP_START,
            clip_end=_CLIP_END,
        )
        assert isinstance(result, ScopeRefusal), name
        assert result.reason == expected_reason, name
        assert any(item["record_index"] == 17 for item in result.violations), name


def test_nonmonotonic_timing_refuses() -> None:
    target = _targeted()
    observations = _observations(target)
    observations[18]["source_start"] = 384.2
    observations[18]["source_end"] = 384.5
    observations[18]["relative_start"] = 384.2 - _CONTEXT_START
    observations[18]["relative_end"] = 384.5 - _CONTEXT_START

    result = derive_final_clip_view(
        raw_observations=observations,
        record_count=target.raw_word_record_count,
        observations_complete=True,
        observation_fingerprint=_fingerprint(observations, target.raw_word_record_count),
        transcript=target.transcript,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        clip_start=_CLIP_START,
        clip_end=_CLIP_END,
    )

    assert isinstance(result, ScopeRefusal)
    assert result.reason == "nonmonotonic_word_timestamps"
    assert any(item["record_index"] == 18 for item in result.violations)


def test_partial_word_at_clip_edge_refuses() -> None:
    target = _targeted()
    observations = _observations(target)
    observations[17]["source_start"] = _CLIP_START - 0.01
    observations[17]["relative_start"] = _CLIP_START - 0.01 - _CONTEXT_START

    result = derive_final_clip_view(
        raw_observations=observations,
        record_count=target.raw_word_record_count,
        observations_complete=True,
        observation_fingerprint=_fingerprint(observations, target.raw_word_record_count),
        transcript=target.transcript,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        clip_start=_CLIP_START,
        clip_end=_CLIP_END,
    )

    assert isinstance(result, ScopeRefusal)
    assert result.reason == SCOPE_REASON_EDGE_WORD
    assert {item["record_index"] for item in result.violations} == {17}


def test_missing_or_incomplete_observations_refuse() -> None:
    target = _targeted()
    missing = derive_final_clip_view(
        raw_observations=_observations(target)[:-1],
        record_count=target.raw_word_record_count,
        observations_complete=True,
        observation_fingerprint=target.raw_word_observations_fingerprint,
        transcript=target.transcript,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        clip_start=_CLIP_START,
        clip_end=_CLIP_END,
    )
    assert isinstance(missing, ScopeRefusal)
    assert missing.reason == SCOPE_REASON_OBSERVATIONS_MISSING

    incomplete = derive_final_clip_view(
        raw_observations=_observations(target),
        record_count=target.raw_word_record_count,
        observations_complete=False,
        observation_fingerprint=target.raw_word_observations_fingerprint,
        transcript=target.transcript,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        clip_start=_CLIP_START,
        clip_end=_CLIP_END,
    )
    assert isinstance(incomplete, ScopeRefusal)
    assert incomplete.reason == "raw_observations_incomplete"


def test_every_invalid_record_is_classified_and_refuses() -> None:
    target = _targeted()
    observations = _observations(target)
    # Record 0: finite zero-duration, inside context, strictly before the clip.
    _set_point(observations[0], 376.5)
    # Record 105: the same shape moved inside the clip.
    _set_point(observations[105], 404.0)

    result = derive_final_clip_view(
        raw_observations=observations,
        record_count=target.raw_word_record_count,
        observations_complete=True,
        observation_fingerprint=_fingerprint(observations, target.raw_word_record_count),
        transcript=target.transcript,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        clip_start=_CLIP_START,
        clip_end=_CLIP_END,
    )

    assert isinstance(result, ScopeRefusal)
    assert result.reason == SCOPE_REASON_ZERO_INSIDE
    assert {item["record_index"] for item in result.violations} == {105}

    # A later unknown-location record is still classified alongside the inside
    # zero-duration record; the first blocking row is not the only one reported.
    observations[2]["source_start"] = "not-finite"
    combined = derive_final_clip_view(
        raw_observations=observations,
        record_count=target.raw_word_record_count,
        observations_complete=True,
        observation_fingerprint=_fingerprint(observations, target.raw_word_record_count),
        transcript=target.transcript,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        clip_start=_CLIP_START,
        clip_end=_CLIP_END,
    )
    assert isinstance(combined, ScopeRefusal)
    assert {item["record_index"] for item in combined.violations} == {2, 105}


def test_unaccounted_transcript_text_refuses() -> None:
    target = _targeted()
    result = derive_final_clip_view(
        raw_observations=_observations(target),
        record_count=target.raw_word_record_count,
        observations_complete=True,
        observation_fingerprint=target.raw_word_observations_fingerprint,
        transcript=target.transcript + " زائد",
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        clip_start=_CLIP_START,
        clip_end=_CLIP_END,
    )
    assert isinstance(result, ScopeRefusal)
    assert result.reason == "unaccounted_transcript_text"


def test_view_fingerprint_binds_bounds_observation_and_mapping() -> None:
    target = _targeted()
    first = _derive(target)
    second = _derive(target)
    assert isinstance(first, FinalClipView) and isinstance(second, FinalClipView)
    assert first.fingerprint == second.fingerprint

    moved = _derive(target, clip_end=_CLIP_END + 0.01)
    assert isinstance(moved, ScopeRefusal)
    assert moved.reason == SCOPE_REASON_EDGE_WORD

    observations = _observations(target)
    _set_point(observations[105], 410.0)
    changed = derive_final_clip_view(
        raw_observations=observations,
        record_count=target.raw_word_record_count,
        observations_complete=True,
        observation_fingerprint=_fingerprint(observations, target.raw_word_record_count),
        transcript=target.transcript,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        clip_start=_CLIP_START,
        clip_end=_CLIP_END,
    )
    assert isinstance(changed, FinalClipView)
    assert changed.context_only_invalid_record_indexes == (105,)
    assert changed.fingerprint != first.fingerprint
