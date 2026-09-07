import json

import pytest

from app.transcription.reconstruction.providers import ReconstructionRequest
from app.transcription.reconstruction.routing import (
    RoutingDecision,
    RoutingEvidence,
    RoutingPriority,
)
from app.transcription.reconstruction.types import (
    AcousticEvidence,
    WordEvidence,
)


def _request(index: int, text: str = "هدف") -> ReconstructionRequest:
    return ReconstructionRequest(
        segment_index=index,
        raw_text=text,
        corrected_text="تصحيح",
        previous=("قبل",),
        following=("بعد",),
        word_evidence=(
            WordEvidence("word1", 1.0, 2.0, 0.45),
            WordEvidence("word2", 2.0, 3.0, 0.95),
        ),
        acoustic=AcousticEvidence(0.5, 0.5, -0.5, 0.1),
        entities=("أحمد",),
        routing_reasons=("multiple_low_probability_words",),
        focus_spans=(WordEvidence("word1", 1.0, 2.0, 0.45),),
        language="ar",
    )


def test_reconstruction_request_serializes_small_local_context() -> None:
    request = _request(4)
    payload = request.to_payload()
    assert payload["segment_id"] == 4
    assert payload["language"] == "ar"
    assert payload["entities"] == ["أحمد"]
    assert payload["routing_reasons"] == ["multiple_low_probability_words"]
    assert payload["raw_text"] == "هدف"
    assert payload["corrected_text"] == "تصحيح"
    assert payload["previous"] == ["قبل"]
    assert payload["following"] == ["بعد"]
    assert len(payload["words"]) == 2
    assert payload["focus_spans"][0]["text"] == "word1"


def test_reconstruction_request_estimates_tokens_for_budgeting() -> None:
    request = _request(4)
    tokens = request.estimated_tokens()
    payload = json.dumps(request.to_payload(), ensure_ascii=False)
    assert tokens == len(payload) // 2
    assert tokens > 0


def test_reconstruction_request_payload_does_not_include_full_transcript() -> None:
    request = _request(4)
    payload = request.to_payload()
    assert "window" not in payload
    assert "full_transcript" not in payload
    assert len(payload["previous"]) <= 2
    assert len(payload["following"]) <= 2
