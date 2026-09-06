import json

import pytest

from app.transcription.reconstruction.providers import (
    GenerationRequest,
    batch_generation_requests,
)
from app.transcription.reconstruction.routing import (
    RoutingDecision,
    RoutingEvidence,
    RoutingPriority,
)
from app.transcription.reconstruction.types import (
    AcousticEvidence,
    ReconstructionWindow,
    WindowSegment,
)


def _request(index: int, text: str = "هدف") -> GenerationRequest:
    window = ReconstructionWindow(
        index,
        (
            WindowSegment(
                index,
                1.0,
                2.0,
                text,
                "تصحيح",
                AcousticEvidence(0.5, 0.5, -0.5, 0.1),
                word_evidence=(),
            ),
        ),
    )
    decision = RoutingDecision(
        RoutingPriority.RECONSTRUCT,
        RoutingEvidence(0.9, 0.8, (), "multiple_low_probability_words"),
        (),
        "multiple_low_probability_words",
    )
    return GenerationRequest(
        window=window, language="ar", entity_forms=("أحمد",), routing_decision=decision
    )


def test_generation_request_serializes_complete_immutable_evidence() -> None:
    request = _request(4)
    payload = request.to_payload()
    assert payload["segment_id"] == 4
    assert payload["language"] == "ar"
    assert payload["entities"] == ["أحمد"]
    assert payload["routing"]["reason"] == "multiple_low_probability_words"
    assert "start" in payload["window"][0] and "end" in payload["window"][0]
    assert payload["window"][0]["raw_text"] == "هدف"
    assert payload["window"][0]["corrected_text"] == "تصحيح"


def test_batch_generation_requests_is_utf8_bounded_and_priority_ordered() -> None:
    requests = [_request(9, "ب" * 20), _request(2, "أ" * 20)]
    requests[1] = GenerationRequest(
        window=requests[1].window,
        language="ar",
        entity_forms=(),
        routing_decision=RoutingDecision(
            RoutingPriority.CONTEXT_CHECK, RoutingEvidence(0, 0, (), "check"), (), "check"
        ),
    )
    batches = batch_generation_requests(requests, max_windows=2, max_characters=10_000)
    assert [item.segment_index for item in batches[0]] == [9, 2]
    encoded = json.dumps(
        [item.to_payload() for item in batches[0]], ensure_ascii=False, sort_keys=True
    ).encode()
    assert len(encoded) <= 10_000


def test_batch_generation_requests_rejects_values_above_hard_maxima() -> None:
    with pytest.raises(ValueError):
        batch_generation_requests([], max_windows=17)
    with pytest.raises(ValueError):
        batch_generation_requests([], max_characters=48_001)
