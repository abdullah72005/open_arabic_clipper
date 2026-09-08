from app.transcription.reconstruction.entities import build_entity_memory
from app.transcription.reconstruction.types import ReconstructionCandidate
from app.transcription.reconstruction.validation import validate_candidate


def test_validator_rejects_mutated_latin_and_numeric_evidence() -> None:
    """A contextual proposal cannot rewrite code-switched names or numbers."""

    memory = build_entity_memory([{"text": "United Fruit Company سنة 1954"}])
    candidate = ReconstructionCandidate("provider-0", "United Fruit Company سنة 1955")

    result = validate_candidate("United Fruit Company سنة 1954", candidate, memory)

    assert result.accepted is False
    assert result.reason == "protected_tokens_changed"


def test_validator_accepts_bounded_multword_candidate_with_source_evidence() -> None:
    """A supported multi-word Egyptian reconstruction remains eligible for contextual ranking."""

    raw = "في الواقع تقل في شركة امريكية دخم اسمها United Fruit Company"
    candidate = ReconstructionCandidate(
        "provider-0",
        "في الوقت اللي فيه شركة أمريكية ضخمة اسمها United Fruit Company",
        evidence_segment_ids=(0,),
    )

    result = validate_candidate(raw, candidate, build_entity_memory([{"text": raw}]))

    assert result.accepted is True
    assert result.phonetic_similarity >= 0.55
