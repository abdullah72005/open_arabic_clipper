"""Deterministic tests for Stage 3.5 evidence and entity logic."""

from app.core.enums import EvidenceKind, EvidenceState
from app.refinement.entities import (
    EntityType,
    compare_entity_sets,
    extract_entities,
    normalize_entity,
)
from app.refinement.evidence import (
    choose_final_transcript,
    dedupe_evidence,
    select_consensus_text,
    validate_transcript_candidate,
)
from app.refinement.types import EvidenceRecord


def _record(
    *,
    kind: EvidenceKind,
    fingerprint: str,
    transcript: str,
    confidence: float,
    state: EvidenceState = EvidenceState.ACCEPTED,
) -> EvidenceRecord:
    return EvidenceRecord(
        kind=kind,
        fingerprint=fingerprint,
        provider="test",
        model="test-model",
        settings={},
        window_start=0.0,
        window_end=1.0,
        transcript=transcript,
        confidence=confidence,
        state=state,
    )


def test_dedupe_evidence_dedupes_by_fingerprint_and_bounds() -> None:
    records = [
        _record(
            kind=EvidenceKind.INDEX_RAW,
            fingerprint=f"f{index}",
            transcript=f"t{index}",
            confidence=0.5,
        )
        for index in range(5)
    ]
    records.append(
        _record(
            kind=EvidenceKind.INDEX_RAW,
            fingerprint="f0",
            transcript="duplicate",
            confidence=0.9,
        )
    )

    bounded = dedupe_evidence(records, limit=3)

    assert [record.fingerprint for record in bounded] == ["f0", "f1", "f2"]
    assert dedupe_evidence(records, limit=0) == ()


def test_consensus_prefers_agreement_and_penalizes_disagreement() -> None:
    left = _record(
        kind=EvidenceKind.TARGETED_LOCAL_ASR,
        fingerprint="a",
        transcript="أنا عملت deploy امبارح",
        confidence=0.7,
    )
    right = _record(
        kind=EvidenceKind.HOSTED_ASR,
        fingerprint="b",
        transcript="أنا عملت deploy امبارح",
        confidence=0.6,
    )

    text, confidence, group = select_consensus_text([left, right])

    assert text == "أنا عملت deploy امبارح"
    assert confidence > 0.7
    assert len(group) == 2

    high = _record(
        kind=EvidenceKind.TARGETED_LOCAL_ASR,
        fingerprint="c",
        transcript="أنا عملت حاجة",
        confidence=0.9,
    )
    low = _record(
        kind=EvidenceKind.HOSTED_ASR,
        fingerprint="d",
        transcript="شيء مختلف",
        confidence=0.4,
    )

    text, confidence, group = select_consensus_text([high, low])

    assert text == "أنا عملت حاجة"
    assert confidence < 0.9
    assert len(group) == 1

    assert select_consensus_text([]) == ("", 0.0, ())


def test_text_only_record_cannot_insert_unsupported_latin_term() -> None:
    accepted, reason = validate_transcript_candidate(
        candidate_text="أنا عملت deploy امبارح",
        reference_text="أنا عملت امبارح",
        word_timestamps=(),
        protected_tokens=(),
        source_dialect="EGYPTIAN",
        candidate_dialect="EGYPTIAN",
        max_edit_ratio=0.6,
        min_phonetic_similarity=0.45,
        repeats_max_ratio=0.4,
    )

    assert accepted is False
    assert reason == "unsupported_omitted_english"

    accepted, reason = validate_transcript_candidate(
        candidate_text="أنا عملت deploy امبارح",
        reference_text="أنا عملت deploy امبارح",
        word_timestamps=(),
        protected_tokens=("deploy",),
        source_dialect="EGYPTIAN",
        candidate_dialect="EGYPTIAN",
        max_edit_ratio=0.6,
        min_phonetic_similarity=0.45,
        repeats_max_ratio=0.4,
    )

    assert accepted is True
    assert reason is None


def test_validate_rejects_protected_change_extreme_edit_and_repetition() -> None:
    accepted, reason = validate_transcript_candidate(
        candidate_text="سنة 1955",
        reference_text="سنة 1954",
        word_timestamps=(),
        protected_tokens=("1954",),
        source_dialect=None,
        candidate_dialect=None,
        max_edit_ratio=0.6,
        min_phonetic_similarity=0.0,
        repeats_max_ratio=0.9,
    )
    assert accepted is False
    assert reason == "protected_tokens_changed"

    accepted, reason = validate_transcript_candidate(
        candidate_text="شيء مختلف تماما هنا الآن",
        reference_text="السلام عليكم ورحمة الله وبركاته",
        word_timestamps=(),
        protected_tokens=(),
        source_dialect=None,
        candidate_dialect=None,
        max_edit_ratio=0.2,
        min_phonetic_similarity=0.0,
        repeats_max_ratio=0.9,
    )
    assert accepted is False
    assert reason == "extreme_edit_ratio"

    accepted, reason = validate_transcript_candidate(
        candidate_text="لا لا لا لا لا",
        reference_text="لا",
        word_timestamps=(),
        protected_tokens=(),
        source_dialect=None,
        candidate_dialect=None,
        max_edit_ratio=0.95,
        min_phonetic_similarity=0.0,
        repeats_max_ratio=0.4,
    )
    assert accepted is False
    assert reason == "repeated_or_hallucinated_text"


def test_choose_final_transcript_priority_and_manual_verbatim() -> None:
    assert (
        choose_final_transcript(
            manual="  manual text  ",
            adjudicated="adj",
            operator_segment_text="op",
            consensus="cons",
            automatic="auto",
            stage27_text="s27",
            stage25_text="s25",
            raw_text="raw",
        )
        == "  manual text  "
    )
    assert (
        choose_final_transcript(
            manual=None,
            adjudicated="adj",
            operator_segment_text="op",
            consensus="cons",
            automatic="auto",
            stage27_text="s27",
            stage25_text="s25",
            raw_text="raw",
        )
        == "op"
    )
    assert (
        choose_final_transcript(
            manual="   ",
            adjudicated="adj",
            operator_segment_text="",
            consensus=None,
            automatic=None,
            stage27_text=None,
            stage25_text=None,
            raw_text="raw",
        )
        == "adj"
    )
    assert (
        choose_final_transcript(
            manual=None,
            adjudicated=None,
            operator_segment_text=None,
            consensus=None,
            automatic="auto",
            stage27_text="s27",
            stage25_text="s25",
            raw_text="raw",
        )
        == "auto"
    )
    assert (
        choose_final_transcript(
            manual=None,
            adjudicated=None,
            operator_segment_text=None,
            consensus=None,
            automatic=None,
            stage27_text="s27",
            stage25_text="s25",
            raw_text="raw",
        )
        == "s27"
    )
    assert (
        choose_final_transcript(
            manual=None,
            adjudicated=None,
            operator_segment_text=None,
            consensus=None,
            automatic=None,
            stage27_text=None,
            stage25_text="s25",
            raw_text="raw",
        )
        == "s25"
    )
    assert (
        choose_final_transcript(
            manual=None,
            adjudicated=None,
            operator_segment_text=None,
            consensus=None,
            automatic=None,
            stage27_text=None,
            stage25_text=None,
            raw_text="raw",
        )
        == "raw"
    )
    assert (
        choose_final_transcript(
            manual=None,
            adjudicated=None,
            operator_segment_text=None,
            consensus=None,
            automatic=None,
            stage27_text=None,
            stage25_text=None,
            raw_text=None,
        )
        == ""
    )


def test_entities_preserve_display_and_report_meaning_critical_conflicts() -> None:
    mentions = extract_entities(
        "سنة 1954 وزاد 15",
        start=1.0,
        end=2.0,
        evidence_fingerprints=("fp",),
    )

    assert [mention.text for mention in mentions] == ["1954", "15"]
    assert all(mention.start == 1.0 and mention.end == 2.0 for mention in mentions)
    assert all(mention.evidence_fingerprints == ("fp",) for mention in mentions)

    arabic_indic = extract_entities("سنة ١٩٥٤")
    assert arabic_indic[0].text == "١٩٥٤"
    assert arabic_indic[0].normalized == normalize_entity("1954")

    conflicts = compare_entity_sets(extract_entities("سنة ١٩٥٤"), extract_entities("سنة 1955"))

    assert conflicts
    assert conflicts[0]["entity_type"] == EntityType.DATE.value
    assert conflicts[0]["meaning_critical"] is True

    assert compare_entity_sets(extract_entities("سنة ١٩٥٤"), extract_entities("سنة 1954")) == []
