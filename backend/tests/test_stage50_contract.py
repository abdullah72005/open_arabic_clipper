"""Stage 5.0 execution-contract assembly, materialization, and caption input."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import (
    caption_fingerprint_for,
    clip_words,
    make_final,
    seed_stage50,
    source_block,
)

from app.db.base import Base
from app.render.compatibility import evaluate_compatibility
from app.render.fingerprints import build_caption_source_payload
from app.render.service import create_render_contract

BIDI_CONTROLS = {
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
}

MIXED = "النهارده بنتكلم عن remote work وازاي productivity اتأثرت"


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _install(monkeypatch: Any) -> FakeGovernanceSettings:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    return settings


def test_contract_blocks_preserve_order_and_slots(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    payload = view.row.contract_payload
    blocks = payload["blocks"]
    assert [block["block_index"] for block in blocks] == sorted(
        block["block_index"] for block in blocks
    )
    source_block_payload = blocks[0]
    assert source_block_payload["slot_kind"] == "SOURCE_MEDIA"
    assert source_block_payload["source_binding"]["final_clip_text"]
    # An ORIGINAL_VALUE block becomes a materialization slot, never fabricated text.
    authored = [block for block in blocks if block["slot_kind"] == "AUTHORED_TEXT"]
    assert authored
    assert authored[0]["materialization"]["reason_code"] == "AUTHORED_TEXT_MATERIALIZATION_REQUIRED"
    assert (
        "draft_line" not in authored[0]["materialization"]
        or authored[0]["materialization"]["authoring_reference"]["draft_only"] is True
    )


def test_caption_input_is_exact_final_clip_transcript(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    caption = view.row.contract_payload["caption_input"]
    final = fixture.selection.refinement
    assert caption["transcript_text"] == final.final_transcript
    assert caption["priority"] == "FINAL_CLIP"
    assert caption["refinement_id"] == str(final.id)
    assert caption["logical_order_preserved"] is True
    assert caption["rendered_assets"] is None
    assert caption["rendered"] is False
    assert caption["word_timestamps"] == [
        {
            "index": index,
            "text": word["text"],
            "start": word["start"],
            "end": word["end"],
            "probability": word.get("probability"),
        }
        for index, word in enumerate(final.word_timestamps or [])
    ]


def test_narration_none_adds_no_narration_slot(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings, narration="NONE")
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    kinds = [slot["slot_kind"] for slot in view.row.contract_payload["materialization"]["slots"]]
    assert "AUTHORED_NARRATION" not in kinds


def test_narration_required_adds_unmaterialized_slot(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings, narration="RECOMMENDED")
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    narration_slots = [
        slot
        for slot in view.row.contract_payload["materialization"]["slots"]
        if slot["slot_kind"] == "AUTHORED_NARRATION"
    ]
    assert narration_slots
    assert all(slot["required"] for slot in narration_slots)
    # No TTS voice/provider/model identity is ever selected.
    payload_text = str(view.row.contract_payload).casefold()
    for forbidden in ("voice_id", "voice:", "tts_model", "tts_provider", "speaker_id"):
        assert forbidden not in payload_text


def test_hero_span_is_identified(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    hero = view.row.contract_payload["hero"]
    assert hero is not None
    assert hero["is_identified"] is True
    assert hero["rebound_span"]["start"] is not None


def test_mixed_arabic_english_text_is_preserved_byte_for_byte() -> None:
    words = clip_words(MIXED, start=21.0, end=44.0)
    final = make_final(MIXED, words=words)
    # The exact stored text is never rewritten, reordered, or bidi-injected.
    assert final.final_transcript == MIXED
    assert not any(control in final.final_transcript for control in BIDI_CONTROLS)

    result = evaluate_compatibility(
        blocks=[source_block(MIXED)],
        hero_block_index=0,
        hook_payoff_evidence={"hook_index": None, "payoff_index": None},
        final=final,
        source_segments=(),
        planning_refined_start=20.5,
        planning_refined_end=44.5,
        exact_identity_match=True,
        caption_source_fingerprint=caption_fingerprint_for(final),
        planning_refinement_id="p",
        planning_refinement_priority="CANDIDATE",
        planning_refinement_quality_level="CANDIDATE",
        planning_output_fingerprint="fp",
    )
    assert result.outcome == "EXACT_MATCH"
    assert result.per_block[0].rebound_text == MIXED

    payload = build_caption_source_payload(
        final_transcript=MIXED,
        word_timestamps=[word.as_dict() for word in words],
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
        code_switch_evidence={"suspected": True, "tokens": ["remote", "work", "productivity"]},
        final_refinement_output_fingerprint="fp",
    )
    assert payload["final_transcript"] == MIXED
    assert unicodedata.normalize("NFC", payload["final_transcript"]) == MIXED
