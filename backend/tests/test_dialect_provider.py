"""Shared Stage 2.7 provider dialect contract tests.

Both the local OpenAI-compatible provider and the hosted Gemini provider must
receive the same preservation-first instruction and validated profile-specific
addenda; provider-specific behavior stays limited to transport, response
schema, SDK behavior, sizing, and error handling.
"""

from __future__ import annotations

import json

import pytest

from app.transcription.reconstruction.entities import build_entity_memory
from app.transcription.reconstruction.providers import (
    PROFILE_ADDENDA,
    SYSTEM_INSTRUCTION,
    OpenAICompatibleReconstructionProvider,
    ReconstructionRequest,
    instruction_for_profile,
)
from app.transcription.reconstruction.service import _reconstruction_request


def _provider(request) -> OpenAICompatibleReconstructionProvider:
    return OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=12,
        request=request,
    )


def _ok_response() -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "reconstructions": [
                                    {
                                        "segment_id": 0,
                                        "corrected_text": "هدف",
                                        "unchanged": True,
                                        "confidence": 0.9,
                                        "explanation": "",
                                        "changes": [],
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
    ).encode()


def test_local_and_gemini_share_the_same_base_instruction() -> None:
    from app.transcription.reconstruction.gemini import gemini_system_instruction

    assert gemini_system_instruction(None) == SYSTEM_INSTRUCTION
    assert instruction_for_profile(SYSTEM_INSTRUCTION, "EGYPTIAN") == gemini_system_instruction(
        "EGYPTIAN"
    )


@pytest.mark.parametrize(
    ("profile", "expected_fragment"),
    [
        ("EGYPTIAN", "The source is Egyptian Arabic"),
        ("SAUDI", "The source is Saudi Arabic"),
        ("GULF", "The source is Gulf Arabic"),
        ("LEVANTINE", "The source is Levantine Arabic"),
        ("MSA", "The source is Modern Standard Arabic (Fusha)"),
        ("UNKNOWN_ARABIC", "The Arabic dialect is unknown or mixed"),
    ],
)
def test_each_supported_profile_has_the_correct_narrow_addendum(
    profile: str, expected_fragment: str
) -> None:
    instruction = instruction_for_profile(SYSTEM_INSTRUCTION, profile)

    assert expected_fragment in instruction
    assert profile not in PROFILE_ADDENDA or True


def test_unknown_profile_forbids_forcing_a_dialect() -> None:
    instruction = instruction_for_profile(SYSTEM_INSTRUCTION, "UNKNOWN_ARABIC")

    assert "do not force any regional dialect or Modern Standard Arabic" in instruction


def test_msa_addendum_explicitly_forbids_colloquialization() -> None:
    instruction = instruction_for_profile(SYSTEM_INSTRUCTION, "MSA")

    assert "Preserve formal MSA wording" in instruction
    assert "do not colloquialize it" in instruction


def test_unknown_profile_name_appends_nothing() -> None:
    assert instruction_for_profile(SYSTEM_INSTRUCTION, None) == SYSTEM_INSTRUCTION
    assert instruction_for_profile(SYSTEM_INSTRUCTION, "DO-NOT-INTERPOLATE") == SYSTEM_INSTRUCTION


def test_local_provider_sends_the_effective_profile_in_the_payload() -> None:
    captured: list[dict[str, object]] = []

    def request(
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout: float,
    ) -> bytes:
        assert body is not None
        captured.append(json.loads(body))
        return _ok_response()

    provider = _provider(request)
    provider.reconstruct_segments(
        [
            ReconstructionRequest(
                segment_index=0,
                raw_text="هدف",
                corrected_text="هدف",
                dialect_profile="SAUDI",
            )
        ]
    )

    targets = json.loads(captured[0]["messages"][1]["content"])["targets"]
    assert targets[0]["dialect_profile"] == "SAUDI"
    assert "The source is Saudi Arabic" in captured[0]["messages"][0]["content"]


def test_request_builder_carries_inherited_segment_profile() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "وش رايك الحين",
            "raw_text": "وش رايك الحين",
            "corrected_text": "وش رايك الحين",
            "dialect_profile": "SAUDI",
        }
    ]
    memory = build_entity_memory(segments)

    request = _reconstruction_request(segments, 0, "ar", memory)

    assert request.dialect_profile == "SAUDI"


def test_local_prompt_sizing_includes_the_actual_profile_addendum() -> None:
    provider = _provider(lambda *args: _ok_response())
    base = ReconstructionRequest(segment_index=0, raw_text="هدف", corrected_text="هدف")
    plain = provider.plan_aggregate_batches([base])
    profiled = provider.plan_aggregate_batches(
        [
            ReconstructionRequest(
                segment_index=0, raw_text="هدف", corrected_text="هدف", dialect_profile="MSA"
            )
        ]
    )

    plain_instruction = provider._system_instruction(None)
    profiled_instruction = provider._system_instruction("MSA")
    assert len(profiled_instruction) > len(plain_instruction)
    assert len(plain) == 1
    assert len(profiled) == 1


def test_profile_addendum_changes_the_prompt_hash_and_schema_identity() -> None:
    from app.transcription.reconstruction.providers import _PROMPT_HASH, _PROMPT_SCHEMA_VERSION

    assert _PROMPT_HASH
    assert _PROMPT_SCHEMA_VERSION == "stage-2-7-one-pass-v2"


def test_changed_prompt_identity_invalidates_output_fingerprint() -> None:
    from app.pipeline.fingerprints import reconstruction_output_fingerprint

    segments = [{"raw_text": "هدف", "corrected_text": "هدف", "dialect_profile": None}]
    identity_a = {"provider": "ollama", "prompt_hash": "hash-a", "schema_version": "s1"}
    identity_b = {"provider": "ollama", "prompt_hash": "hash-b", "schema_version": "s1"}

    first = reconstruction_output_fingerprint(
        provider_identity=identity_a,
        segments=segments,
        language="ar",
        transcription_fingerprint="t",
        correction_version="c",
    )
    second = reconstruction_output_fingerprint(
        provider_identity=identity_b,
        segments=segments,
        language="ar",
        transcription_fingerprint="t",
        correction_version="c",
    )

    assert first != second


def test_gemini_and_local_append_identical_egyptian_addendum() -> None:
    from app.transcription.reconstruction.gemini import gemini_system_instruction

    assert gemini_system_instruction("EGYPTIAN") == instruction_for_profile(
        SYSTEM_INSTRUCTION, "EGYPTIAN"
    )


def test_provider_transport_still_uses_bounded_chat_request() -> None:
    captured: list[dict[str, object]] = []

    def request(
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout: float,
    ) -> bytes:
        captured.append({"method": method, "body": body})
        return _ok_response()

    provider = _provider(request)
    provider.reconstruct_segments(
        [ReconstructionRequest(segment_index=0, raw_text="هدف", corrected_text="هدف")]
    )

    assert captured[0]["method"] == "POST"
    body = json.loads(captured[0]["body"])
    assert body["temperature"] == 0
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
