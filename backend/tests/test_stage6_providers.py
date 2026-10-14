"""Stage 6 semantic writing orchestration (no hosted calls)."""

from __future__ import annotations

from test_stage6_materialization_core import TRANSCRIPT, make_inputs

from app.materialization.inputs import MaterializationInputs
from app.materialization.policy import DEFAULT_CONFIG, CallLedger
from app.materialization.providers import (
    DeterministicWritingProvider,
    FakeCritiqueProvider,
    FakeWritingProvider,
    build_writing_request,
    materialize_script,
)

NOVEL_COPY = "The mechanism is that mentorship loss compounds"


def _inputs_with_draft(draft: str | None) -> MaterializationInputs:
    inputs = make_inputs()
    blocks = list(inputs.blocks)
    blocks[1] = {**blocks[1], "draft_line": draft, "grounding_refs": ["block:0"]}
    return make_inputs(blocks=tuple(blocks))


def test_deterministic_provider_defers_novel_copy_without_critique() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    script = materialize_script(
        DeterministicWritingProvider(),
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=False,
    )
    assert script.deferred
    assert script.authored_blocks == ()


def test_deterministic_provider_accepts_draft_after_semantic_critique() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    script = materialize_script(
        DeterministicWritingProvider(),
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        critique_provider=FakeCritiqueProvider(accepted={1: True}),
    )
    assert script.accepted
    assert script.authored_blocks[0].copy == NOVEL_COPY


def test_writing_request_carries_bounded_actual_evidence() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    request = build_writing_request(inputs, config=DEFAULT_CONFIG)
    assert request.targets[0].source_evidence
    assert TRANSCRIPT.split()[0] in request.targets[0].source_evidence


def test_deterministic_provider_makes_no_hosted_calls() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    ledger = CallLedger()
    materialize_script(
        DeterministicWritingProvider(),
        inputs,
        ledger=ledger,
        config=DEFAULT_CONFIG,
        allow_hosted=False,
    )
    assert ledger.semantic == 0
    assert ledger.synthesis == 0


def test_deterministic_provider_without_draft_produces_nothing() -> None:
    inputs = _inputs_with_draft(None)
    script = materialize_script(
        DeterministicWritingProvider(),
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=False,
    )
    assert script.authored_blocks == ()
    assert not script.accepted


def test_paraphrase_copy_is_rejected_by_orchestration() -> None:
    inputs = _inputs_with_draft(TRANSCRIPT)
    script = materialize_script(
        DeterministicWritingProvider(),
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=False,
    )
    assert script.authored_blocks == ()
    assert any("PARAPHRASE_ONLY" in code for code in script.reason_codes)


def test_hosted_disabled_defers_without_calling_provider() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    provider = FakeWritingProvider(copies={1: NOVEL_COPY})
    script = materialize_script(
        provider, inputs, ledger=CallLedger(), config=DEFAULT_CONFIG, allow_hosted=False
    )
    assert script.deferred
    assert provider.calls == []


def test_raw_calls_are_accounted_before_invocation() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    provider = FakeWritingProvider(copies={1: NOVEL_COPY})
    ledger = CallLedger()
    script = materialize_script(
        provider,
        inputs,
        ledger=ledger,
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        critique_provider=FakeCritiqueProvider(accepted={1: True}),
    )
    assert script.accepted
    # One writing call plus one bounded critique call.
    assert ledger.semantic == 2
    assert len(provider.calls) == 1


def test_unavailable_critique_defers_novel_copy() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    provider = FakeWritingProvider(copies={1: NOVEL_COPY})
    script = materialize_script(
        provider,
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        critique_provider=FakeCritiqueProvider(available=False),
    )
    assert script.deferred


def test_semantic_budget_ceiling_defers() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    provider = FakeWritingProvider(copies={1: NOVEL_COPY})
    ledger = CallLedger(semantic=DEFAULT_CONFIG.max_raw_semantic_calls)
    script = materialize_script(
        provider, inputs, ledger=ledger, config=DEFAULT_CONFIG, allow_hosted=True
    )
    assert script.deferred
    assert provider.calls == []


def test_missing_provider_key_defers() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    provider = FakeWritingProvider(available=False)
    script = materialize_script(
        provider, inputs, ledger=CallLedger(), config=DEFAULT_CONFIG, allow_hosted=True
    )
    assert script.deferred


def test_writing_request_batches_substantive_blocks() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    request = build_writing_request(inputs, config=DEFAULT_CONFIG)
    assert [target.block_index for target in request.targets] == [1]
    assert request.tier == "ROUTINE"
