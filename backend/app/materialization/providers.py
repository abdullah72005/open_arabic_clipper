"""Stage 6 semantic writing orchestration and provider interface.

Stage 6 materializes authored copy through one bounded writing request per
generation. The deterministic default makes zero hosted calls and can only reuse
already-authored plan wording (``draft_line``); it never invents copy. Adaptive
mode routes necessary final-candidate work to a hosted provider under a shared
admission gate and a hard raw-call ceiling. Missing key, denial, 429, refusal, or
malformed output defers honestly; unvalidated provider output is never persisted
as authoritative content.

The hosted Gemini adapter is intentionally not yet wired (see
``docs/STAGE_6_OPERATIONS.md``); the deterministic and fake providers are
implemented and the interface is stable so the adapter can be added without
changing callers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.core.enums import ClaimClass
from app.materialization.inputs import MaterializationInputs
from app.materialization.policy import (
    SEMANTIC_ROUTINE_TIER,
    SEMANTIC_STRONG_TIER,
    CallLedger,
    Stage6Config,
)
from app.materialization.scripting import resolve_grounding, validate_authored_block
from app.materialization.types import AuthoredBlock, GroundingRef

_SUBSTANTIVE_BLOCK_TYPES = frozenset({"ORIGINAL_VALUE", "TEXTUAL_ANNOTATION"})


@dataclass(frozen=True)
class WritingTarget:
    """One plan block the provider must materialize copy for."""

    block_index: int
    block_type: str
    substantive_value_kind: str
    semantic_intent: str
    draft_line: str | None
    grounding_refs: tuple[GroundingRef, ...]
    source_evidence: str = ""
    verification_dependency_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "block_type": self.block_type,
            "substantive_value_kind": self.substantive_value_kind,
            "semantic_intent": self.semantic_intent,
            "draft_line": self.draft_line,
            "grounding_refs": [ref.as_dict() for ref in self.grounding_refs],
            "source_evidence": self.source_evidence,
            "verification_dependency_id": self.verification_dependency_id,
        }


@dataclass(frozen=True)
class CritiqueTarget:
    """One authored block the semantic critique must assess for grounding."""

    block_index: int
    copy: str
    substantive_value_kind: str
    semantic_intent: str
    claim_class: str
    source_evidence: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "copy": self.copy,
            "substantive_value_kind": self.substantive_value_kind,
            "semantic_intent": self.semantic_intent,
            "claim_class": self.claim_class,
            "source_evidence": self.source_evidence,
        }


@dataclass(frozen=True)
class CritiqueRequest:
    """One batched semantic-critique request for grounded authored copy."""

    targets: tuple[CritiqueTarget, ...]
    tier: str = SEMANTIC_ROUTINE_TIER

    def as_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "targets": [target.as_dict() for target in self.targets],
        }


@dataclass(frozen=True)
class CritiqueResult:
    """Semantic-critique outcome: whether each grounded block is supported."""

    accepted: Mapping[int, bool]
    provider_identity: Mapping[str, object]
    notes: str = ""
    deferred: bool = False
    reason_code: str | None = None


@dataclass(frozen=True)
class WritingRequest:
    """One batched writing request for a whole materialization generation."""

    targets: tuple[WritingTarget, ...]
    priority: str
    profile_semantic_context: Mapping[str, object]
    tier: str = SEMANTIC_ROUTINE_TIER

    def as_dict(self) -> dict[str, object]:
        return {
            "priority": self.priority,
            "tier": self.tier,
            "profile_semantic_context": dict(self.profile_semantic_context),
            "targets": [target.as_dict() for target in self.targets],
        }


@dataclass(frozen=True)
class WritingResult:
    """Provider writing output (unvalidated copy per block index)."""

    copies: Mapping[int, str]
    provider_identity: Mapping[str, object]
    notes: str = ""
    deferred: bool = False
    reason_code: str | None = None


@runtime_checkable
class MaterializationWritingProvider(Protocol):
    """Provider-neutral authored-copy writing interface."""

    provider_name: str

    def available(self) -> bool: ...

    def runtime_identity(self) -> dict[str, object]: ...

    def write(self, request: WritingRequest) -> WritingResult: ...

    def release(self) -> None: ...


class DeterministicWritingProvider:
    """Zero-hosted-call provider that only reuses existing authored wording."""

    provider_name = "deterministic"

    def available(self) -> bool:
        return True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "deterministic", "model": "none", "api_version": "none"}

    def write(self, request: WritingRequest) -> WritingResult:
        copies: dict[int, str] = {}
        for target in request.targets:
            if target.draft_line:
                copies[target.block_index] = target.draft_line
        return WritingResult(
            copies=copies,
            provider_identity=self.runtime_identity(),
            notes="deterministic draft reuse",
        )

    def release(self) -> None:
        return None


class FakeWritingProvider:
    """Deterministic in-memory provider for tests."""

    provider_name = "fake-writing"

    def __init__(
        self,
        copies: Mapping[int, str] | None = None,
        *,
        available: bool = True,
        deferred: bool = False,
        model: str = "fake-model",
    ) -> None:
        self._copies = dict(copies or {})
        self._available = available
        self._deferred = deferred
        self.model = model
        self.calls: list[WritingRequest] = []
        self.released = 0

    def available(self) -> bool:
        return self._available

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": self.provider_name, "model": self.model, "api_version": "v1"}

    def write(self, request: WritingRequest) -> WritingResult:
        self.calls.append(request)
        if self._deferred:
            return WritingResult(
                copies={},
                provider_identity=self.runtime_identity(),
                deferred=True,
                reason_code="DEFERRED_PROVIDER",
            )
        return WritingResult(
            copies=dict(self._copies),
            provider_identity=self.runtime_identity(),
            notes="fake",
        )

    def release(self) -> None:
        self.released += 1


@runtime_checkable
class MaterializationCritiqueProvider(Protocol):
    """Provider-neutral semantic critique for grounded authored copy."""

    provider_name: str

    def available(self) -> bool: ...

    def runtime_identity(self) -> dict[str, object]: ...

    def critique(self, request: CritiqueRequest) -> CritiqueResult: ...

    def release(self) -> None: ...


class FakeCritiqueProvider:
    """Deterministic in-memory critique provider for tests."""

    provider_name = "fake-critique"

    def __init__(
        self,
        accepted: Mapping[int, bool] | None = None,
        *,
        available: bool = True,
        deferred: bool = False,
        model: str = "fake-critique-model",
    ) -> None:
        self._accepted = dict(accepted or {})
        self._available = available
        self._deferred = deferred
        self.model = model
        self.calls: list[CritiqueRequest] = []
        self.released = 0

    def available(self) -> bool:
        return self._available

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": self.provider_name, "model": self.model, "api_version": "v1"}

    def critique(self, request: CritiqueRequest) -> CritiqueResult:
        self.calls.append(request)
        if self._deferred:
            return CritiqueResult(
                accepted={},
                provider_identity=self.runtime_identity(),
                deferred=True,
                reason_code="DEFERRED_PROVIDER",
            )
        return CritiqueResult(
            accepted=dict(self._accepted),
            provider_identity=self.runtime_identity(),
            notes="fake",
        )

    def release(self) -> None:
        self.released += 1


def build_writing_request(inputs: MaterializationInputs, *, config: Stage6Config) -> WritingRequest:
    """Build one bounded writing request from the plan's substantive blocks.

    Each target carries bounded *actual* resolved source evidence (never only a
    semantic instruction or a reference label) so the provider writes grounded
    copy, and a later critique assesses that copy against real evidence.
    """

    targets: list[WritingTarget] = []
    for block in inputs.blocks:
        block_type = str(block.get("block_type", ""))
        if block_type not in _SUBSTANTIVE_BLOCK_TYPES:
            continue
        index = _int(block.get("index", -1))
        refs = _refs(block.get("grounding_refs"))
        evidence = resolve_grounding(inputs, refs).text[: config.max_evidence_chars_per_target]
        targets.append(
            WritingTarget(
                block_index=index,
                block_type=block_type,
                substantive_value_kind=str(block.get("substantive_value_kind", "")),
                semantic_intent=str(block.get("semantic_intent", "")),
                draft_line=_optional_str(block.get("draft_line")),
                grounding_refs=refs,
                source_evidence=evidence,
                verification_dependency_id=_first_dependency(block, inputs),
            )
        )
    tier = SEMANTIC_STRONG_TIER if inputs.intensity == "STRONG" else SEMANTIC_ROUTINE_TIER
    return WritingRequest(
        targets=tuple(targets),
        priority=inputs.priority,
        profile_semantic_context={
            "target_market": inputs.profile.target_market,
            "output_language_policy": inputs.profile.output_language_policy,
            "register_intent": inputs.profile.register_intent,
        },
        tier=tier,
    )


@dataclass(frozen=True)
class MaterializationScript:
    """Validated authored blocks plus the claim ledger for one generation."""

    authored_blocks: tuple[AuthoredBlock, ...]
    accepted: bool
    provider_identity: Mapping[str, object]
    deferred: bool
    reason_codes: tuple[str, ...]


def materialize_script(
    provider: MaterializationWritingProvider,
    inputs: MaterializationInputs,
    *,
    ledger: CallLedger,
    config: Stage6Config,
    allow_hosted: bool,
    critique_provider: MaterializationCritiqueProvider | None = None,
) -> MaterializationScript:
    """Run exactly one bounded writing request and validate its copy.

    Copy that is lexically novel is not accepted on novelty alone: it is routed
    to a bounded semantic critique that assesses the grounded copy. Without an
    available critique, such copy defers honestly.
    """

    request = build_writing_request(inputs, config=config)
    identity = provider.runtime_identity()
    if not request.targets:
        return MaterializationScript((), True, identity, False, ())

    hosted = provider.provider_name != "deterministic"
    if hosted and not allow_hosted:
        return MaterializationScript((), False, identity, True, ("HOSTED_DISABLED",))
    if hosted:
        if ledger.semantic >= config.max_raw_semantic_calls:
            return MaterializationScript((), False, identity, True, ("SEMANTIC_BUDGET_EXHAUSTED",))
        if (
            request.tier == SEMANTIC_STRONG_TIER
            and ledger.strong >= config.max_strong_tier_semantic_calls
        ):
            return MaterializationScript((), False, identity, True, ("STRONG_BUDGET_EXHAUSTED",))
        if not provider.available():
            return MaterializationScript((), False, identity, True, ("PROVIDER_UNAVAILABLE",))
        # Account the raw call before invocation so a crash can never under-count.
        ledger.semantic += 1
        ledger.ledger_entries.append({"kind": "semantic", "tier": request.tier})
        if request.tier == SEMANTIC_STRONG_TIER:
            ledger.strong += 1

    result = provider.write(request)
    if result.deferred:
        return MaterializationScript(
            (), False, result.provider_identity, True, (result.reason_code or "DEFERRED_PROVIDER",)
        )

    authored: list[AuthoredBlock] = []
    review_targets: list[CritiqueTarget] = []
    reasons: list[str] = []
    accepted_indexes: dict[int, bool] = {}
    for target in request.targets:
        copy = result.copies.get(target.block_index, "")
        block_payload = {
            "block_index": target.block_index,
            "block_type": target.block_type,
            "substantive_value_kind": target.substantive_value_kind,
            "copy": copy,
            "grounding_refs": [ref.as_dict() for ref in target.grounding_refs],
            "verification_dependency_id": target.verification_dependency_id,
        }
        validation = validate_authored_block(block_payload, inputs, config)
        for finding in validation.findings:
            reasons.extend(finding.reason_codes)
        if validation.accepted:
            accepted_indexes[target.block_index] = True
        elif validation.requires_semantic_review and validation.claim is not None:
            review_targets.append(
                CritiqueTarget(
                    block_index=target.block_index,
                    copy=copy,
                    substantive_value_kind=target.substantive_value_kind,
                    semantic_intent=target.semantic_intent,
                    claim_class=validation.claim.claim_class,
                    source_evidence=target.source_evidence,
                )
            )

    if review_targets:
        reviewed = _run_critique(
            critique_provider,
            review_targets,
            ledger=ledger,
            config=config,
            allow_hosted=allow_hosted,
        )
        if reviewed is None:
            return MaterializationScript(
                (),
                False,
                identity,
                True,
                ("GROUNDED_INFERENCE_REQUIRES_REVIEW",),
            )
        accepted_indexes.update(reviewed)

    for target in request.targets:
        if not accepted_indexes.get(target.block_index, False):
            continue
        copy = result.copies.get(target.block_index, "")
        authored.append(
            AuthoredBlock(
                block_index=target.block_index,
                block_type=target.block_type,
                slot_id=f"block-{target.block_index}",
                substantive_value_kind=target.substantive_value_kind,
                copy=copy,
                adds_beyond_source=target.semantic_intent,
                claim_class=ClaimClass.SOURCE_GROUNDED_INFERENCE.value,
                grounding_refs=target.grounding_refs,
                validation_evidence={"accepted": True},
            )
        )
    return MaterializationScript(
        authored_blocks=tuple(authored),
        accepted=len(authored) == len(request.targets),
        provider_identity=result.provider_identity,
        deferred=False,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _run_critique(
    critique_provider: MaterializationCritiqueProvider | None,
    targets: Sequence[CritiqueTarget],
    *,
    ledger: CallLedger,
    config: Stage6Config,
    allow_hosted: bool,
) -> dict[int, bool] | None:
    if critique_provider is None or not allow_hosted:
        return None
    if ledger.semantic >= config.max_raw_semantic_calls:
        return None
    if not critique_provider.available():
        return None
    ledger.semantic += 1
    ledger.ledger_entries.append({"kind": "critique", "tier": SEMANTIC_ROUTINE_TIER})
    result = critique_provider.critique(CritiqueRequest(targets=tuple(targets)))
    if result.deferred:
        return None
    return {int(index): bool(value) for index, value in result.accepted.items()}


def _refs(value: object) -> tuple[GroundingRef, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    refs: list[GroundingRef] = []
    for item in value:
        if isinstance(item, Mapping):
            refs.append(
                GroundingRef(
                    kind=str(item.get("kind", "")),
                    block_index=_optional_int(item.get("block_index")),
                    source_start=_optional_float(item.get("source_start")),
                    source_end=_optional_float(item.get("source_end")),
                    claim_id=_optional_str(item.get("claim_id")),
                )
            )
        elif isinstance(item, str) and item.startswith("block:"):
            refs.append(GroundingRef(kind="SOURCE_EXCERPT", block_index=int(item.split(":", 1)[1])))
    return tuple(refs)


def _first_dependency(block: Mapping[str, object], inputs: MaterializationInputs) -> str | None:
    deps = block.get("dependency_ids")
    if isinstance(deps, (list, tuple)) and deps:
        return str(deps[0])
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return -1
    return value


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = [
    "CritiqueRequest",
    "CritiqueResult",
    "CritiqueTarget",
    "DeterministicWritingProvider",
    "FakeCritiqueProvider",
    "FakeWritingProvider",
    "MaterializationCritiqueProvider",
    "MaterializationScript",
    "MaterializationWritingProvider",
    "WritingRequest",
    "WritingResult",
    "WritingTarget",
    "build_writing_request",
    "materialize_script",
]
