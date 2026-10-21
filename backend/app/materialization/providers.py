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
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from app.core.enums import ClaimClass
from app.materialization.grounding import parse_grounding_refs
from app.materialization.inputs import MaterializationInputs
from app.materialization.policy import (
    SEMANTIC_ROUTINE_TIER,
    SEMANTIC_STRONG_TIER,
    CallLedger,
    Stage6Config,
)
from app.materialization.scripting import resolve_grounding, validate_authored_block
from app.materialization.timeline import MIN_AUTHORED_CARD_SECONDS, READING_CHARS_PER_SECOND
from app.materialization.types import AuthoredBlock, GroundingRef, QualityFinding

_SUBSTANTIVE_BLOCK_TYPES = frozenset({"ORIGINAL_VALUE", "TEXTUAL_ANNOTATION"})


class MaterializationProviderError(Exception):
    """Typed Stage 6 provider failure (category + truthful reason code)."""

    def __init__(self, category: str, reason_code: str | None = None) -> None:
        self.category = category
        self.reason_code = reason_code or category
        super().__init__(self.reason_code)


#: Closed categorical critique dimensions (spec section 7.1). Each is resolved as
#: PASS/CONCERN/FAIL/UNKNOWN with short evidence. There is no scalar score.
CRITIQUE_DIMENSIONS = (
    "GROUNDING_ENTAILMENT",
    "SUBSTANTIVE_ADDED_VALUE",
    "INTENT_FULFILLMENT",
    "SEMANTIC_FIDELITY",
    "UNSUPPORTED_FACTS",
    "CODE_SWITCH_PRESERVATION",
    "REDUNDANCY",
    "DIALECT_REGISTER_FIT",
    "NATURALNESS",
    "LENGTH_NECESSITY",
)

CRITIQUE_LEVELS = ("PASS", "CONCERN", "FAIL", "UNKNOWN")

#: Dimensions whose CONCERN/FAIL is a critical failure: no approval is possible.
#: GROUNDING_ENTAILMENT, SUBSTANTIVE_ADDED_VALUE and INTENT_FULFILLMENT
#: additionally must be a *known positive* PASS (an UNKNOWN/absent verdict
#: defers, never approves).
CRITICAL_CRITIQUE_DIMENSIONS = frozenset(
    {
        "GROUNDING_ENTAILMENT",
        "SUBSTANTIVE_ADDED_VALUE",
        "INTENT_FULFILLMENT",
        "SEMANTIC_FIDELITY",
        "UNSUPPORTED_FACTS",
        "CODE_SWITCH_PRESERVATION",
    }
)
_POSITIVE_REQUIRED_DIMENSIONS = (
    "GROUNDING_ENTAILMENT",
    "SUBSTANTIVE_ADDED_VALUE",
    "INTENT_FULFILLMENT",
)

#: Dimensions that make a substantive block worthless when merely CONCERN.
#: REDUNDANCY cannot be waved through by an otherwise favorable critique.
_SUBSTANTIVE_CONCERN_CONSEQUENTIAL = frozenset({"REDUNDANCY"})

#: A block verdict is a closed tri-state. ``ACCEPTED`` requires exactly one valid
#: finding for every requested dimension and a known PASS for every critical
#: dimension. ``DEFERRED`` means the critique evidence is missing, malformed,
#: duplicated, contradictory, or critical-UNKNOWN and therefore can never imply
#: approval. ``REJECTED`` means a critical dimension is CONCERN/FAIL.
CRITIQUE_ACCEPTED = "ACCEPTED"
CRITIQUE_DEFERRED = "DEFERRED"
CRITIQUE_REJECTED = "REJECTED"

#: Map a categorical critique dimension to the persisted Stage 6 quality
#: dimension so noncritical concerns survive into assessment/final findings.
CRITIQUE_TO_QUALITY_DIMENSION: dict[str, str] = {
    "GROUNDING_ENTAILMENT": "GROUNDING_VERIFICATION",
    "SUBSTANTIVE_ADDED_VALUE": "SUBSTANTIVE_VALUE",
    "INTENT_FULFILLMENT": "SUBSTANTIVE_VALUE",
    "SEMANTIC_FIDELITY": "FIDELITY_ATTRIBUTION",
    "UNSUPPORTED_FACTS": "GROUNDING_VERIFICATION",
    "CODE_SWITCH_PRESERVATION": "LANGUAGE_REGISTER_NATURALNESS",
    "REDUNDANCY": "SUBSTANTIVE_VALUE",
    "DIALECT_REGISTER_FIT": "LANGUAGE_REGISTER_NATURALNESS",
    "NATURALNESS": "LANGUAGE_REGISTER_NATURALNESS",
    "LENGTH_NECESSITY": "PACING_CONTEXT_DENSITY",
}

#: Categorical critique level -> persisted quality level.
CRITIQUE_TO_QUALITY_LEVEL: dict[str, str] = {
    "PASS": "SATISFIED",
    "CONCERN": "CONCERN",
    "FAIL": "UNMET",
    "UNKNOWN": "UNKNOWN",
}


@dataclass(frozen=True)
class CritiqueDimensionFinding:
    """One categorical critique finding for a single dimension.

    ``copy_quote``/``grounding_refs``/``added_relation`` are review evidence for
    ``INTENT_FULFILLMENT`` only: an exact quote from the actual copy, references
    drawn from the target's offered evidence, and the relationship that quote
    expresses beyond restatement. They are never new authored copy or timestamps.
    """

    dimension: str
    level: str
    evidence: str = ""
    copy_quote: str = ""
    grounding_refs: tuple[str, ...] = ()
    added_relation: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "level": self.level,
            "evidence": self.evidence,
            "copy_quote": self.copy_quote,
            "grounding_refs": list(self.grounding_refs),
            "added_relation": self.added_relation,
        }


@dataclass(frozen=True)
class BlockCritique:
    """Categorical critique for one authored block, with a derived decision."""

    block_index: int
    findings: tuple[CritiqueDimensionFinding, ...] = ()
    substantive: bool = False
    require_length_necessity: bool = False

    def _level(self, dimension: str) -> str | None:
        levels = self.levels()
        return levels.get(dimension)

    def levels(self, requested: Sequence[str] = CRITIQUE_DIMENSIONS) -> dict[str, str]:
        """Return the known level per dimension, dropping duplicates/non-requested.

        A duplicate or out-of-vocabulary finding for a requested dimension is
        recorded via :meth:`structural_issues` and never silently resolved to a
        favorable value.
        """

        allowed = set(requested)
        levels: dict[str, str] = {}
        for finding in self.findings:
            if finding.dimension not in allowed:
                continue
            if finding.dimension in levels:
                continue
            if finding.level not in CRITIQUE_LEVELS:
                continue
            levels[finding.dimension] = finding.level
        return levels

    def structural_issues(self, requested: Sequence[str] = CRITIQUE_DIMENSIONS) -> list[str]:
        """Malformed/duplicate/missing findings that cannot imply approval."""

        allowed = set(requested)
        issues: list[str] = []
        seen: set[str] = set()
        for finding in self.findings:
            if finding.dimension not in allowed:
                issues.append(f"UNEXPECTED_DIMENSION:{finding.dimension}")
                continue
            if finding.level not in CRITIQUE_LEVELS:
                issues.append(f"INVALID_LEVEL:{finding.dimension}:{finding.level}")
                continue
            if finding.dimension in seen:
                issues.append(f"DUPLICATE_DIMENSION:{finding.dimension}")
                continue
            seen.add(finding.dimension)
        for dimension in requested:
            if dimension not in seen:
                issues.append(f"MISSING_DIMENSION:{dimension}")
        return issues

    def verdict(self, requested: Sequence[str] = CRITIQUE_DIMENSIONS) -> str:
        """Tri-state block verdict derived from complete categorical evidence.

        Exactly one valid finding is required for every requested dimension. A
        missing/malformed/duplicate finding defers; any critical CONCERN/FAIL
        rejects; a critical UNKNOWN defers; only an all-critical-PASS block is
        accepted. Noncritical CONCERN/FAIL/UNKNOWN never block acceptance.
        """

        if self.structural_issues(requested):
            return CRITIQUE_DEFERRED
        levels = self.levels(requested)
        for dimension in CRITICAL_CRITIQUE_DIMENSIONS:
            level = levels.get(dimension)
            if level in {"CONCERN", "FAIL"}:
                return CRITIQUE_REJECTED
            if level != "PASS":
                return CRITIQUE_DEFERRED
        if self.substantive:
            # Redundant substantive copy adds no value even if grounding and
            # novelty are otherwise favorable; a missing judgment defers.
            redundancy = levels.get("REDUNDANCY")
            if redundancy in {"CONCERN", "FAIL"}:
                return CRITIQUE_REJECTED
            if redundancy != "PASS":
                return CRITIQUE_DEFERRED
        if self.require_length_necessity:
            # Standalone authored delivery must have its actual interruption
            # judged, not assumed. Missing evidence defers.
            length = levels.get("LENGTH_NECESSITY")
            if length in {"CONCERN", "FAIL"}:
                return CRITIQUE_REJECTED
            if length != "PASS":
                return CRITIQUE_DEFERRED
        return CRITIQUE_ACCEPTED

    @property
    def accepted(self) -> bool:
        return self.verdict() == CRITIQUE_ACCEPTED

    def as_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "accepted": self.accepted,
            "verdict": self.verdict(),
            "substantive": self.substantive,
            "require_length_necessity": self.require_length_necessity,
            "findings": [finding.as_dict() for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BlockCritique":
        raw = data.get("findings")
        findings: list[CritiqueDimensionFinding] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, Mapping):
                    refs = item.get("grounding_refs")
                    findings.append(
                        CritiqueDimensionFinding(
                            dimension=str(item.get("dimension", "")),
                            level=str(item.get("level", "")),
                            evidence=str(item.get("evidence", "")),
                            copy_quote=str(item.get("copy_quote", "") or ""),
                            grounding_refs=(
                                tuple(str(ref) for ref in refs) if isinstance(refs, list) else ()
                            ),
                            added_relation=str(item.get("added_relation", "") or ""),
                        )
                    )
        index = data.get("block_index")
        block_index = index if isinstance(index, int) and not isinstance(index, bool) else -1
        return cls(
            block_index=block_index,
            findings=tuple(findings),
            substantive=bool(data.get("substantive", False)),
            require_length_necessity=bool(data.get("require_length_necessity", False)),
        )


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
    why_unavailable: str = ""
    delivery_intent: str = ""
    proposed_delivery: Mapping[str, object] = field(default_factory=dict)

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
            "why_unavailable": self.why_unavailable,
            "delivery_intent": self.delivery_intent,
            "proposed_delivery": dict(self.proposed_delivery),
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
    offered_grounding_refs: tuple[str, ...] = ()
    delivery_intent: str = ""
    proposed_delivery: Mapping[str, object] = field(default_factory=dict)
    substantive: bool = True
    require_length_necessity: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "copy": self.copy,
            "substantive_value_kind": self.substantive_value_kind,
            "semantic_intent": self.semantic_intent,
            "claim_class": self.claim_class,
            "source_evidence": self.source_evidence,
            "offered_grounding_refs": list(self.offered_grounding_refs),
            "delivery_intent": self.delivery_intent,
            "proposed_delivery": dict(self.proposed_delivery),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CritiqueTarget | None":
        """Parse the exact bounded target snapshot persisted with a critique.

        Returns ``None`` for malformed/missing fields. ``substantive`` and
        ``require_length_necessity`` are not part of the provider request
        snapshot; callers that need them derive them from accepted evidence.
        """

        index = data.get("block_index")
        if isinstance(index, bool) or not isinstance(index, int):
            return None
        values: dict[str, str] = {}
        for key in (
            "copy",
            "substantive_value_kind",
            "semantic_intent",
            "claim_class",
            "source_evidence",
            "delivery_intent",
        ):
            value = data.get(key)
            if not isinstance(value, str):
                return None
            values[key] = value
        raw_refs = data.get("offered_grounding_refs")
        if not isinstance(raw_refs, list) or any(not isinstance(ref, str) for ref in raw_refs):
            return None
        proposed = data.get("proposed_delivery")
        if not isinstance(proposed, Mapping):
            return None
        return cls(
            block_index=index,
            offered_grounding_refs=tuple(raw_refs),
            proposed_delivery=dict(proposed),
            **values,
        )


@dataclass(frozen=True)
class CritiqueRequest:
    """One batched semantic-critique request for grounded authored copy."""

    targets: tuple[CritiqueTarget, ...]
    tier: str = SEMANTIC_ROUTINE_TIER
    source_context: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "source_context": dict(self.source_context),
            "targets": [target.as_dict() for target in self.targets],
        }


@dataclass(frozen=True)
class CritiqueResult:
    """Semantic-critique outcome: whether each grounded block is supported.

    ``blocks`` carries the full categorical findings; ``accepted`` is a legacy
    bool-only map retained for parsing older checkpoints/test adapters. A
    boolean-positive entry is never approval evidence: only complete categorical
    findings can authorize acceptance. A boolean-negative entry remains a safe
    rejection.
    """

    accepted: Mapping[int, bool] = field(default_factory=dict)
    provider_identity: Mapping[str, object] = field(default_factory=dict)
    notes: str = ""
    deferred: bool = False
    reason_code: str | None = None
    blocks: Mapping[int, "BlockCritique"] = field(default_factory=dict)
    #: Exact bounded target facts (``CritiqueTarget.as_dict()``) a replayed
    #: checkpointed critique actually reviewed. Persisted findings use these
    #: facts, never the current target, so an old judgment can never be
    #: relabelled with delivery/timing evidence the provider did not see.
    reviewed_targets: Mapping[int, Mapping[str, object]] = field(default_factory=dict)

    def decision(self, index: int) -> bool | None:
        block = self.blocks.get(index)
        if block is not None:
            return block.accepted
        if index in self.accepted and not self.accepted[index]:
            # A known-negative boolean remains a safe rejection. A positive
            # boolean is not complete categorical evidence, so it can never
            # authorize approval; it stays unknown.
            return False
        return None

    def verdict(self, index: int, requested: Sequence[str] = CRITIQUE_DIMENSIONS) -> str | None:
        """Tri-state verdict for one block, preferring full categorical evidence.

        Complete categorical ``blocks`` are required. A legacy boolean-positive
        result is not evidence and defers; a negative boolean still rejects.
        """

        block = self.blocks.get(index)
        if block is not None:
            return block.verdict(requested)
        if index in self.accepted:
            # An explicit negative legacy boolean is a safe rejection, matching
            # :meth:`decision`. A positive boolean is not complete categorical
            # evidence and can never authorize approval; it defers.
            return CRITIQUE_DEFERRED if self.accepted[index] else CRITIQUE_REJECTED
        return None

    def findings_for(self, index: int) -> tuple[CritiqueDimensionFinding, ...]:
        block = self.blocks.get(index)
        return block.findings if block is not None else ()


@dataclass(frozen=True)
class WritingRequest:
    """One batched writing request for a whole materialization generation."""

    targets: tuple[WritingTarget, ...]
    priority: str
    profile_semantic_context: Mapping[str, object]
    tier: str = SEMANTIC_ROUTINE_TIER
    source_context: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "priority": self.priority,
            "tier": self.tier,
            "profile_semantic_context": dict(self.profile_semantic_context),
            "source_context": dict(self.source_context),
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
        targets = {target.block_index: target for target in request.targets}
        blocks: dict[int, BlockCritique] = {}
        for index, accepted in self._accepted.items():
            target = targets.get(index)
            if target is None:
                continue
            level = "PASS" if accepted else "FAIL"
            findings = tuple(
                CritiqueDimensionFinding(
                    dimension=dimension,
                    level=level,
                    copy_quote=target.copy if dimension == "INTENT_FULFILLMENT" else "",
                    grounding_refs=(
                        target.offered_grounding_refs[:1]
                        if dimension == "INTENT_FULFILLMENT"
                        else ()
                    ),
                    added_relation=(
                        "Fake critique accepted the supplied relation."
                        if dimension == "INTENT_FULFILLMENT"
                        else ""
                    ),
                )
                for dimension in CRITIQUE_DIMENSIONS
            )
            blocks[index] = BlockCritique(block_index=index, findings=findings)
        return CritiqueResult(
            accepted=dict(self._accepted),
            provider_identity=self.runtime_identity(),
            notes="fake",
            blocks=blocks,
        )

    def release(self) -> None:
        self.released += 1


def build_writing_request(
    inputs: MaterializationInputs,
    *,
    config: Stage6Config,
    narration_available: bool = True,
) -> WritingRequest:
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
                why_unavailable=str(block.get("why_unavailable", "") or ""),
                delivery_intent=str(block.get("delivery_intent", "") or ""),
                proposed_delivery=_proposed_delivery(
                    block, inputs, narration_available=narration_available
                ),
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
        source_context=_source_context(inputs),
    )


def _proposed_delivery(
    block: Mapping[str, object],
    inputs: MaterializationInputs,
    *,
    narration_available: bool,
) -> dict[str, object]:
    """Executable delivery facts for the critique's LENGTH_NECESSITY.

    The timeline delivers narration only when the plan requires/recommends it,
    narration is allowed by the profile, and a synthesizer is available; every
    other authored block becomes a silent standalone card. A plan's
    ``delivery_intent`` (including ``FLEXIBLE``) is not the executable delivery
    and cannot make a silent card look like spoken text.
    """

    narration_need = str(inputs.narration.get("need", "NONE"))
    narration_will_run = (
        narration_need in {"RECOMMENDED", "REQUIRED"}
        and narration_available
        and bool(inputs.profile.narration_allowed)
    )
    kind = "NARRATION" if narration_will_run else "STANDALONE_CARD"
    estimated = block.get("estimated_duration")
    duration = (
        float(estimated)
        if isinstance(estimated, (int, float)) and not isinstance(estimated, bool)
        else 0.0
    )
    return {
        "delivery": kind,
        "planned_duration_seconds": round(duration, 6),
        "silent": kind == "STANDALONE_CARD",
        "purpose": str(block.get("purpose", "") or ""),
    }


def _offered_refs(refs: Sequence[GroundingRef]) -> tuple[str, ...]:
    return tuple(ref.raw for ref in refs if isinstance(ref.raw, str) and ref.raw.strip())


def _source_context(inputs: MaterializationInputs) -> dict[str, object]:
    """Source language/dialect identity, supplied separately from target context.

    The source language/dialect is descriptive evidence about the material, never
    a deployment-wide default or a target audience; it is passed to providers
    separately so the shared instruction stays dialect-neutral.
    """

    return {
        "source_language": inputs.source_language,
        "source_dialect": dict(inputs.source_dialect),
    }


def critique_source_context(inputs: MaterializationInputs) -> dict[str, object]:
    """Public source-context accessor for critique request binding validation."""

    return _source_context(inputs)


@dataclass(frozen=True)
class MaterializationScript:
    """Validated authored blocks plus the claim ledger for one generation."""

    authored_blocks: tuple[AuthoredBlock, ...]
    accepted: bool
    provider_identity: Mapping[str, object]
    deferred: bool
    reason_codes: tuple[str, ...]
    findings: tuple[QualityFinding, ...] = ()


def materialize_script(
    provider: MaterializationWritingProvider,
    inputs: MaterializationInputs,
    *,
    ledger: CallLedger,
    config: Stage6Config,
    allow_hosted: bool,
    attempt_scope: str = "",
    critique_provider: MaterializationCritiqueProvider | None = None,
    narration_available: bool = True,
) -> MaterializationScript:
    """Run exactly one bounded writing request and validate its copy.

    Copy that is lexically novel is not accepted on novelty alone: it is routed
    to a bounded semantic critique that assesses the grounded copy. Without an
    available critique, such copy defers honestly. Each hosted raw request is
    reserved under a stable ``attempt_scope``-derived identity before invocation,
    so a replayed checkpoint or a crashed attempt never spends the budget twice.
    """

    request = build_writing_request(inputs, config=config, narration_available=narration_available)
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
        # Reserve the raw attempt before invocation so a crash can never
        # under-count; a replayed/crashed attempt stays budget-consumed and is
        # never invoked a second time.
        if not ledger.reserve(
            f"{attempt_scope}:write:{request.tier}", kind="semantic", tier=request.tier
        ):
            return MaterializationScript((), False, identity, True, ("ATTEMPT_ALREADY_RESERVED",))

    result = provider.write(request)
    if result.deferred:
        return MaterializationScript(
            (), False, result.provider_identity, True, (result.reason_code or "DEFERRED_PROVIDER",)
        )

    authored: list[AuthoredBlock] = []
    review_targets: list[CritiqueTarget] = []
    reasons: list[str] = []
    findings_by_block: dict[int, list[QualityFinding]] = {}
    accepted_indexes: dict[int, bool] = {}
    critique_by_block: dict[int, BlockCritique] = {}
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
        findings_by_block[target.block_index] = list(validation.findings)
        for finding in validation.findings:
            reasons.extend(finding.reason_codes)
        if validation.accepted:
            accepted_indexes[target.block_index] = True
        elif validation.requires_semantic_review and validation.claim is not None:
            substantive = (
                target.block_type in _SUBSTANTIVE_BLOCK_TYPES
                and target.substantive_value_kind != "SOURCE_AS_EVIDENCE"
            )
            proposed_delivery = dict(target.proposed_delivery)
            reading_seconds = len(copy) / READING_CHARS_PER_SECOND
            if str(proposed_delivery.get("delivery", "")) == "STANDALONE_CARD":
                # The timeline's executable duration for a silent card is the
                # readable minimum derived from the actual copy, not the plan's
                # estimate. The critique must judge the interruption that will
                # actually be delivered.
                proposed_delivery["planned_duration_seconds"] = round(
                    max(MIN_AUTHORED_CARD_SECONDS, reading_seconds), 6
                )
                proposed_delivery["silent"] = True
            proposed_delivery["copy_characters"] = len(copy)
            proposed_delivery["reading_seconds"] = round(reading_seconds, 6)
            require_length_necessity = substantive and (
                str(proposed_delivery.get("delivery", "")) == "STANDALONE_CARD"
                or str(inputs.narration.get("need", "NONE")) == "RECOMMENDED"
            )
            review_targets.append(
                CritiqueTarget(
                    block_index=target.block_index,
                    copy=copy,
                    substantive_value_kind=target.substantive_value_kind,
                    semantic_intent=target.semantic_intent,
                    claim_class=validation.claim.claim_class,
                    source_evidence=target.source_evidence,
                    offered_grounding_refs=_offered_refs(target.grounding_refs),
                    delivery_intent=target.delivery_intent,
                    proposed_delivery=proposed_delivery,
                    substantive=substantive,
                    require_length_necessity=require_length_necessity,
                )
            )

    unresolved = False
    reviewed_by_block = {
        target.block_index: _target_review_facts(target) for target in review_targets
    }
    if review_targets:
        evaluation = _run_critique(
            critique_provider,
            review_targets,
            ledger=ledger,
            config=config,
            allow_hosted=allow_hosted,
            attempt_scope=attempt_scope,
            source_context=_source_context(inputs),
        )
        if evaluation is None:
            # No usable critique evidence: the review-required block defers
            # honestly rather than being accepted on novelty alone. Preserve a
            # checkpoint-level deferral reason (e.g. a request mismatch) so the
            # mismatch is observable without a provider resend.
            deferred_reason = getattr(critique_provider, "deferred_reason", None)
            reasons_out = ("GROUNDED_INFERENCE_REQUIRES_REVIEW",)
            if isinstance(deferred_reason, str) and deferred_reason:
                reasons_out = (deferred_reason, *reasons_out)
            return MaterializationScript(
                (),
                False,
                identity,
                True,
                reasons_out,
                _combined_findings(findings_by_block, accepted_indexes, {}, reviewed_by_block),
            )
        # A replayed checkpoint contributes the facts it actually reviewed;
        # those win over the current targets so findings stay bound to the
        # judgment that produced them.
        reviewed_by_block.update(evaluation.reviewed_facts)
        for target in review_targets:
            index = target.block_index
            block = evaluation.blocks.get(index)
            if block is not None:
                critique_by_block[index] = block
            verdict = evaluation.verdict(index)
            intent_issue = _intent_fulfillment_issue(block, target)
            if intent_issue is not None:
                findings_by_block.setdefault(index, []).append(
                    QualityFinding(
                        dimension="SUBSTANTIVE_VALUE",
                        level="UNKNOWN",
                        reason_codes=(intent_issue,),
                        evidence={"block_index": index},
                    )
                )
            if verdict == CRITIQUE_ACCEPTED and intent_issue is None:
                accepted_indexes[index] = True
            elif verdict == CRITIQUE_REJECTED:
                accepted_indexes[index] = False
                reasons.extend(_critique_reasons(block))
            else:
                # Missing/malformed/duplicate/contradictory/critical-UNKNOWN
                # evidence can never imply approval: the generation defers.
                unresolved = True

    if unresolved:
        return MaterializationScript(
            (),
            False,
            identity,
            True,
            tuple(dict.fromkeys((*reasons, "GROUNDED_INFERENCE_REQUIRES_REVIEW"))),
            _combined_findings(
                findings_by_block, accepted_indexes, critique_by_block, reviewed_by_block
            ),
        )

    for target in request.targets:
        if not accepted_indexes.get(target.block_index, False):
            continue
        copy = result.copies.get(target.block_index, "")
        evidence: dict[str, object] = {"accepted": True}
        block_critique = critique_by_block.get(target.block_index)
        added_relation = ""
        if block_critique is not None:
            evidence["critique"] = block_critique.as_dict()
            intent = next(
                (
                    item
                    for item in block_critique.findings
                    if item.dimension == "INTENT_FULFILLMENT"
                ),
                None,
            )
            if intent is not None and intent.level == "PASS":
                added_relation = intent.added_relation.strip()
        authored.append(
            AuthoredBlock(
                block_index=target.block_index,
                block_type=target.block_type,
                slot_id=f"block-{target.block_index}",
                substantive_value_kind=target.substantive_value_kind,
                copy=copy,
                # Never echo the plan's promise as proof of added value. The
                # concrete relation comes from reviewed intent-fulfillment
                # evidence; without it the field stays empty.
                adds_beyond_source=added_relation,
                claim_class=ClaimClass.SOURCE_GROUNDED_INFERENCE.value,
                grounding_refs=target.grounding_refs,
                validation_evidence=evidence,
            )
        )
    return MaterializationScript(
        authored_blocks=tuple(authored),
        accepted=len(authored) == len(request.targets),
        provider_identity=result.provider_identity,
        deferred=False,
        reason_codes=tuple(dict.fromkeys(reasons)),
        findings=_combined_findings(
            findings_by_block, accepted_indexes, critique_by_block, reviewed_by_block
        ),
    )


def _target_review_facts(target: CritiqueTarget) -> dict[str, object]:
    """The bounded delivery/timing facts a critique of this target reviewed."""

    proposed = target.proposed_delivery
    return {
        "block_index": target.block_index,
        "copy": target.copy,
        "reviewed_delivery": proposed.get("delivery"),
        "reviewed_duration_seconds": proposed.get("planned_duration_seconds"),
        "reviewed_copy_characters": proposed.get("copy_characters"),
    }


def _review_facts_from_target(index: int, data: Mapping[str, object]) -> dict[str, object]:
    """Reconstruct reviewed facts from a persisted ``CritiqueTarget.as_dict()``."""

    proposed = data.get("proposed_delivery")
    delivery = proposed if isinstance(proposed, Mapping) else {}
    return {
        "block_index": index,
        "copy": data.get("copy"),
        "reviewed_delivery": delivery.get("delivery"),
        "reviewed_duration_seconds": delivery.get("planned_duration_seconds"),
        "reviewed_copy_characters": delivery.get("copy_characters"),
    }


def _combined_findings(
    findings_by_block: Mapping[int, list[QualityFinding]],
    accepted_indexes: Mapping[int, bool],
    critique_by_block: Mapping[int, BlockCritique],
    reviewed_facts: Mapping[int, Mapping[str, object]] | None = None,
) -> tuple[QualityFinding, ...]:
    """All persisted script findings, including noncritical categorical concerns.

    Deterministic findings for blocks that were not ultimately accepted are kept
    (a favorable critique that resolved a soft inference concern drops only that
    block's own deterministic concern). Categorical critique findings are carried
    as quality findings for *every* reviewed block, so a noncritical concern
    (e.g. NATURALNESS) survives into persisted script evidence and the final
    assessment even when the block was accepted.

    Positive ``LENGTH_NECESSITY`` evidence additionally records the exact block,
    copy and delivery/duration it reviewed, so timeline assessment can never
    credit a sibling block's PASS or a stale copy. ``reviewed_facts`` carries the
    facts snapshot from the checkpoint that actually produced the judgment.
    """

    targets = reviewed_facts or {}
    combined: list[QualityFinding] = []
    for index, findings in findings_by_block.items():
        if accepted_indexes.get(index, False):
            continue
        combined.extend(findings)
    for block in critique_by_block.values():
        facts: dict[str, object] = {"block_index": block.block_index}
        reviewed = targets.get(block.block_index)
        if isinstance(reviewed, Mapping):
            facts.update(dict(reviewed))
            facts["block_index"] = block.block_index
        for finding in block.findings:
            dimension = CRITIQUE_TO_QUALITY_DIMENSION.get(finding.dimension)
            level = CRITIQUE_TO_QUALITY_LEVEL.get(finding.level)
            if dimension is None or level is None:
                continue
            if level == "SATISFIED":
                # Keep the positive evidence that actually authorizes a
                # standalone delivery or an intent-fulfilled substantive block;
                # drop other routine PASS noise.
                if finding.dimension not in {"LENGTH_NECESSITY", "INTENT_FULFILLMENT"}:
                    continue
                combined.append(
                    QualityFinding(
                        dimension=dimension,
                        level=level,
                        reason_codes=(f"CRITIQUE_{finding.dimension}_PASS",),
                        evidence={
                            **facts,
                            "critique_dimension": finding.dimension,
                            "critique_level": finding.level,
                            "evidence": finding.evidence,
                        },
                    )
                )
                continue
            combined.append(
                QualityFinding(
                    dimension=dimension,
                    level=level,
                    reason_codes=(f"CRITIQUE_{finding.dimension}_{finding.level}",),
                    evidence={
                        **facts,
                        "critique_dimension": finding.dimension,
                        "critique_level": finding.level,
                        "evidence": finding.evidence,
                    },
                )
            )
    return tuple(combined)


def _intent_fulfillment_issue(block: BlockCritique | None, target: CritiqueTarget) -> str | None:
    """Deterministically validate the INTENT_FULFILLMENT review evidence.

    A PASS must carry an exact copy quote, nonempty offered-and-resolvable
    grounding references, and a bounded added relation. Structure is validated
    here; whether the relation actually fulfils the promised intent remains the
    reviewer's categorical judgment.
    """

    if block is None:
        return "MISSING_INTENT_FULFILLMENT_EVIDENCE"
    finding = next(
        (item for item in block.findings if item.dimension == "INTENT_FULFILLMENT"),
        None,
    )
    if finding is None:
        return "MISSING_INTENT_FULFILLMENT_EVIDENCE"
    if finding.level in {"CONCERN", "FAIL", "UNKNOWN"}:
        # The verdict already rejects/defers these levels.
        return None
    quote = finding.copy_quote.strip()
    if not quote or quote not in target.copy:
        return "INVALID_INTENT_FULFILLMENT_COPY_QUOTE"
    refs = tuple(ref.strip() for ref in finding.grounding_refs if ref.strip())
    if not refs:
        return "MISSING_INTENT_FULFILLMENT_GROUNDING"
    offered = {ref.strip() for ref in target.offered_grounding_refs if ref.strip()}
    parsed = parse_grounding_refs(list(refs))
    if any(ref.invalid_reason for ref in parsed) or len(parsed) != len(refs):
        return "INVALID_INTENT_FULFILLMENT_GROUNDING"
    if any(ref not in offered for ref in refs):
        return "OUT_OF_SCOPE_INTENT_FULFILLMENT_GROUNDING"
    relation = finding.added_relation.strip()
    if not relation or len(relation) > 400:
        return "MISSING_INTENT_FULFILLMENT_RELATION"
    return None


def _critique_reasons(block: BlockCritique | None) -> list[str]:
    if block is None:
        return []
    dimensions = set(CRITICAL_CRITIQUE_DIMENSIONS)
    if block.substantive:
        dimensions.add("REDUNDANCY")
    if block.require_length_necessity:
        dimensions.add("LENGTH_NECESSITY")
    reasons: list[str] = []
    for dimension in dimensions:
        level = block._level(dimension)
        if level in {"CONCERN", "FAIL"}:
            reasons.append(f"CRITIQUE_{dimension}_{level}")
    return reasons


@dataclass(frozen=True)
class CritiqueEvaluation:
    """Validated categorical critique for the requested review targets."""

    blocks: Mapping[int, BlockCritique]
    #: Per-block facts the accepted judgment actually reviewed. A replayed
    #: checkpoint supplies its recorded snapshot; a fresh response uses the
    #: current bounded request.
    reviewed_facts: Mapping[int, Mapping[str, object]] = field(default_factory=dict)

    def verdict(self, index: int) -> str | None:
        block = self.blocks.get(index)
        return block.verdict() if block is not None else None


def _run_critique(
    critique_provider: MaterializationCritiqueProvider | None,
    targets: Sequence[CritiqueTarget],
    *,
    ledger: CallLedger,
    config: Stage6Config,
    allow_hosted: bool,
    attempt_scope: str,
    source_context: Mapping[str, object] | None = None,
) -> CritiqueEvaluation | None:
    if critique_provider is None or not allow_hosted:
        return None
    if ledger.semantic >= config.max_raw_semantic_calls:
        return None
    if not critique_provider.available():
        return None
    if not ledger.reserve(f"{attempt_scope}:critique", kind="critique"):
        return None
    result = critique_provider.critique(
        CritiqueRequest(targets=tuple(targets), source_context=dict(source_context or {}))
    )
    if result.deferred:
        return None
    reviewed_facts = {target.block_index: _target_review_facts(target) for target in targets}
    # A bound replay reports the exact target snapshot it reviewed; prefer it so
    # persisted findings can never describe facts the provider did not see.
    for index, data in result.reviewed_targets.items():
        if isinstance(data, Mapping):
            reviewed_facts[index] = _review_facts_from_target(index, data)
    blocks: dict[int, BlockCritique] = {}
    for target in targets:
        block = result.blocks.get(target.block_index)
        if block is not None:
            blocks[target.block_index] = replace(
                block,
                substantive=target.substantive,
                require_length_necessity=target.require_length_necessity,
            )
        # A boolean-only (legacy/malformed) result has no complete categorical
        # evidence. It is never promoted into findings, quotes, references or an
        # added relation. Leaving the block absent makes the caller defer
        # without resending the provider request.
    if not blocks:
        return None
    return CritiqueEvaluation(blocks=blocks, reviewed_facts=reviewed_facts)


def _refs(value: object) -> tuple[GroundingRef, ...]:
    """Preserve every declared reference; invalid declarations stay explicit."""

    return parse_grounding_refs(value)


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
    "CRITICAL_CRITIQUE_DIMENSIONS",
    "CRITIQUE_ACCEPTED",
    "CRITIQUE_DEFERRED",
    "CRITIQUE_DIMENSIONS",
    "CRITIQUE_LEVELS",
    "CRITIQUE_REJECTED",
    "BlockCritique",
    "CritiqueDimensionFinding",
    "CritiqueEvaluation",
    "CritiqueRequest",
    "CritiqueResult",
    "CritiqueTarget",
    "DeterministicWritingProvider",
    "FakeCritiqueProvider",
    "FakeWritingProvider",
    "MaterializationCritiqueProvider",
    "MaterializationProviderError",
    "MaterializationScript",
    "MaterializationWritingProvider",
    "WritingRequest",
    "WritingResult",
    "WritingTarget",
    "build_writing_request",
    "materialize_script",
]
