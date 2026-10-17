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
from dataclasses import dataclass, field
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
#: GROUNDING_ENTAILMENT and SUBSTANTIVE_ADDED_VALUE additionally must be a *known
#: positive* PASS (an UNKNOWN/absent verdict defers, never approves).
CRITICAL_CRITIQUE_DIMENSIONS = frozenset(
    {
        "GROUNDING_ENTAILMENT",
        "SUBSTANTIVE_ADDED_VALUE",
        "SEMANTIC_FIDELITY",
        "UNSUPPORTED_FACTS",
        "CODE_SWITCH_PRESERVATION",
    }
)
_POSITIVE_REQUIRED_DIMENSIONS = ("GROUNDING_ENTAILMENT", "SUBSTANTIVE_ADDED_VALUE")

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
    """One categorical critique finding for a single dimension."""

    dimension: str
    level: str
    evidence: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"dimension": self.dimension, "level": self.level, "evidence": self.evidence}


@dataclass(frozen=True)
class BlockCritique:
    """Categorical critique for one authored block, with a derived decision."""

    block_index: int
    findings: tuple[CritiqueDimensionFinding, ...] = ()

    def _level(self, dimension: str) -> str | None:
        levels = self.levels()
        return levels.get(dimension)

    def levels(
        self, requested: Sequence[str] = CRITIQUE_DIMENSIONS
    ) -> dict[str, str]:
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

    def structural_issues(
        self, requested: Sequence[str] = CRITIQUE_DIMENSIONS
    ) -> list[str]:
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
        return CRITIQUE_ACCEPTED

    @property
    def accepted(self) -> bool:
        return self.verdict() == CRITIQUE_ACCEPTED

    def as_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "accepted": self.accepted,
            "verdict": self.verdict(),
            "findings": [finding.as_dict() for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "BlockCritique":
        raw = data.get("findings")
        findings: list[CritiqueDimensionFinding] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, Mapping):
                    findings.append(
                        CritiqueDimensionFinding(
                            dimension=str(item.get("dimension", "")),
                            level=str(item.get("level", "")),
                            evidence=str(item.get("evidence", "")),
                        )
                    )
        index = data.get("block_index")
        block_index = index if isinstance(index, int) and not isinstance(index, bool) else -1
        return cls(block_index=block_index, findings=tuple(findings))


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

    ``blocks`` carries the full categorical findings; ``accepted`` is the legacy
    bool-only map (kept for simple/fake providers). ``decision`` prefers the
    categorical verdict and only falls back to ``accepted`` when no categorical
    finding exists for the block.
    """

    accepted: Mapping[int, bool] = field(default_factory=dict)
    provider_identity: Mapping[str, object] = field(default_factory=dict)
    notes: str = ""
    deferred: bool = False
    reason_code: str | None = None
    blocks: Mapping[int, "BlockCritique"] = field(default_factory=dict)

    def decision(self, index: int) -> bool | None:
        block = self.blocks.get(index)
        if block is not None:
            return block.accepted
        if index in self.accepted:
            return bool(self.accepted[index])
        return None

    def verdict(self, index: int, requested: Sequence[str] = CRITIQUE_DIMENSIONS) -> str | None:
        """Tri-state verdict for one block, preferring full categorical evidence.

        A boolean-only result (a legacy/fake provider) is honored for test
        compatibility, but the production acceptance path always supplies
        categorical ``blocks`` and therefore never relies on a bare boolean.
        """

        block = self.blocks.get(index)
        if block is not None:
            return block.verdict(requested)
        if index in self.accepted:
            return CRITIQUE_ACCEPTED if self.accepted[index] else CRITIQUE_REJECTED
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
        source_context=_source_context(inputs),
    )


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
) -> MaterializationScript:
    """Run exactly one bounded writing request and validate its copy.

    Copy that is lexically novel is not accepted on novelty alone: it is routed
    to a bounded semantic critique that assesses the grounded copy. Without an
    available critique, such copy defers honestly. Each hosted raw request is
    reserved under a stable ``attempt_scope``-derived identity before invocation,
    so a replayed checkpoint or a crashed attempt never spends the budget twice.
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

    unresolved = False
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
            # honestly rather than being accepted on novelty alone.
            return MaterializationScript(
                (),
                False,
                identity,
                True,
                ("GROUNDED_INFERENCE_REQUIRES_REVIEW",),
                _combined_findings(findings_by_block, accepted_indexes, {}),
            )
        for target in review_targets:
            index = target.block_index
            block = evaluation.blocks.get(index)
            if block is not None:
                critique_by_block[index] = block
            verdict = evaluation.verdict(index)
            if verdict == CRITIQUE_ACCEPTED:
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
            _combined_findings(findings_by_block, accepted_indexes, critique_by_block),
        )

    for target in request.targets:
        if not accepted_indexes.get(target.block_index, False):
            continue
        copy = result.copies.get(target.block_index, "")
        evidence: dict[str, object] = {"accepted": True}
        block_critique = critique_by_block.get(target.block_index)
        if block_critique is not None:
            evidence["critique"] = block_critique.as_dict()
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
                validation_evidence=evidence,
            )
        )
    return MaterializationScript(
        authored_blocks=tuple(authored),
        accepted=len(authored) == len(request.targets),
        provider_identity=result.provider_identity,
        deferred=False,
        reason_codes=tuple(dict.fromkeys(reasons)),
        findings=_combined_findings(findings_by_block, accepted_indexes, critique_by_block),
    )


def _combined_findings(
    findings_by_block: Mapping[int, list[QualityFinding]],
    accepted_indexes: Mapping[int, bool],
    critique_by_block: Mapping[int, BlockCritique],
) -> tuple[QualityFinding, ...]:
    """All persisted script findings, including noncritical categorical concerns.

    Deterministic findings for blocks that were not ultimately accepted are kept
    (a favorable critique that resolved a soft inference concern drops only that
    block's own deterministic concern). Categorical critique findings are carried
    as quality findings for *every* reviewed block, so a noncritical concern
    (e.g. NATURALNESS) survives into persisted script evidence and the final
    assessment even when the block was accepted.
    """

    combined: list[QualityFinding] = []
    for index, findings in findings_by_block.items():
        if accepted_indexes.get(index, False):
            continue
        combined.extend(findings)
    for block in critique_by_block.values():
        for finding in block.findings:
            dimension = CRITIQUE_TO_QUALITY_DIMENSION.get(finding.dimension)
            level = CRITIQUE_TO_QUALITY_LEVEL.get(finding.level)
            if dimension is None or level is None or level == "SATISFIED":
                continue
            combined.append(
                QualityFinding(
                    dimension=dimension,
                    level=level,
                    reason_codes=(f"CRITIQUE_{finding.dimension}_{finding.level}",),
                    evidence={
                        "critique_dimension": finding.dimension,
                        "critique_level": finding.level,
                        "evidence": finding.evidence,
                    },
                )
            )
    return tuple(combined)


def _critique_reasons(block: BlockCritique | None) -> list[str]:
    if block is None:
        return []
    reasons: list[str] = []
    for dimension in CRITICAL_CRITIQUE_DIMENSIONS:
        level = block._level(dimension)
        if level in {"CONCERN", "FAIL"}:
            reasons.append(f"CRITIQUE_{dimension}_{level}")
    return reasons


@dataclass(frozen=True)
class CritiqueEvaluation:
    """Validated categorical critique for the requested review targets."""

    blocks: Mapping[int, BlockCritique]

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
    blocks: dict[int, BlockCritique] = {}
    for target in targets:
        block = result.blocks.get(target.block_index)
        if block is not None:
            blocks[target.block_index] = block
        elif target.block_index in result.accepted:
            # Boolean-only legacy/fake result: synthesize a block that carries
            # the boolean as complete evidence only when it is a rejection.
            accepted = bool(result.accepted[target.block_index])
            level = "PASS" if accepted else "FAIL"
            blocks[target.block_index] = BlockCritique(
                block_index=target.block_index,
                findings=tuple(
                    CritiqueDimensionFinding(dimension=dimension, level=level)
                    for dimension in CRITIQUE_DIMENSIONS
                ),
            )
    if not blocks:
        return None
    return CritiqueEvaluation(blocks=blocks)


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
