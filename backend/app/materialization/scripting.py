"""Stage 6 deterministic authored-copy scripting, slot resolution, and claim
ledger.

This module is the deterministic gate between provider/operator copy and the
authored timeline. It never calls a provider: it validates already-produced copy,
classifies claims, resolves required/optional slots, and records the exact
evidence a reviewer needs. A bounded semantic critique (see ``providers.py``) is
consulted only where deterministic lexical checks cannot resolve support,
usefulness, or Arabic naturalness.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.core.enums import ClaimClass, QualityLevel, SubstantiveValueKind
from app.materialization.inputs import MaterializationInputs
from app.materialization.policy import (
    REASON_GENERIC_FILLER,
    REASON_INVENTED_NUMBER,
    REASON_PARAPHRASE_ONLY,
    REASON_PRESENTATION_ONLY,
    REASON_PROTECTED_TOKEN_MUTATED,
    REASON_UNSUPPORTED_ESSENTIAL_CLAIM,
    Stage6Config,
)
from app.materialization.types import (
    AuthoredBlock,
    ClaimLedgerEntry,
    GroundingRef,
    QualityFinding,
    SlotResolution,
)
from app.transcription.dialect import extract_protected_tokens
from app.transformation.planning.policy import (
    GENERIC_VALUE_MARKERS,
    PARAPHRASE_SCAFFOLD_MARKERS,
    PRESENTATION_ONLY_CHANGES,
    PRESENTATION_ONLY_TERMS,
)

_WORD_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_NUMBER_RE = re.compile(r"[0-9\u0660-\u0669]+(?:[.,\u066b\u066c][0-9\u0660-\u0669]+)*")

_SUBSTANTIVE_BLOCK_TYPES = frozenset({"ORIGINAL_VALUE", "TEXTUAL_ANNOTATION"})
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "that",
        "this",
        "it",
        "as",
        "for",
        "with",
        "at",
        "by",
        "from",
        "be",
        "been",
        "which",
        "who",
        "why",
        "how",
        "في",
        "من",
        "على",
        "ال",
        "هذا",
        "هذه",
        "الذي",
        "التي",
        "و",
        "او",
        "أو",
    }
)


@dataclass(frozen=True)
class BlockValidation:
    """Deterministic validation result for one authored block."""

    block_index: int
    findings: tuple[QualityFinding, ...]
    claim: ClaimLedgerEntry | None
    accepted: bool

    @property
    def blocking(self) -> bool:
        return any(finding.level == QualityLevel.UNMET.value for finding in self.findings)


def _content_tokens(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).casefold() for match in _WORD_RE.finditer(text))
        if token and token not in _STOPWORDS
    ]


def _has_marker(text: str, markers: Sequence[str]) -> bool:
    folded = text.casefold()
    return any(marker.casefold() in folded for marker in markers)


def _numbers(text: str) -> set[str]:
    return {match.group(0) for match in _NUMBER_RE.finditer(text)}


def grounding_source_text(inputs: MaterializationInputs, refs: Sequence[GroundingRef]) -> str:
    """Resolve the actual source text a grounding reference points at."""

    chunks: list[str] = []
    for ref in refs:
        if ref.block_index is not None:
            block = inputs.plan_block(ref.block_index)
            if block is not None and block.get("block_type") == "SOURCE_EXCERPT":
                text = str(block.get("source_text", "") or "")
                if text:
                    chunks.append(text)
        if ref.kind == "FINAL_CLIP":
            chunks.append(inputs.final_clip.text)
    if not chunks and inputs.final_clip.text:
        chunks.append(inputs.final_clip.text)
    return " ".join(chunks).strip()


def classify_claim(
    copy: str,
    *,
    grounding_text: str,
    verification_dependency_id: str | None,
) -> ClaimClass:
    """Classify authored copy into the claim ledger vocabulary."""

    if verification_dependency_id:
        return ClaimClass.EXTERNAL_REQUIRES_VERIFICATION
    if not grounding_text.strip():
        # Copy that cites no source evidence and needs no verification dependency
        # is treated as a clearly-framed inference, never silently "supported".
        return ClaimClass.SOURCE_GROUNDED_INFERENCE
    copy_numbers = _numbers(copy)
    if copy_numbers and not copy_numbers.issubset(_numbers(grounding_text)):
        return ClaimClass.EXTERNAL_REQUIRES_VERIFICATION
    copy_tokens = set(_content_tokens(copy))
    source_tokens = set(_content_tokens(grounding_text))
    novel = copy_tokens - source_tokens
    if novel and len(novel) >= max(1, len(copy_tokens) // 4):
        return ClaimClass.SOURCE_GROUNDED_INFERENCE
    return ClaimClass.SOURCE_SUPPORTED


def validate_authored_block(
    block: Mapping[str, object],
    inputs: MaterializationInputs,
    config: Stage6Config,
) -> BlockValidation:
    """Deterministically validate one authored block's final copy."""

    index = _int(block.get("block_index", block.get("index", -1)))
    copy = str(block.get("copy", "") or "").strip()
    kind = str(block.get("substantive_value_kind", ""))
    block_type = str(block.get("block_type", ""))
    refs = _grounding_refs(block.get("grounding_refs"))
    dependency_id = _optional_str(block.get("verification_dependency_id"))
    grounding_text = grounding_source_text(inputs, refs)

    findings: list[QualityFinding] = []
    if not copy:
        findings.append(
            QualityFinding(
                dimension="SLOT_FULFILLMENT",
                level=QualityLevel.UNMET.value,
                reason_codes=("MISSING_REQUIRED_VALUE",),
                evidence={"block_index": index},
            )
        )

    if copy:
        if _has_marker(copy, PRESENTATION_ONLY_TERMS) or _has_marker(
            copy, tuple(PRESENTATION_ONLY_CHANGES)
        ):
            findings.append(_finding("SUBSTANTIVE_VALUE", REASON_PRESENTATION_ONLY, index))
        if _has_marker(copy, GENERIC_VALUE_MARKERS) or _has_marker(
            copy, PARAPHRASE_SCAFFOLD_MARKERS
        ):
            findings.append(_finding("SUBSTANTIVE_VALUE", REASON_GENERIC_FILLER, index))

        if grounding_text:
            if _is_paraphrase_only(copy, grounding_text, config):
                findings.append(_finding("SUBSTANTIVE_VALUE", REASON_PARAPHRASE_ONLY, index))

            copy_numbers = _numbers(copy)
            invented = copy_numbers - _numbers(grounding_text)
            if invented:
                findings.append(
                    QualityFinding(
                        dimension="FIDELITY_ATTRIBUTION",
                        level=QualityLevel.UNMET.value,
                        reason_codes=(REASON_INVENTED_NUMBER,),
                        evidence={"block_index": index, "numbers": sorted(invented)},
                    )
                )

            copy_protected = extract_protected_tokens(copy)
            mutated = _protected_token_mutations(copy_protected, grounding_text)
            if mutated:
                findings.append(
                    QualityFinding(
                        dimension="FIDELITY_ATTRIBUTION",
                        level=QualityLevel.UNMET.value,
                        reason_codes=(REASON_PROTECTED_TOKEN_MUTATED,),
                        evidence={"block_index": index, "tokens": sorted(mutated)},
                    )
                )

    claim_class = classify_claim(
        copy, grounding_text=grounding_text, verification_dependency_id=dependency_id
    )
    essential = block_type in _SUBSTANTIVE_BLOCK_TYPES and kind not in {
        SubstantiveValueKind.SOURCE_AS_EVIDENCE.value,
    }
    if claim_class == ClaimClass.EXTERNAL_REQUIRES_VERIFICATION and essential:
        findings.append(
            QualityFinding(
                dimension="GROUNDING_VERIFICATION",
                level=QualityLevel.UNMET.value,
                reason_codes=(REASON_UNSUPPORTED_ESSENTIAL_CLAIM,),
                evidence={"block_index": index, "verification_dependency_id": dependency_id},
            )
        )

    claim = ClaimLedgerEntry(
        claim_id=str(block.get("claim_id", f"claim-{index}")),
        block_index=index,
        claim_class=claim_class.value,
        text=copy,
        grounding_refs=refs,
        verification_dependency_id=dependency_id,
        resolved=claim_class != ClaimClass.EXTERNAL_REQUIRES_VERIFICATION,
        essential=essential,
    )
    accepted = bool(copy) and not any(
        finding.level == QualityLevel.UNMET.value for finding in findings
    )
    return BlockValidation(
        block_index=index, findings=tuple(findings), claim=claim, accepted=accepted
    )


def _protected_token_mutations(copy_tokens: Sequence[str], source: str) -> set[str]:
    """Copy tokens that look like a mutated source protected token.

    A genuinely new authored word (``mechanism``, ``mentorship``) is allowed; a
    copy token that differs from a *distinctive* source protected token (a name,
    number, abbreviation, technical form, or long token) by a single edit (case
    aside) is treated as corruption and rejected. Ordinary short words are never
    treated as protected identities, so ``loss`` is not a mutation of ``lost``.
    """

    source_tokens = {
        token.casefold(): token
        for token in extract_protected_tokens(source)
        if _is_distinctive(token)
    }
    mutated: set[str] = set()
    for token in copy_tokens:
        folded = token.casefold()
        if folded in source_tokens:
            continue
        for source_folded, _source_token in source_tokens.items():
            if abs(len(folded) - len(source_folded)) > 1:
                continue
            if _edit_distance(folded, source_folded) == 1:
                mutated.add(token)
                break
    return mutated


def _is_distinctive(token: str) -> bool:
    if any(character.isdigit() for character in token):
        return True
    if any(character in "-_/+@#.:" for character in token):
        return True
    if any(character.isupper() for character in token):
        return True
    return len(token) >= 8


def _edit_distance(left: str, right: str) -> int:
    if abs(len(left) - len(right)) > 1:
        return 2
    if len(left) < len(right):
        left, right = right, left
    # len(left) >= len(right)
    if len(left) == len(right):
        return sum(1 for a, b in zip(left, right, strict=True) if a != b)
    # One insertion in the longer string.
    index = 0
    while index < len(right) and left[index] == right[index]:
        index += 1
    return 1 if left[index + 1 :] == right[index:] else 2


def _is_paraphrase_only(copy: str, source: str, config: Stage6Config) -> bool:
    copy_tokens = _content_tokens(copy)
    if not copy_tokens:
        return False
    source_tokens = set(_content_tokens(source))
    shared = sum(1 for token in copy_tokens if token in source_tokens)
    overlap = shared / len(copy_tokens)
    novel = {token for token in copy_tokens if token not in source_tokens}
    threshold = getattr(config, "paraphrase_overlap_threshold", 0.7)
    return overlap >= threshold and not novel


def _finding(dimension: str, reason: str, index: int) -> QualityFinding:
    return QualityFinding(
        dimension=dimension,
        level=QualityLevel.UNMET.value,
        reason_codes=(reason,),
        evidence={"block_index": index},
    )


def _grounding_refs(value: object) -> tuple[GroundingRef, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    refs: list[GroundingRef] = []
    for item in value:
        if isinstance(item, GroundingRef):
            refs.append(item)
        elif isinstance(item, Mapping):
            refs.append(
                GroundingRef(
                    kind=str(item.get("kind", "")),
                    block_index=_optional_int(item.get("block_index")),
                    source_start=_optional_float(item.get("source_start")),
                    source_end=_optional_float(item.get("source_end")),
                    claim_id=_optional_str(item.get("claim_id")),
                )
            )
    return tuple(refs)


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


def resolve_slots(
    inputs: MaterializationInputs,
    authored: Sequence[AuthoredBlock],
    *,
    narration_satisfied_blocks: frozenset[int] = frozenset(),
) -> tuple[SlotResolution, ...]:
    """Resolve every required authored/contract slot explicitly.

    Required substantive blocks must be fulfilled by an authored block or, when
    the plan allows narration and a single audio asset fulfils both, by an
    accepted narration chunk. Optional slots are recorded as declined rather than
    silently dropped.
    """

    authored_by_index = {block.block_index: block for block in authored}
    resolutions: list[SlotResolution] = []
    for block in inputs.blocks:
        index = _int(block.get("index", -1))
        block_type = str(block.get("block_type", ""))
        if block_type not in _SUBSTANTIVE_BLOCK_TYPES:
            continue
        slot_id = f"block-{index}"
        if index in authored_by_index:
            resolutions.append(
                SlotResolution(
                    slot_id=slot_id,
                    block_index=index,
                    slot_kind=block_type,
                    resolution="FULFILLED",
                )
            )
        elif index in narration_satisfied_blocks:
            resolutions.append(
                SlotResolution(
                    slot_id=slot_id,
                    block_index=index,
                    slot_kind=block_type,
                    resolution="FULFILLED",
                )
            )
        else:
            resolutions.append(
                SlotResolution(
                    slot_id=slot_id,
                    block_index=index,
                    slot_kind=block_type,
                    resolution="DEFERRED",
                    reason_code="MISSING_REQUIRED_VALUE",
                )
            )
    return tuple(resolutions)


def all_required_slots_fulfilled(resolutions: Sequence[SlotResolution]) -> bool:
    return all(item.resolution == "FULFILLED" for item in resolutions)


__all__ = [
    "BlockValidation",
    "all_required_slots_fulfilled",
    "classify_claim",
    "grounding_source_text",
    "resolve_slots",
    "validate_authored_block",
]
