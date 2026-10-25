"""Shared semantic-claim contract for Stage 4 planning/governance and Stage 6.

The contract distinguishes what the source establishes from interpretation,
explicitly authored source-linked suggestions, and external factual/causal
claims. It is intentionally instruction-and-evidence based: it never grants
approval, and a prefix such as ``try`` or ``اقتراح`` never changes a claim's
evidence requirement.
"""

from __future__ import annotations

import re

SEMANTIC_CLAIM_CONTRACT_VERSION = "semantic-claim-contract-v1"

SEMANTIC_CLAIM_CONTRACT = (
    "Apply this claim contract consistently. Distinguish four kinds of content and "
    "never blur them: (1) SOURCE-ESTABLISHED — the rendered selected evidence states "
    "it, including a source speaker's own opinion when clearly attributed; do not "
    "present an interpretation as the source's own. (2) SOURCE-GROUNDED "
    "INTERPRETATION — a bounded reading of the rendered evidence that may explain or "
    "frame it but may not add a new factual predicate, causal mechanism, motive, or "
    "outcome. (3) AUTHORED SOURCE-LINKED SUGGESTION — an explicitly authored, concrete "
    "action that is relevant to the rendered evidence; it may go beyond the source but "
    "must not say or imply that the speaker recommended it, that it works, or that it "
    "treats, prevents, causes, or guarantees any result, and its link to the evidence "
    "must be clear. (4) EXTERNAL FACTUAL OR CAUSAL CLAIM — any new fact, statistic, "
    "attribution, mechanism, motive, or cause-and-effect claim not established by the "
    "rendered evidence; it requires the existing verification dependency and is never "
    "approved as authored prose. A prefix such as 'try', 'suggestion', 'اقتراح', or "
    "'جرّب' changes neither the claim kind nor the evidence requirement. Irrelevant "
    "advice, invented motives, concealed factual or effectiveness claims, generic "
    "filler, and paraphrase still fail. Preserve uncertainty: when evidence is "
    "insufficient, defer or mark verification required rather than guessing."
)

# Text is normalized to lower-case words separated by single spaces. Arabic
# diacritics and punctuation are removed so phrase matching cannot be defeated by
# a prefix, punctuation, or a case change.
_DIACRITICS_RE = re.compile(r"[\u0640\u064B-\u0652\u0670]")
_NON_WORD_RE = re.compile(r"[^\w\u0600-\u06FF]+", re.UNICODE)


def normalize_claim_text(text: str) -> str:
    """Normalize text for conservative phrase matching only."""

    folded = _DIACRITICS_RE.sub("", str(text).casefold())
    return " ".join(_NON_WORD_RE.sub(" ", folded).split())


def contains_claim_phrase(text: str, phrase: str) -> bool:
    """Whole-phrase containment, never a bare substring of an Arabic word."""

    normalized_text = normalize_claim_text(text)
    normalized_phrase = normalize_claim_text(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


CAUSAL_OUTCOME_MARKERS: tuple[str, ...] = (
    "prevents",
    "prevent",
    "solves",
    "solve",
    "fixes",
    "cures",
    "cure",
    "leads to",
    "results in",
    "causes",
    "caused by",
    "because of",
    "due to",
    "guarantees",
    "guaranteed",
    "ensures",
    "proves",
    "proven",
    "mechanism",
    "root cause",
    "psychological",
    "subconscious",
    "self absorption",
    "لتجنب",
    "لتفادي",
    "يمنع",
    "منع",
    "يعالج",
    "علاج",
    "يحل",
    "الحل",
    "يضمن",
    "ضمان",
    "يؤدي",
    "يسبب",
    "بسبب",
    "نتيجة",
    "السبب",
    "ميكانيزم",
    "آلية نفسية",
    "نفسي",
    "اللاوعي",
    "التمركز حول الذات",
    "لفت الانتباه",
)

MOTIVE_MARKERS: tuple[str, ...] = (
    "needs validation",
    "seeking attention",
    "wants attention",
    "is insecure",
    "insecure",
    "narcissist",
    "subconscious need",
    "because he wants",
    "because she wants",
    "يحتاج تقدير",
    "يبحث عن الاهتمام",
    "يبحث عن انتباه",
    "يحاول يثبت نفسه",
    "يثبت نفسه",
    "بسبب شعوره",
    "يشعر بالنقص",
    "نرجسي",
    "غير واثق",
    "يحتاج اهتمام",
)

EFFECTIVENESS_MARKERS: tuple[str, ...] = (
    "proven",
    "proven to",
    "scientifically proven",
    "studies show",
    "research shows",
    "guaranteed",
    "guarantee",
    "effective",
    "works every time",
    "always works",
    "مثبت علميا",
    "الدراسات تثبت",
    "الابحاث",
    "فعال",
    "مضمون",
    "مضمونة",
    "يضمن",
    "ينجح",
    "حل فعال",
)

GENERIC_ADVICE_MARKERS: tuple[str, ...] = (
    "always smile",
    "be confident",
    "be yourself",
    "just listen",
    "communicate better",
    "think positive",
    "stay positive",
    "ابتسم",
    "كن واثقا",
    "كن نفسك",
    "استمع اكثر",
    "تواصل بشكل افضل",
    "فكر بايجابية",
    "كن ايجابيا",
)

_CAUSAL_EVIDENCE_MARKERS: tuple[str, ...] = (
    "because",
    "caused",
    "due to",
    "leads to",
    "results in",
    "so that",
    "therefore",
    "بسبب",
    "لان",
    "نتيجة",
    "يؤدي",
    "يسبب",
    "لذلك",
)

_SUGGESTION_PREFIXES: tuple[str, ...] = (
    "suggestion",
    "suggest",
    "try",
    "consider",
    "اقتراح",
    "جرب",
    "حاول",
    "فكر في",
)


def _first_unsupported_marker(
    text: str, evidence_text: str, markers: tuple[str, ...]
) -> str | None:
    for marker in markers:
        if contains_claim_phrase(text, marker) and not contains_claim_phrase(evidence_text, marker):
            return marker
    return None


def _has_any(text: str, markers: tuple[str, ...]) -> str | None:
    for marker in markers:
        if contains_claim_phrase(text, marker):
            return marker
    return None


def has_causal_evidence(evidence_text: str) -> bool:
    """Whether the cited evidence contains an explicit causal connective.

    This is a compatibility guard, not proof of entailment. It prevents a
    legitimate request to explain a cause the source itself states from being
    rejected merely because the planning wording contains ``mechanism``.
    """

    return _has_any(evidence_text, _CAUSAL_EVIDENCE_MARKERS) is not None


def is_explicit_suggestion(text: str) -> bool:
    """Whether text explicitly presents itself as authored advice.

    This never authorizes the advice; it only helps label the claim class so the
    critique applies the suggestion-specific evidence rules.
    """

    normalized = normalize_claim_text(text)
    return any(
        normalized == prefix or normalized.startswith(f"{prefix} ")
        for prefix in (normalize_claim_text(item) for item in _SUGGESTION_PREFIXES)
    ) or contains_claim_phrase(text, "conversational suggestion")


def unverified_semantic_claim_reason(text: str, evidence_text: str) -> str | None:
    """Return a bounded reason when text adds an unsupported semantic claim.

    The function is deliberately conservative and deterministic. It catches the
    production failure mode and its prefixed variants; it does not claim to
    replace semantic review for nuance, which remains required for substantive
    copy.
    """

    if not str(text).strip():
        return None
    if _has_any(text, GENERIC_ADVICE_MARKERS) is not None:
        return "GENERIC_ADVICE"
    if _first_unsupported_marker(text, evidence_text, EFFECTIVENESS_MARKERS) is not None:
        return "UNVERIFIED_EFFECTIVENESS_CLAIM"
    if _first_unsupported_marker(text, evidence_text, MOTIVE_MARKERS) is not None:
        return "UNSUPPORTED_MOTIVE"
    if _first_unsupported_marker(text, evidence_text, CAUSAL_OUTCOME_MARKERS) is not None:
        if not has_causal_evidence(evidence_text):
            return "UNSUPPORTED_CAUSAL_CLAIM"
    return None


__all__ = [
    "CAUSAL_OUTCOME_MARKERS",
    "EFFECTIVENESS_MARKERS",
    "GENERIC_ADVICE_MARKERS",
    "MOTIVE_MARKERS",
    "SEMANTIC_CLAIM_CONTRACT",
    "SEMANTIC_CLAIM_CONTRACT_VERSION",
    "contains_claim_phrase",
    "has_causal_evidence",
    "is_explicit_suggestion",
    "normalize_claim_text",
    "unverified_semantic_claim_reason",
]
