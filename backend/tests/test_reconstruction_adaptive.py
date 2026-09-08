from __future__ import annotations

from app.core.enums import ReconstructionStatus
from app.transcription.reconstruction.confidence import CONFIDENCE_POLICY_VERSION
from app.transcription.reconstruction.gemini import GeminiErrorCategory, GeminiProviderError
from app.transcription.reconstruction.providers import (
    ProviderResponseError,
    ReconstructionRequest,
)
from app.transcription.reconstruction.routing import (
    AdaptiveRoutingConfig,
    ReconstructionRoute,
    RoutingMode,
    route_adaptive,
)
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import (
    ConfidenceLevel,
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
)
from app.transcription.reconstruction.validation import VALIDATION_VERSION


def _words(probabilities: list[float], texts: list[str] | None = None) -> list[dict[str, object]]:
    texts = texts or [f"w{index}" for index in range(len(probabilities))]
    return [
        {"word": text, "probability": probability}
        for text, probability in zip(texts, probabilities)
    ]


def _segment(
    raw: str = "دخم",
    corrected: str | None = None,
    words: list[dict[str, object]] | None = None,
    operator: str | None = None,
) -> dict[str, object]:
    corrected = corrected or raw
    segment: dict[str, object] = {
        "start": 0.0,
        "end": 1.0,
        "text": raw,
        "raw_text": raw,
        "corrected_text": corrected,
    }
    if words is not None:
        segment["words"] = words
    if operator is not None:
        segment["operator_text"] = operator
    return segment


class FakeLocal:
    def __init__(
        self,
        candidate: ReconstructionCandidate | None = None,
        failure: type[Exception] | None = None,
        available: bool = True,
    ) -> None:
        self.calls = 0
        self.requests: list[ReconstructionRequest] = []
        self.candidate = candidate or ReconstructionCandidate(
            "provider-0", "ضخمة", provider_confidence=1.0
        )
        self.failure = failure
        self.available = available

    def health(self) -> ProviderHealth:
        if not self.available:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE, "ollama", "qwen3.5:4b", None, "not installed"
            )
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "ollama", "qwen3.5:4b", "sha256:x", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "ollama",
            "model": "qwen3.5:4b",
            "digest": "sha256:x",
            "prompt_hash": "p",
            "schema_version": "s",
            "max_context_tokens": 4096,
            "output_tokens": 256,
            "chat_framing_reserve": 64,
            "safety_reserve": 128,
            "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
            "validation_version": VALIDATION_VERSION,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def release(self) -> None:
        pass

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        self.requests.extend(requests)
        if self.failure is not None:
            raise self.failure("local failure")
        return {request.segment_index: self.candidate for request in requests}


class FakeGemini:
    def __init__(
        self,
        candidate: ReconstructionCandidate | None = None,
        error: GeminiProviderError | None = None,
        available: bool = True,
    ) -> None:
        self.model = "gemini-3.6-flash"
        self.calls = 0
        self.requests: list[ReconstructionRequest] = []
        self.candidate = candidate or ReconstructionCandidate(
            "provider-0", "ضخمة", provider_confidence=1.0
        )
        self.error = error
        self.available = available

    def health(self) -> ProviderHealth:
        if not self.available:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE, "gemini", "gemini-3.6-flash", None, "offline"
            )
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "gemini", "gemini-3.6-flash", "sha256:g", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "gemini",
            "model": "gemini-3.6-flash",
            "digest": "sha256:g",
            "prompt_hash": "p",
            "schema_version": "s",
            "timeout_seconds": 30.0,
            "retry_attempts": 0,
            "retry_backoff_seconds": 0.0,
            "max_output_tokens": 256,
            "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
            "validation_version": VALIDATION_VERSION,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def release(self) -> None:
        pass

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        self.requests.extend(requests)
        if self.error is not None:
            raise self.error
        return {request.segment_index: self.candidate for request in requests}

    def usage_summary(self) -> dict[str, int]:
        return {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15}


def _reconstructor(
    local: FakeLocal | None,
    gemini: FakeGemini | None = None,
    mode: RoutingMode = RoutingMode.ADAPTIVE,
    budget: int = 10,
) -> tuple[ContextualReconstructor, FakeLocal | None, FakeGemini | None]:
    reconstructor = ContextualReconstructor(
        local,
        gemini_provider=gemini,
        routing=AdaptiveRoutingConfig(mode=mode),
        gemini_budget=budget,
    )
    return reconstructor, local, gemini


def _run(reconstructor: ContextualReconstructor, segments: list[dict[str, object]]) -> object:
    return reconstructor.reconstruct(
        segments,
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )


def test_route_adaptive_easy_high_confidence_is_no_llm() -> None:
    segment = _segment(words=_words([0.98, 0.99], ["ده", "كلام"]))
    decision = route_adaptive(segment, AdaptiveRoutingConfig())
    assert decision.route is ReconstructionRoute.NO_LLM


def test_route_adaptive_mild_uncertainty_is_local() -> None:
    segment = _segment(words=_words([0.98, 0.60, 0.55, 0.98], ["ده", "كلام", "جديد", "مصري"]))
    decision = route_adaptive(segment, AdaptiveRoutingConfig())
    assert decision.route is ReconstructionRoute.LOCAL


def test_route_adaptive_contiguous_very_low_words_is_gemini_direct() -> None:
    segment = _segment(words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]))
    decision = route_adaptive(segment, AdaptiveRoutingConfig())
    assert decision.route is ReconstructionRoute.GEMINI_DIRECT
    assert "contiguous_very_low_words=3" in decision.evidence


def test_route_adaptive_protected_number_overlap_is_gemini_direct() -> None:
    segment = _segment(words=_words([0.40, 0.45, 0.30], ["م", "ش", "2025"]))
    decision = route_adaptive(segment, AdaptiveRoutingConfig())
    assert decision.route is ReconstructionRoute.GEMINI_DIRECT
    assert "uncertain_protected_entity_overlap" in decision.evidence


def test_route_adaptive_missing_word_evidence_keeps_conservative_local() -> None:
    decision = route_adaptive(_segment(), AdaptiveRoutingConfig(), language="ar")
    assert decision.route is ReconstructionRoute.LOCAL


def test_good_stage25_never_invokes_any_llm() -> None:
    reconstructor, local, gemini = _reconstructor(
        FakeLocal(), FakeGemini(), mode=RoutingMode.ADAPTIVE
    )
    result = _run(reconstructor, [_segment(words=_words([0.98, 0.99], ["ده", "كلام"]))])
    segment = result.segments[0]
    assert local is not None and local.calls == 0
    assert gemini is not None and gemini.calls == 0
    assert segment.route == "NO_LLM"
    assert segment.status is ReconstructionStatus.UNCHANGED_HIGH_CONFIDENCE
    assert segment.final_provider == "stage25"


def test_mild_uncertainty_uses_local_and_never_gemini_when_local_accepted() -> None:
    reconstructor, local, gemini = _reconstructor(
        FakeLocal(), FakeGemini(), mode=RoutingMode.ADAPTIVE
    )
    result = _run(
        reconstructor,
        [_segment(words=_words([0.98, 0.60, 0.55, 0.98], ["ده", "كلام", "جديد", "مصري"]))],
    )
    segment = result.segments[0]
    assert local is not None and local.calls == 1
    assert gemini is not None and gemini.calls == 0
    assert segment.applied is True
    assert segment.confidence_level is ConfidenceLevel.HIGH
    assert segment.final_provider == "ollama:qwen3.5:4b"


def test_hard_segment_uses_gemini_direct_and_skips_qwen() -> None:
    reconstructor, local, gemini = _reconstructor(
        FakeLocal(), FakeGemini(), mode=RoutingMode.ADAPTIVE
    )
    result = _run(
        reconstructor,
        [_segment(words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]))],
    )
    segment = result.segments[0]
    assert local is not None and local.calls == 0
    assert gemini is not None and gemini.calls == 1
    assert segment.applied is True
    assert segment.route == "GEMINI_DIRECT"
    assert segment.final_provider.startswith("gemini:")


def test_local_unresolved_escalates_to_gemini() -> None:
    local = FakeLocal(
        candidate=ReconstructionCandidate("provider-0", "الرئيس 70", provider_confidence=1.0)
    )
    reconstructor, _, gemini = _reconstructor(local, FakeGemini(), mode=RoutingMode.ADAPTIVE)
    result = _run(reconstructor, [_segment(raw="دخم", corrected="دخم")])
    segment = result.segments[0]
    assert gemini is not None and gemini.calls == 1
    assert segment.gemini_attempted is True
    assert segment.route == "LOCAL_THEN_GEMINI"
    assert segment.applied is True
    assert segment.escalation_reason == "local_validation_rejected"


def test_local_malformed_or_failure_escalates_to_gemini() -> None:
    local = FakeLocal(failure=ProviderResponseError)
    reconstructor, _, gemini = _reconstructor(local, FakeGemini(), mode=RoutingMode.ADAPTIVE)
    result = _run(reconstructor, [_segment(words=_words([0.98, 0.60, 0.55, 0.98]))])
    segment = result.segments[0]
    assert gemini is not None and gemini.calls == 1
    assert segment.gemini_attempted is True
    assert segment.escalation_reason == "local_provider_error"
    assert segment.applied is True


def test_gemini_absent_hard_segment_falls_back_to_local() -> None:
    reconstructor, local, _ = _reconstructor(FakeLocal(), None, mode=RoutingMode.ADAPTIVE)
    result = _run(
        reconstructor,
        [_segment(words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]))],
    )
    segment = result.segments[0]
    assert local is not None and local.calls == 1
    assert segment.local_attempted is True
    assert segment.escalation_reason == "gemini_unavailable"


def test_gemini_budget_exhausted_falls_back_to_local() -> None:
    hard_a = _segment(raw="دخم", words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]))
    hard_b = _segment(
        raw="ماشي", words=_words([0.20, 0.18, 0.25, 0.15], ["م", "ش", "ماشي", "خالص"])
    )
    reconstructor, local, gemini = _reconstructor(
        FakeLocal(), FakeGemini(), mode=RoutingMode.ADAPTIVE, budget=1
    )
    result = _run(reconstructor, [hard_a, hard_b])
    assert gemini is not None and gemini.calls == 1
    assert result.segments[0].applied is True
    assert result.segments[1].escalation_reason == "gemini_budget_exhausted"
    assert local is not None and local.calls >= 1


def test_gemini_429_is_not_retried_and_blocks_remaining_targets() -> None:
    gemini = FakeGemini(error=GeminiProviderError(GeminiErrorCategory.RATE_LIMITED))
    hard_a = _segment(raw="دخم", words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]))
    hard_b = _segment(
        raw="ماشي", words=_words([0.20, 0.18, 0.25, 0.15], ["م", "ش", "ماشي", "خالص"])
    )
    reconstructor, local, gemini = _reconstructor(
        FakeLocal(), gemini, mode=RoutingMode.ADAPTIVE, budget=10
    )
    result = _run(reconstructor, [hard_a, hard_b])
    assert gemini.calls == 1
    assert result.segments[0].gemini_result_state == "failure:RATE_LIMITED"
    assert result.segments[1].gemini_attempted is False
    assert result.segments[1].escalation_reason == "gemini_rate_limit_exhausted"


def test_gemini_timeout_falls_back_safely() -> None:
    gemini = FakeGemini(error=GeminiProviderError(GeminiErrorCategory.TIMEOUT))
    reconstructor, local, gemini = _reconstructor(FakeLocal(), gemini, mode=RoutingMode.ADAPTIVE)
    result = _run(
        reconstructor,
        [_segment(words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]))],
    )
    segment = result.segments[0]
    assert gemini.calls == 1
    assert segment.gemini_result_state == "failure:TIMEOUT"
    assert local is not None and local.calls == 1


def test_gemini_malformed_output_is_rejected() -> None:
    gemini = FakeGemini(error=GeminiProviderError(GeminiErrorCategory.MALFORMED_OUTPUT))
    reconstructor, local, gemini = _reconstructor(FakeLocal(), gemini, mode=RoutingMode.ADAPTIVE)
    result = _run(
        reconstructor,
        [_segment(words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]))],
    )
    segment = result.segments[0]
    assert segment.gemini_result_state == "failure:MALFORMED_OUTPUT"
    assert segment.applied is True  # local fallback applied


def test_gemini_refusal_falls_back_safely() -> None:
    gemini = FakeGemini(error=GeminiProviderError(GeminiErrorCategory.SAFETY_REFUSAL))
    reconstructor, local, gemini = _reconstructor(FakeLocal(), gemini, mode=RoutingMode.ADAPTIVE)
    result = _run(
        reconstructor,
        [_segment(words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]))],
    )
    assert result.segments[0].gemini_result_state == "failure:SAFETY_REFUSAL"
    assert local is not None and local.calls == 1


def test_manual_override_never_invokes_providers() -> None:
    reconstructor, local, gemini = _reconstructor(
        FakeLocal(), FakeGemini(), mode=RoutingMode.ADAPTIVE
    )
    result = _run(reconstructor, [_segment(operator="يدوي")])
    segment = result.segments[0]
    assert local is not None and local.calls == 0
    assert gemini is not None and gemini.calls == 0
    assert segment.status is ReconstructionStatus.MANUAL_OVERRIDE
    assert segment.final_provider == "operator:manual"


def test_local_only_mode_never_invokes_gemini() -> None:
    reconstructor, local, gemini = _reconstructor(
        FakeLocal(), FakeGemini(), mode=RoutingMode.LOCAL_ONLY
    )
    hard = _segment(words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]))
    result = _run(reconstructor, [hard])
    assert gemini is not None and gemini.calls == 0
    assert local is not None and local.calls == 1
    assert result.segments[0].escalation_reason == "gemini_blocked_local_only"


def test_gemini_only_skips_qwen_and_still_permits_no_llm() -> None:
    reconstructor, local, gemini = _reconstructor(
        FakeLocal(), FakeGemini(), mode=RoutingMode.GEMINI_ONLY
    )
    result = _run(
        reconstructor,
        [
            _segment(words=_words([0.98, 0.99], ["ده", "كلام"])),
            _segment(words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"])),
        ],
    )
    assert local is not None and local.calls == 0
    assert gemini is not None and gemini.calls == 1
    assert result.segments[0].route == "NO_LLM"
    assert result.segments[1].applied is True


def test_manual_override_wins_inside_gemini_only() -> None:
    reconstructor, local, gemini = _reconstructor(
        FakeLocal(), FakeGemini(), mode=RoutingMode.GEMINI_ONLY
    )
    result = _run(reconstructor, [_segment(operator="يدوي")])
    assert local is not None and local.calls == 0
    assert gemini is not None and gemini.calls == 0
    assert result.segments[0].final_provider == "operator:manual"


def test_routing_counts_are_retained_in_metadata() -> None:
    reconstructor, _, gemini = _reconstructor(FakeLocal(), FakeGemini(), mode=RoutingMode.ADAPTIVE)
    result = _run(
        reconstructor,
        [
            _segment(words=_words([0.98, 0.99], ["ده", "كلام"])),
            _segment(words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"])),
        ],
    )
    counts = result.metadata["routing_counts"]
    assert isinstance(counts, dict)
    assert counts["no_llm"] == 1
    assert counts["gemini_direct"] == 1
    assert counts["gemini_accepted"] == 1
    assert result.metadata["gemini_usage"]["total_token_count"] == 15


def test_completed_identical_work_has_identical_fingerprint_without_duplicate_call() -> None:
    segments = [_segment(words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]))]
    reconstructor, _, gemini = _reconstructor(
        FakeLocal(), FakeGemini(), mode=RoutingMode.ADAPTIVE, budget=10
    )
    first = _run(reconstructor, segments)
    second = _run(reconstructor, segments)
    assert first.fingerprint == second.fingerprint
    assert gemini is not None and gemini.calls == 2  # one per run, no extra calls


def test_changed_gemini_model_invalidates_fingerprint() -> None:
    segments = [_segment(words=_words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]))]
    base = _reconstructor(FakeLocal(), FakeGemini(), mode=RoutingMode.ADAPTIVE)
    first = _run(base[0], segments)
    other = FakeGemini()
    other_identity = dict(other.runtime_identity())
    other_identity["model"] = "gemini-other"
    replaced = ContextualReconstructor(
        FakeLocal(),
        gemini_provider=_IdentityGemini(other_identity),
        routing=AdaptiveRoutingConfig(mode=RoutingMode.ADAPTIVE),
        gemini_budget=10,
    )
    second = _run(replaced, segments)
    assert first.fingerprint != second.fingerprint


class _IdentityGemini(FakeGemini):
    def __init__(self, identity: dict[str, object]) -> None:
        super().__init__()
        self._identity = identity

    def runtime_identity(self) -> dict[str, object]:
        return dict(self._identity)


def test_secret_strings_never_appear_in_results() -> None:
    reconstructor, _, gemini = _reconstructor(FakeLocal(), FakeGemini(), mode=RoutingMode.ADAPTIVE)
    result = _run(reconstructor, [_segment(words=_words([0.30, 0.25, 0.20, 0.90]))])
    serialized = repr(result.metadata) + repr(result.segments)
    assert "api_key" not in serialized.casefold()
    assert "AIza" not in serialized
    assert gemini is not None
