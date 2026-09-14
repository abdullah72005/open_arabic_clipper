"""Focused deterministic Stage 4.0 eligibility and strategy tests.

Pure fixtures only: no database, no network, no model loading.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.enums import (
    ContentType,
    ExternalFactRequirement,
    OriginalityRisk,
    RightsRisk,
    SemanticProviderMode,
    StrategyDisposition,
    SubstantiveValueKind,
    TransformationEligibilityOutcome,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.transformation.policy import DEFAULT_CONFIG
from app.transformation.providers import (
    TransformationProvider,
    TransformationStrategyRequest,
)
from app.transformation.service import TransformationEligibilityService
from app.transformation.types import (
    TransformationInputs,
    TransformationOutcome,
    TransformationProviderResult,
    TransformationProviderStrategy,
)


def make_inputs(**overrides: object) -> TransformationInputs:
    base: dict[str, object] = {
        "candidate_id": "candidate-1",
        "candidate_key": "key-1",
        "source_id": "source-1",
        "disposition": "CANDIDATE",
        "content_type": ContentType.INTERVIEW_INSIGHT,
        "secondary_content_types": (),
        "coarse_start": 0.0,
        "coarse_end": 45.0,
        "refined_start": 0.5,
        "refined_end": 44.0,
        "transcript": (
            "The guest argues that remote work collapsed productivity because managers "
            "lost the ability to mentor junior staff and promotion rates fell sharply."
        ),
        "transcript_confidence": 0.9,
        "refinement_confidence": 0.9,
        "word_timestamps": ({"text": "remote", "start": 1.0, "end": 1.2},),
        "unresolved_spans": (),
        "entity_evidence": (),
        "dialect_profile": "EGYPTIAN",
        "dialect_confidence": 0.8,
        "code_switch": {},
        "context_segments": ("The host asked about the future of office work.",),
        "clip_score": 0.8,
        "short_form_score": 0.75,
        "moment_density_score": 0.65,
        "ending_quality_score": 0.7,
        "loopability_score": 0.5,
        "idea_summary": "Remote work hurts junior mentorship",
        "topic_summary": "future of remote work",
        "hooks": ({"type": "DIRECT_CLAIM", "text": "Remote work collapse"},),
        "rights_risk": RightsRisk.LOW,
        "originality_risk": OriginalityRisk.NOT_INDICATED,
        "rights_status": "OWNED",
        "media_origin": "PODCAST_INTERVIEW",
        "provenance_snapshot": {},
        "refinement_priority": "CANDIDATE",
        "refinement_quality_level": "CANDIDATE",
        "refinement_status": "CANDIDATE_REFINED",
        "refinement_output_fingerprint": "refinement-fp",
        "stage3_analysis_fingerprint": "stage3-fp",
        "stage3_policy_version": "stage3-v1",
    }
    base.update(overrides)
    return TransformationInputs(**base)  # type: ignore[arg-type]


class FakeProvider:
    provider_name = "fake"

    def __init__(self, strategies: Sequence[TransformationProviderStrategy]) -> None:
        self.model = "fake-model"
        self.calls = 0
        self._strategies = strategies

    def select_tier(self, requests: Sequence[TransformationStrategyRequest]) -> str:
        return "ROUTINE"

    def discover(
        self, requests: Sequence[TransformationStrategyRequest]
    ) -> dict[str, TransformationProviderResult]:
        self.calls += 1
        return {
            requests[0].candidate_id: TransformationProviderResult(
                candidate_id=requests[0].candidate_id,
                strategies=tuple(self._strategies),
                notes="",
                confidence=0.7,
            )
        }

    def release(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "fake", "model": self.model}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()


def _provider_strategy(**overrides: object) -> TransformationProviderStrategy:
    base: dict[str, object] = {
        "strategy_type": TransformationStrategyType.ANALYSIS,
        "disposition": StrategyDisposition.RECOMMENDED,
        "intensity": TransformationIntensity.MODERATE,
        "direction_summary": "Analyze the mentorship mechanism",
        "added_value_focus": "Explain why junior mentorship collapsed in remote settings",
        "substantive_value_kind": SubstantiveValueKind.AUTHORED_THESIS,
        "preservation_requirements": (),
        "external_verification_requirement": ExternalFactRequirement.NOT_REQUIRED,
        "verification_requirements": (),
        "rejection_reasons": (),
        "confidence": 0.7,
        "retention_preservation": 0.7,
        "source_moment_damage_risk": 0.3,
        "added_value_density": 0.6,
        "originality_potential": 0.7,
        "source_dominance_risk": 0.4,
        "generic_filler_risk": 0.3,
        "redundant_commentary_risk": 0.3,
        "template_staleness_risk": 0.3,
    }
    base.update(overrides)
    return TransformationProviderStrategy(**base)  # type: ignore[arg-type]


def evaluate(
    inputs: TransformationInputs, provider: TransformationProvider | None = None
) -> TransformationOutcome:
    service = TransformationEligibilityService(
        config=DEFAULT_CONFIG,
        provider=provider,
        mode=SemanticProviderMode.ADAPTIVE
        if provider is not None
        else SemanticProviderMode.DETERMINISTIC,
    )
    return service.evaluate(inputs, input_fingerprint="input-fp")


def test_strong_viral_moment_is_eligible() -> None:
    outcome = evaluate(make_inputs())
    assert (
        outcome.eligibility_outcome is TransformationEligibilityOutcome.ELIGIBLE_FOR_TRANSFORMATION
    )
    assert outcome.recommended
    assert all(item.disposition is StrategyDisposition.RECOMMENDED for item in outcome.recommended)


def test_only_cosmetic_or_weak_moment_is_no_strategy() -> None:
    weak = make_inputs(
        content_type=ContentType.OTHER,
        transcript="generic filler words repeated over and over with nothing specific at all.",
        idea_summary="",
        topic_summary="",
        hooks=(),
        clip_score=0.2,
        short_form_score=0.2,
        moment_density_score=0.1,
        ending_quality_score=0.2,
        loopability_score=0.1,
    )
    outcome = evaluate(weak)
    assert (
        outcome.eligibility_outcome
        is TransformationEligibilityOutcome.NO_TRANSFORMATION_STRATEGY_WORTH_USING
    )
    assert outcome.recommended == ()
    assert outcome.cache_eligible is True


def test_setup_before_short_payoff_is_rejected_for_damage() -> None:
    short = make_inputs(
        coarse_end=10.0,
        refined_end=9.5,
        moment_density_score=0.9,
        transcript=(
            "He jumped from the roof into the pool because his friends dared him and "
            "everyone screamed and laughed loudly."
        ),
        idea_summary="Friends dared him to jump and he did",
        hooks=({"type": "PAYOFF_FIRST"},),
    )
    outcome = evaluate(short)
    rejected = [
        item for item in outcome.strategies if item.disposition is StrategyDisposition.REJECTED
    ]
    assert any("HOOK_PAYOFF_DAMAGE" in item.rejection_reasons for item in rejected)


def test_funny_without_grounded_value_has_no_strategy() -> None:
    funny = make_inputs(
        content_type=ContentType.FUNNY,
        transcript="He walked straight into the glass door while everyone watched and laughed.",
        idea_summary="guy walks into a glass door",
        hooks=({"type": "PAYOFF_FIRST"},),
    )
    outcome = evaluate(funny)
    assert (
        outcome.eligibility_outcome
        is TransformationEligibilityOutcome.NO_TRANSFORMATION_STRATEGY_WORTH_USING
    )
    assert outcome.recommended == ()


def test_funny_with_grounded_inference_prefers_minimal_and_preserves_retention() -> None:
    grounded = make_inputs(
        content_type=ContentType.FUNNY,
        transcript=(
            "The pigeon stole his sandwich because he left it on the bench and then "
            "everyone laughed."
        ),
        idea_summary="Pigeon stole the sandwich when he looked away",
        hooks=({"type": "PAYOFF_FIRST"},),
    )
    outcome = evaluate(grounded)
    assert outcome.recommended
    assert outcome.recommended[0].intensity is TransformationIntensity.MINIMAL
    assert outcome.recommended[0].assessments.retention_preservation >= 0.8


def test_interview_suitability_includes_evidence_analysis_counterpoint() -> None:
    outcome = evaluate(make_inputs())
    types = {item.strategy_type for item in outcome.recommended}
    assert TransformationStrategyType.SOURCE_AS_EVIDENCE in types
    assert TransformationStrategyType.ANALYSIS in types


def test_educational_prefers_explanation_and_takeaway() -> None:
    educational = make_inputs(
        content_type=ContentType.EDUCATIONAL,
        transcript="Hold the stone at 20 degrees and push the blade away five times per side.",
        idea_summary="knife sharpening angle technique",
        hooks=(),
    )
    outcome = evaluate(educational)
    types = {item.strategy_type for item in outcome.recommended}
    assert TransformationStrategyType.EXPLANATORY in types
    assert TransformationStrategyType.HOOK_PLUS_TAKEAWAY in types


def test_news_marks_external_verification_requirement() -> None:
    news = make_inputs(
        content_type=ContentType.NEWS_CURRENT_EVENT,
        transcript="Parliament passed the new budget today by a narrow margin after a late session",
        idea_summary="parliament passed the budget",
        hooks=(),
    )
    outcome = evaluate(news)
    news_strategies = [
        item
        for item in outcome.strategies
        if item.strategy_type is TransformationStrategyType.NEWS_CONTEXT
    ]
    assert news_strategies
    assert (
        news_strategies[0].external_verification_requirement
        is ExternalFactRequirement.REQUIRES_EXTERNAL_FACT_VERIFICATION
    )
    assert news_strategies[0].verification_requirements
    assert outcome.eligibility_outcome is TransformationEligibilityOutcome.ELIGIBLE_WITH_CAUTION


def test_third_party_requires_transformation_and_carries_risk() -> None:
    third_party = make_inputs(
        rights_status="THIRD_PARTY_UNKNOWN",
        originality_risk=OriginalityRisk.TRANSFORMATION_REQUIRED,
        media_origin="MOVIE_TV",
    )
    outcome = evaluate(third_party)
    assert outcome.eligibility_outcome is TransformationEligibilityOutcome.TRANSFORMATION_REQUIRED
    assert outcome.platform_risk["transformation_required"] is True
    assert outcome.recommended


def test_provenance_conflict_is_unresolved_but_unknown_is_not() -> None:
    conflict = make_inputs(provenance_snapshot={"provenance_conflict": True})
    outcome = evaluate(conflict)
    assert (
        outcome.eligibility_outcome
        is TransformationEligibilityOutcome.UNRESOLVED_POLICY_OR_PROVENANCE_RISK
    )
    assert outcome.recommended == ()
    unknown = make_inputs(rights_status="THIRD_PARTY_UNKNOWN")
    unknown_outcome = evaluate(unknown)
    assert (
        unknown_outcome.eligibility_outcome
        is not TransformationEligibilityOutcome.UNRESOLVED_POLICY_OR_PROVENANCE_RISK
    )


def test_too_uncertain_transcript_is_insufficient_confidence_without_provider() -> None:
    low = make_inputs(transcript_confidence=0.1, transcript="هو قال كده")
    provider = FakeProvider([_provider_strategy()])
    outcome = evaluate(low, provider=provider)
    assert (
        outcome.eligibility_outcome
        is TransformationEligibilityOutcome.INSUFFICIENT_TRANSCRIPT_CONFIDENCE
    )
    assert provider.calls == 0


def test_missing_context_is_insufficient_context() -> None:
    fragment = make_inputs(
        transcript="And then he left the room without saying anything else at all today",
        context_segments=(),
    )
    outcome = evaluate(fragment)
    assert outcome.eligibility_outcome is TransformationEligibilityOutcome.INSUFFICIENT_CONTEXT


def test_paraphrase_commentary_is_rejected() -> None:
    provider = FakeProvider(
        [
            _provider_strategy(
                strategy_type=TransformationStrategyType.COMMENTARY,
                direction_summary="Basically says the same thing again",
                added_value_focus="In other words, restate his point in fewer words",
            )
        ]
    )
    outcome = evaluate(make_inputs(), provider=provider)
    rejected = [
        item for item in outcome.strategies if item.disposition is StrategyDisposition.REJECTED
    ]
    assert any("PARAPHRASE_ONLY" in item.rejection_reasons for item in rejected)


def test_fake_hook_and_distortion_are_rejected() -> None:
    provider = FakeProvider(
        [
            _provider_strategy(
                direction_summary="You won't believe what happened next",
                added_value_focus="Fake drama over the same clip",
            ),
            _provider_strategy(
                strategy_type=TransformationStrategyType.SUMMARY,
                direction_summary="He utterly humiliated everyone",
                added_value_focus="Everyone is furious and destroyed",
            ),
        ]
    )
    outcome = evaluate(make_inputs(), provider=provider)
    rejected = [
        item for item in outcome.strategies if item.disposition is StrategyDisposition.REJECTED
    ]
    reasons = {reason for item in rejected for reason in item.rejection_reasons}
    assert "FAKE_DRAMATIC_HOOK" in reasons or "SOURCE_DISTORTION" in reasons


def test_presentation_only_provider_claim_gets_zero_credit() -> None:
    provider = FakeProvider(
        [
            _provider_strategy(
                strategy_type=TransformationStrategyType.SOURCE_LED_MINIMAL,
                direction_summary="Add captions and a border with zoom punch-ins",
                added_value_focus="crop reframe border emoji music",
            )
        ]
    )
    outcome = evaluate(make_inputs(), provider=provider)
    rejected = [
        item for item in outcome.strategies if item.disposition is StrategyDisposition.REJECTED
    ]
    assert any("PRESENTATION_ONLY" in item.rejection_reasons for item in rejected)


def test_strategies_contain_no_script_or_timeline_fields() -> None:
    outcome = evaluate(make_inputs())
    forbidden = {"script", "timeline", "tts", "render", "shot_list", "voiceover", "narration"}
    for item in outcome.strategies:
        assert not (forbidden & set(item.fingerprint_payload()))
        assert "script" not in item.direction_summary.casefold()


def test_multiple_candidates_do_not_receive_identical_templates() -> None:
    first = evaluate(
        make_inputs(
            candidate_id="c1",
            transcript="The guest argues remote work collapsed mentorship for juniors everywhere.",
        )
    )
    second = evaluate(
        make_inputs(
            candidate_id="c2",
            content_type=ContentType.EDUCATIONAL,
            transcript="Sharpen the knife by holding it at twenty degrees against the stone.",
            idea_summary="knife sharpening technique",
            hooks=(),
        )
    )
    first_focus = {item.added_value_focus for item in first.recommended}
    second_focus = {item.added_value_focus for item in second.recommended}
    assert first_focus != second_focus


def test_high_quality_moment_without_evidenced_value_has_no_strategy() -> None:
    """High Stage 3 scores alone never establish substantive original value."""

    high_quality = make_inputs(
        content_type=ContentType.REACTION_WORTHY,
        transcript="The car flipped three times and landed on its wheels and he walked out.",
        idea_summary="car flips and driver walks out",
        hooks=({"type": "PAYOFF_FIRST"},),
        clip_score=0.95,
        short_form_score=0.9,
        moment_density_score=0.85,
        ending_quality_score=0.9,
        loopability_score=0.8,
    )
    outcome = evaluate(high_quality)
    assert (
        outcome.eligibility_outcome
        is TransformationEligibilityOutcome.NO_TRANSFORMATION_STRATEGY_WORTH_USING
    )
    assert outcome.recommended == ()
    assert any("NO_EVIDENCED_VALUE_BASIS" in item.rejection_reasons for item in outcome.rejected)


def test_high_quality_moment_with_grounded_inference_is_eligible() -> None:
    grounded = make_inputs(
        content_type=ContentType.OTHER,
        transcript="The new tariff policy caused prices to spike because supply collapsed.",
        idea_summary="tariff policy caused price spike",
        topic_summary="tariff policy economics",
        hooks=(),
        clip_score=0.95,
        short_form_score=0.9,
        moment_density_score=0.85,
        ending_quality_score=0.9,
        loopability_score=0.8,
    )
    outcome = evaluate(grounded)
    assert (
        outcome.eligibility_outcome is TransformationEligibilityOutcome.ELIGIBLE_FOR_TRANSFORMATION
    )
    assert outcome.recommended
    assert all(item.added_value_focus for item in outcome.recommended)


def test_dialect_remains_source_evidence_not_target_market() -> None:
    outcome = evaluate(make_inputs(dialect_profile="GULF", dialect_confidence=0.7))
    payload = outcome.strategies[0].fingerprint_payload()
    assert "target_market" not in payload
    assert "localization" not in payload
    assert "dialect" not in payload
