"""Stage 3 candidate discovery, scoring, novelty, and provider orchestration."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Callable

from app.candidates.classification import classify_content
from app.candidates.fingerprints import (
    candidate_analysis_input_fingerprint,
    candidate_key,
    candidate_output_fingerprint,
    provider_input_fingerprint,
    stage3_config_payload,
    stage3_policy_payload,
)
from app.candidates.hooks import generate_hooks
from app.candidates.novelty import NoveltyItem, score_novelty
from app.candidates.policy import DEFAULT_CONFIG, Stage3Config
from app.candidates.proposals import generate_proposals
from app.candidates.providers import (
    ProviderErrorCategory,
    SemanticEvaluationRequest,
    SemanticEvaluationResult,
    SemanticProvider,
    SemanticProviderError,
    rebuild_hooks,
)
from app.candidates.scoring import (
    compute_scores,
    compute_uncertainty,
    material_uncertainty,
    recompute_clip_score,
)
from app.candidates.text import analysis_segment_text
from app.candidates.types import (
    CandidateAnalysisOutcome,
    CandidateDraft,
    HookRecord,
    Proposal,
    UncertaintyEvidence,
)
from app.core.enums import (
    CandidateDisposition,
    ContentType,
    MediaOriginType,
    OriginalityRisk,
    RefinementReason,
    RightsRisk,
    RightsStatus,
    SemanticProviderMode,
)
from app.pipeline.executor import StageCancelled

_RIGHTS_LOW = {
    RightsStatus.OWNED,
    RightsStatus.LICENSED,
    RightsStatus.PERMISSION,
    RightsStatus.PUBLIC_DOMAIN,
    RightsStatus.OTHER_ALLOWED,
    RightsStatus.THIRD_PARTY_REUSE,
}
_TRANSFORMATION_ORIGINS = {
    MediaOriginType.MOVIE_TV,
    MediaOriginType.NEWS_CLIP,
    MediaOriginType.SPORTS_BROADCAST,
}
_TRANSFORMATION_RIGHTS = {
    RightsStatus.LICENSED,
    RightsStatus.PERMISSION,
    RightsStatus.PUBLIC_DOMAIN,
    RightsStatus.OTHER_ALLOWED,
    RightsStatus.THIRD_PARTY_REUSE,
}


def derive_risks(
    rights_status: RightsStatus, media_origin: MediaOriginType
) -> tuple[RightsRisk, OriginalityRisk]:
    """Derive rights and originality risk independently and conservatively."""

    if rights_status in {RightsStatus.UNKNOWN, RightsStatus.THIRD_PARTY_UNKNOWN}:
        rights_risk = RightsRisk.UNDETERMINED
    elif rights_status in _RIGHTS_LOW:
        rights_risk = RightsRisk.LOW
    else:
        rights_risk = RightsRisk.UNDETERMINED

    if rights_status in {RightsStatus.UNKNOWN, RightsStatus.THIRD_PARTY_UNKNOWN}:
        originality = OriginalityRisk.UNDETERMINED
    elif media_origin in _TRANSFORMATION_ORIGINS or rights_status in _TRANSFORMATION_RIGHTS:
        originality = OriginalityRisk.TRANSFORMATION_REQUIRED
    elif rights_status is RightsStatus.OWNED:
        originality = OriginalityRisk.NOT_INDICATED
    else:
        originality = OriginalityRisk.UNDETERMINED
    return rights_risk, originality


class CandidateAnalysisService:
    """Pure-ish Stage 3 analysis pipeline with an injectable semantic provider."""

    def __init__(
        self,
        *,
        config: Stage3Config = DEFAULT_CONFIG,
        provider: SemanticProvider | None = None,
        mode: SemanticProviderMode = SemanticProviderMode.DETERMINISTIC,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._mode = mode
        self._is_cancelled = is_cancelled or (lambda: False)

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise StageCancelled("candidate analysis cancelled")

    def analyze(
        self,
        *,
        source_id: str,
        segments: Sequence[Mapping[str, object]],
        duration: float,
        language: str | None,
        dialect_profile: str | None,
        dialect_confidence: float,
        silence_intervals: Sequence[Mapping[str, object]] = (),
        audio_features: Sequence[Mapping[str, object]] = (),
        rights_status: RightsStatus = RightsStatus.UNKNOWN,
        media_origin: MediaOriginType = MediaOriginType.OTHER,
        provenance_metadata: Mapping[str, object] | None = None,
        historical_corpus: Sequence[NoveltyItem] = (),
        reuse: Mapping[str, tuple[str, SemanticEvaluationResult]] | None = None,
    ) -> CandidateAnalysisOutcome:
        started = time.monotonic()
        proposals = generate_proposals(
            segments,
            duration=duration,
            config=self._config,
            silence_intervals=silence_intervals,
            features=audio_features,
        )
        self._check_cancelled()
        drafts = [
            self._score_proposal(
                proposal,
                segments,
                source_id=source_id,
                rights_status=rights_status,
                media_origin=media_origin,
                dialect_profile=dialect_profile,
                dialect_confidence=dialect_confidence,
                provenance_metadata=provenance_metadata or {},
            )
            for proposal in proposals
        ]
        drafts = self._apply_novelty(drafts, historical_corpus)
        self._check_cancelled()
        (
            drafts,
            provider_metrics,
            provider_status,
            cache_eligible,
        ) = self._run_provider(
            drafts,
            segments,
            source_id=source_id,
            language=language,
            dialect_profile=dialect_profile,
            dialect_confidence=dialect_confidence,
            reuse=reuse or {},
        )
        self._check_cancelled()
        drafts = self._finalize(drafts)
        self._check_cancelled()
        metrics = self._metrics(proposals, drafts, provider_metrics, time.monotonic() - started)
        output_fingerprint = candidate_output_fingerprint(
            [self._candidate_payload(draft) for draft in drafts]
        )
        return CandidateAnalysisOutcome(
            input_fingerprint="",
            output_fingerprint=output_fingerprint,
            provider_status=provider_status,
            semantic_provider_mode=self._mode.value,
            cache_eligible=cache_eligible,
            metrics=metrics,
            candidates=tuple(drafts),
        )

    # ------------------------------------------------------------------
    # deterministic phases

    def _score_proposal(
        self,
        proposal: Proposal,
        segments: Sequence[Mapping[str, object]],
        *,
        source_id: str,
        rights_status: RightsStatus,
        media_origin: MediaOriginType,
        dialect_profile: str | None,
        dialect_confidence: float,
        provenance_metadata: Mapping[str, object],
    ) -> CandidateDraft:
        classification = classify_content(proposal.text, config=self._config)
        uncertainty = compute_uncertainty(proposal, segments, config=self._config)
        scores = compute_scores(
            proposal, segments, classification, uncertainty, config=self._config
        )
        rights_risk, originality = derive_risks(rights_status, media_origin)
        draft = CandidateDraft(
            candidate_key=candidate_key(
                source_id=source_id,
                start_segment_index=proposal.start_segment_index,
                end_segment_index=proposal.end_segment_index,
            ),
            proposal=proposal,
            content=classification,
            scores=scores,
            disposition=CandidateDisposition.DO_NOT_CLIP,
            hooks=generate_hooks(proposal, segments, config=self._config),
            provenance_snapshot={
                "rights_status": rights_status.value,
                "media_origin": media_origin.value,
                "metadata_keys": sorted(str(key) for key in provenance_metadata),
            },
            rights_risk=rights_risk,
            originality_risk=originality,
            dialect_profile=dialect_profile,
            dialect_confidence=dialect_confidence,
            code_switch_suspected=bool(uncertainty.code_switch_tokens),
            transcript_excerpt=proposal.text[: self._config.max_excerpt_characters],
            evidence_snapshot=self._evidence_snapshot(proposal, segments, uncertainty),
        )
        draft.idea_summary = proposal.text[:280]
        draft.topic_summary = " ".join(proposal.text.split()[:20])[:200]
        return draft

    def _apply_novelty(
        self, drafts: list[CandidateDraft], historical_corpus: Sequence[NoveltyItem]
    ) -> list[CandidateDraft]:
        if not drafts:
            return drafts
        items = [
            NoveltyItem(
                key=draft.candidate_key,
                source_id="",
                idea_text=draft.idea_summary,
                topic_text=draft.topic_summary,
                clip_score=draft.scores.clip_score,
            )
            for draft in drafts
        ]
        results = score_novelty(items, historical_corpus, config=self._config)
        updated: list[CandidateDraft] = []
        for draft, novelty in zip(drafts, results):
            scores = replace(
                draft.scores,
                idea_novelty_score=novelty.idea_novelty_score,
                topic_novelty_score=novelty.topic_novelty_score,
                recent_semantic_similarity_risk=novelty.recent_semantic_similarity_risk,
            )
            updated.append(
                replace(
                    draft,
                    scores=scores,
                    disposition=(
                        CandidateDisposition.DO_NOT_CLIP_RECENTLY_REDUNDANT
                        if novelty.redundant
                        else CandidateDisposition.DO_NOT_CLIP
                    ),
                )
            )
        return updated

    # ------------------------------------------------------------------
    # provider phase

    def _run_provider(
        self,
        drafts: list[CandidateDraft],
        segments: Sequence[Mapping[str, object]],
        *,
        source_id: str,
        language: str | None,
        dialect_profile: str | None,
        dialect_confidence: float,
        reuse: Mapping[str, tuple[str, SemanticEvaluationResult]],
    ) -> tuple[list[CandidateDraft], dict[str, object], str, bool]:
        metrics: dict[str, object] = {
            "provider_calls": 0,
            "provider_candidates_evaluated": 0,
            "provider_tokens": {},
            "provider_failures": 0,
            "provider_rate_limits": 0,
            "provider_malformed_items": 0,
        }
        if self._mode is SemanticProviderMode.DETERMINISTIC or self._provider is None:
            return drafts, metrics, "DETERMINISTIC", True
        if self._provider_identity_unavailable():
            return drafts, metrics, "DETERMINISTIC_NO_PROVIDER", True

        identity = self._provider.runtime_identity() if self._provider else {}
        reuse_applied = self._apply_reuse(drafts, segments, reuse, identity)
        drafts = reuse_applied[0]
        metrics["provider_reused"] = reuse_applied[1]

        eligible = self._provider_eligible_indexes(drafts)
        selected = eligible[: self._config.max_provider_candidates]
        batches = _chunk(selected, self._config.provider_candidates_per_request)
        status = "PROVIDER_EVALUATED" if selected else "NO_PROVIDER_ELIGIBLE"
        cache_eligible = True
        calls = 0
        for batch in batches:
            if calls >= self._config.max_provider_calls_per_source:
                status = "PROVIDER_CALL_CAP"
                cache_eligible = False
                break
            if self._is_cancelled():
                raise StageCancelled("candidate analysis cancelled")
            requests = [
                self._build_request(drafts[index], segments, dialect_profile, dialect_confidence)
                for index in batch
            ]
            if self._is_cancelled():
                raise StageCancelled("candidate analysis cancelled")
            try:
                results = self._provider.evaluate(requests)
                calls += 1
                metrics["provider_calls"] = calls
                metrics["provider_candidates_evaluated"] = _as_int(
                    metrics.get("provider_candidates_evaluated")
                ) + len(requests)
                malformed = len(requests) - len(results)
                metrics["provider_malformed_items"] = (
                    _as_int(metrics.get("provider_malformed_items")) + malformed
                )
                self._apply_provider_results(drafts, batch, results, requests, dialect_profile)
            except SemanticProviderError as error:
                calls += 1
                metrics["provider_calls"] = calls
                if error.category is ProviderErrorCategory.RATE_LIMITED:
                    metrics["provider_rate_limits"] = (
                        _as_int(metrics.get("provider_rate_limits")) + 1
                    )
                    status = "RATE_LIMITED"
                else:
                    metrics["provider_failures"] = _as_int(metrics.get("provider_failures")) + 1
                    status = "PROVIDER_DEGRADED"
                cache_eligible = False
                break
        usage = getattr(self._provider, "usage_summary", None)
        if callable(usage):
            metrics["provider_tokens"] = usage()
        if self._is_cancelled():
            raise StageCancelled("candidate analysis cancelled")
        return drafts, metrics, status, cache_eligible

    def _provider_identity_unavailable(self) -> bool:
        identity = self._provider.runtime_identity() if self._provider else {}
        return identity.get("provider") == "deterministic" or self._provider is None

    def _apply_reuse(
        self,
        drafts: list[CandidateDraft],
        segments: Sequence[Mapping[str, object]],
        reuse: Mapping[str, tuple[str, SemanticEvaluationResult]],
        identity: Mapping[str, object],
    ) -> tuple[list[CandidateDraft], int]:
        if not reuse:
            return drafts, 0
        count = 0
        for index, draft in enumerate(drafts):
            if draft.candidate_key not in reuse:
                continue
            stored_fingerprint, result = reuse[draft.candidate_key]
            request = self._build_request(
                draft, segments, draft.dialect_profile, draft.dialect_confidence
            )
            fingerprint = self._provider_input_fingerprint(request, identity)
            if stored_fingerprint != fingerprint:
                continue
            self._apply_single_result(drafts, index, result, draft.dialect_profile)
            drafts[index].provider_input_fingerprint = fingerprint
            count += 1
        return drafts, count

    def _provider_eligible_indexes(self, drafts: Sequence[CandidateDraft]) -> list[int]:
        indexed = [
            index
            for index, draft in enumerate(drafts)
            if draft.scores.clip_score >= self._config.conflict_retention_threshold
            and draft.provider_evidence.get("accepted") is not True
        ]
        indexed.sort(key=lambda index: (-drafts[index].scores.clip_score, index))
        return indexed

    def _build_request(
        self,
        draft: CandidateDraft,
        segments: Sequence[Mapping[str, object]],
        dialect_profile: str | None,
        dialect_confidence: float,
    ) -> SemanticEvaluationRequest:
        proposal = draft.proposal
        previous = self._context(segments, proposal.start_segment_index, forward=False)
        following = self._context(segments, proposal.end_segment_index, forward=True)
        return SemanticEvaluationRequest(
            candidate_key=draft.candidate_key,
            text=draft.transcript_excerpt,
            previous_context=previous,
            following_context=following,
            start=proposal.start_time,
            end=proposal.end_time,
            feature_summary={
                "clip_score": draft.scores.clip_score,
                "moment_density_score": draft.scores.moment_density_score,
                "short_form_score": draft.scores.short_form_score,
                "boredom_risk_score": draft.scores.boredom_risk_score,
                "content_type": draft.content.primary.value,
            },
            uncertainty_summary={
                "transcript_confidence": draft.scores.transcript_confidence,
                "boundary_confidence": draft.scores.boundary_confidence,
                "uncertainty_severity": draft.scores.uncertainty_severity,
                "refinement_reasons": list(draft.scores.reasons),
            },
            dialect_profile=dialect_profile,
            dialect_confidence=dialect_confidence,
            protected_tokens=tuple(
                str(value) for value in _as_list(draft.evidence_snapshot.get("protected_tokens"))
            ),
            code_switch_tokens=tuple(
                str(value) for value in _as_list(draft.evidence_snapshot.get("code_switch_tokens"))
            ),
        )

    def _provider_input_fingerprint(
        self, request: SemanticEvaluationRequest, identity: Mapping[str, object]
    ) -> str:
        return provider_input_fingerprint(
            {
                "candidate_key": request.candidate_key,
                "text": request.text,
                "previous_context": request.previous_context,
                "following_context": request.following_context,
                "feature_summary": dict(request.feature_summary),
                "uncertainty_summary": dict(request.uncertainty_summary),
                "dialect_profile": request.dialect_profile,
                "code_switch_tokens": list(request.code_switch_tokens),
                "protected_tokens": list(request.protected_tokens),
                "provider_identity": dict(identity),
                "scoring": stage3_policy_payload(),
                "config": stage3_config_payload(self._config),
            }
        )

    def _apply_provider_results(
        self,
        drafts: list[CandidateDraft],
        indexes: Sequence[int],
        results: Mapping[str, SemanticEvaluationResult],
        requests: Sequence[SemanticEvaluationRequest],
        dialect_profile: str | None,
    ) -> None:
        identity = self._provider.runtime_identity() if self._provider else {}
        request_by_key = {request.candidate_key: request for request in requests}
        for index in indexes:
            draft = drafts[index]
            result = results.get(draft.candidate_key)
            if result is None:
                continue
            self._apply_single_result(drafts, index, result, dialect_profile)
            request = request_by_key.get(draft.candidate_key)
            if request is not None:
                drafts[index].provider_input_fingerprint = self._provider_input_fingerprint(
                    request, identity
                )

    def _apply_single_result(
        self,
        drafts: list[CandidateDraft],
        index: int,
        result: SemanticEvaluationResult,
        dialect_profile: str | None,
    ) -> None:
        draft = drafts[index]
        adjustments = result.score_adjustments

        def adjusted(name: str, base: float) -> float:
            return _clamp(base + float(adjustments.get(name, 0.0)))

        scores = replace(
            draft.scores,
            moment_density_score=adjusted(
                "moment_density_score", draft.scores.moment_density_score
            ),
            short_form_score=adjusted("short_form_score", draft.scores.short_form_score),
            ending_quality_score=adjusted(
                "ending_quality_score", draft.scores.ending_quality_score
            ),
            loopability_score=adjusted("loopability_score", draft.scores.loopability_score),
            boredom_risk_score=adjusted("boredom_risk_score", draft.scores.boredom_risk_score),
        )
        scores = replace(scores, clip_score=recompute_clip_score(scores))
        classification = classify_content(
            draft.transcript_excerpt,
            config=self._config,
            provider_primary=result.primary_content_type,
            provider_secondary=result.secondary_content_types,
        )
        provider_hooks = rebuild_hooks(
            result,
            proposal_text=draft.transcript_excerpt,
            context_text="",
            config=self._config,
        )
        hooks = _merge_hooks(draft.hooks, provider_hooks, self._config.max_hooks_per_candidate)
        drafts[index] = replace(
            draft,
            scores=scores,
            content=classification,
            hooks=hooks,
            idea_summary=result.idea_summary or draft.idea_summary,
            topic_summary=result.topic_summary or draft.topic_summary,
            provider_evidence={
                "accepted": True,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "score_adjustments": dict(result.score_adjustments),
                "primary_content_type": (
                    result.primary_content_type.value if result.primary_content_type else None
                ),
                "secondary_content_types": [item.value for item in result.secondary_content_types],
                "idea_summary": result.idea_summary,
                "topic_summary": result.topic_summary,
                "hooks": [hook.as_dict() for hook in provider_hooks],
            },
            analysis_fingerprint="",
        )

    # ------------------------------------------------------------------
    # finalization

    def _finalize(self, drafts: list[CandidateDraft]) -> list[CandidateDraft]:
        ordered = sorted(drafts, key=lambda item: item.scores.clip_score, reverse=True)
        retained = 0
        finalized: list[CandidateDraft] = []
        for draft in ordered:
            disposition = draft.disposition
            if disposition is CandidateDisposition.DO_NOT_CLIP_RECENTLY_REDUNDANT:
                finalized.append(draft)
                continue
            if draft.scores.clip_score < self._config.retention_threshold:
                finalized.append(replace(draft, disposition=CandidateDisposition.DO_NOT_CLIP))
                continue
            if retained >= self._config.max_retained_candidates:
                finalized.append(replace(draft, disposition=CandidateDisposition.DO_NOT_CLIP))
                continue
            retained += 1
            uncertainty = _uncertainty_from_draft(draft)
            if material_uncertainty(draft.scores, uncertainty, self._config):
                finalized.append(
                    replace(
                        draft,
                        disposition=CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
                        refinement_reasons=uncertainty.reasons,
                        refinement_evidence=draft.evidence_snapshot,
                    )
                )
            else:
                finalized.append(
                    replace(
                        draft,
                        disposition=CandidateDisposition.CANDIDATE,
                        refinement_reasons=(),
                        refinement_evidence={},
                    )
                )
        return finalized

    def _metrics(
        self,
        proposals: Sequence[Proposal],
        drafts: Sequence[CandidateDraft],
        provider_metrics: Mapping[str, object],
        duration: float,
    ) -> dict[str, object]:
        disposition_counts: dict[str, int] = {}
        content_counts: dict[str, int] = {}
        for draft in drafts:
            disposition_counts[draft.disposition.value] = (
                disposition_counts.get(draft.disposition.value, 0) + 1
            )
            content_counts[draft.content.primary.value] = (
                content_counts.get(draft.content.primary.value, 0) + 1
            )
        retained = [
            draft
            for draft in drafts
            if draft.disposition
            in {
                CandidateDisposition.CANDIDATE,
                CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
            }
        ]
        return {
            "proposals_generated": len(proposals),
            "candidates_retained": len(retained),
            "candidates_rejected": len(drafts) - len(retained),
            "refinement_needed_count": sum(
                1
                for draft in drafts
                if draft.disposition is CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT
            ),
            "redundant_count": sum(
                1
                for draft in drafts
                if draft.disposition is CandidateDisposition.DO_NOT_CLIP_RECENTLY_REDUNDANT
            ),
            "disposition_counts": disposition_counts,
            "content_type_distribution": content_counts,
            "processing_duration": duration,
            **provider_metrics,
        }

    def _candidate_payload(self, draft: CandidateDraft) -> dict[str, object]:
        return {
            "candidate_key": draft.candidate_key,
            "start_segment_index": draft.proposal.start_segment_index,
            "end_segment_index": draft.proposal.end_segment_index,
            "disposition": draft.disposition.value,
            "clip_score": round(draft.scores.clip_score, 6),
            "provider_input_fingerprint": draft.provider_input_fingerprint,
            "hooks": [hook.as_dict() for hook in draft.hooks],
        }

    def _evidence_snapshot(
        self,
        proposal: Proposal,
        segments: Sequence[Mapping[str, object]],
        uncertainty: UncertaintyEvidence,
    ) -> dict[str, object]:
        return {
            "boundary_reason": proposal.boundary_reason,
            "start_segment_index": proposal.start_segment_index,
            "end_segment_index": proposal.end_segment_index,
            "segment_count": len(proposal.segment_indexes),
            "low_confidence_spans": list(uncertainty.low_confidence_spans[:20]),
            "unresolved_segment_indexes": list(uncertainty.unresolved_segment_indexes[:50]),
            "protected_tokens": list(uncertainty.protected_tokens[:20]),
            "code_switch_tokens": list(uncertainty.code_switch_tokens[:20]),
            "low_confidence_word_span_ratio": uncertainty.low_confidence_word_span_ratio,
            "unresolved_ratio": uncertainty.unresolved_ratio,
        }

    def _context(
        self,
        segments: Sequence[Mapping[str, object]],
        boundary: int,
        *,
        forward: bool,
    ) -> str:
        indexes = (
            range(boundary + 1, min(len(segments), boundary + 3))
            if forward
            else range(max(0, boundary - 2), boundary)
        )
        text = " ".join(
            analysis_segment_text(segments[index]).strip()
            for index in indexes
            if analysis_segment_text(segments[index]).strip()
        )
        return text[: self._config.provider_context_characters]


def _uncertainty_from_draft(draft: CandidateDraft) -> UncertaintyEvidence:
    evidence = draft.evidence_snapshot
    reasons: list[RefinementReason] = []
    for value in draft.scores.reasons:
        try:
            reasons.append(RefinementReason(value))
        except ValueError:
            continue
    return UncertaintyEvidence(
        transcript_confidence=draft.scores.transcript_confidence,
        low_confidence_spans=tuple(
            dict(item)
            for item in _as_list(evidence.get("low_confidence_spans"))
            if isinstance(item, Mapping)
        ),
        unresolved_segment_indexes=tuple(
            int(value)
            for value in _as_list(evidence.get("unresolved_segment_indexes"))
            if isinstance(value, int) and not isinstance(value, bool)
        ),
        low_confidence_word_span_ratio=_as_float(evidence.get("low_confidence_word_span_ratio")),
        unresolved_ratio=_as_float(evidence.get("unresolved_ratio")),
        protected_tokens=tuple(str(value) for value in _as_list(evidence.get("protected_tokens"))),
        code_switch_tokens=tuple(
            str(value) for value in _as_list(evidence.get("code_switch_tokens"))
        ),
        code_switch_uncertainty=RefinementReason.CODE_SWITCH_UNCERTAINTY.value
        in draft.scores.reasons,
        reasons=tuple(reasons),
    )


def _merge_hooks(
    deterministic: Sequence[HookRecord], provider: Sequence[HookRecord], limit: int
) -> list[HookRecord]:
    merged: list[HookRecord] = []
    seen: set[str] = set()
    for hook in [*provider, *deterministic]:
        marker = (hook.text or "").casefold()
        if marker and marker in seen:
            continue
        if marker:
            seen.add(marker)
        merged.append(hook)
        if len(merged) >= limit:
            break
    return merged


def _chunk(values: Sequence[int], size: int) -> list[list[int]]:
    if size <= 0:
        return [list(values)] if values else []
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _clamp(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def build_input_fingerprint_payload(
    *,
    source_id: str,
    content_hash: str | None,
    duration: float,
    rights_status: RightsStatus,
    media_origin: MediaOriginType,
    provenance_metadata: Mapping[str, object],
    transcript_input_fingerprint: str,
    transcription_revision: int,
    normalization_fingerprint: str,
    reconstruction_fingerprint: str,
    reconstruction_status: str,
    reconstruction_version: str,
    segments: Sequence[Mapping[str, object]],
    language: str | None,
    dialect_profile: str | None,
    dialect_confidence: float,
    dialect_policy_version: str,
    audio_input_fingerprint: str,
    silence_intervals: Sequence[Mapping[str, object]],
    audio_features: Sequence[Mapping[str, object]],
    quality_input_fingerprint: str,
    config: Stage3Config,
    provider_identity: Mapping[str, object],
    novelty_digest: str,
) -> str:
    """Build the complete stable Stage 3 input dependency payload."""

    segment_payload = []
    for index, segment in enumerate(segments):
        segment_payload.append(
            {
                "index": index,
                "text": analysis_segment_text(segment),
                "operator_text": segment.get("operator_text"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "words": segment.get("words"),
                "avg_logprob": segment.get("avg_logprob"),
                "no_speech_prob": segment.get("no_speech_prob"),
                "reconstruction_status": segment.get("reconstruction_status"),
                "needs_refinement": segment.get("needs_refinement"),
                "reconstruction_method": segment.get("reconstruction_method"),
                "correction_applied": segment.get("correction_applied"),
                "correction_confidence": segment.get("correction_confidence"),
                "dialect_profile": segment.get("dialect_profile"),
                "dialect_confidence": segment.get("dialect_confidence"),
                "dialect_policy_version": segment.get("dialect_policy_version"),
                "code_switch_suspected": segment.get("code_switch_suspected"),
                "code_switch_tokens": segment.get("code_switch_tokens"),
            }
        )
    return candidate_analysis_input_fingerprint(
        {
            "source_id": source_id,
            "content_hash": content_hash,
            "duration": duration,
            "rights_status": rights_status.value,
            "media_origin": media_origin.value,
            "provenance_metadata": dict(provenance_metadata),
            "transcript_input_fingerprint": transcript_input_fingerprint,
            "transcription_revision": transcription_revision,
            "normalization_fingerprint": normalization_fingerprint,
            "reconstruction_fingerprint": reconstruction_fingerprint,
            "reconstruction_status": reconstruction_status,
            "reconstruction_version": reconstruction_version,
            "language": language,
            "dialect_profile": dialect_profile,
            "dialect_confidence": dialect_confidence,
            "dialect_policy_version": dialect_policy_version,
            "audio_input_fingerprint": audio_input_fingerprint,
            "silence_intervals": list(silence_intervals),
            "audio_features": list(audio_features),
            "quality_input_fingerprint": quality_input_fingerprint,
            "segments": segment_payload,
            "policy": stage3_policy_payload(),
            "config": stage3_config_payload(config),
            "provider_identity": dict(provider_identity),
            "novelty_digest": novelty_digest,
        }
    )


def semantic_result_from_evidence(
    evidence: Mapping[str, object], candidate_key_value: str
) -> SemanticEvaluationResult | None:
    """Rebuild a reusable provider evaluation from persisted evidence."""

    if evidence.get("accepted") is not True:
        return None
    primary = _content_type(evidence.get("primary_content_type"))
    secondary = tuple(
        item
        for item in (
            _content_type(value) for value in _as_list(evidence.get("secondary_content_types"))
        )
        if item is not None
    )
    adjustments: dict[str, float] = {}
    for key, value in _as_mapping(evidence.get("score_adjustments")).items():
        if isinstance(value, int | float) and not isinstance(value, bool):
            adjustments[str(key)] = float(value)
    hooks = tuple(item for item in _as_list(evidence.get("hooks")) if isinstance(item, Mapping))
    return SemanticEvaluationResult(
        candidate_key=candidate_key_value,
        primary_content_type=primary,
        secondary_content_types=secondary,
        score_adjustments=adjustments,
        idea_summary=str(evidence.get("idea_summary") or ""),
        topic_summary=str(evidence.get("topic_summary") or ""),
        hooks=hooks,
        confidence=_as_float(evidence.get("confidence")),
        explanation=str(evidence.get("explanation") or ""),
    )


def _content_type(value: object) -> ContentType | None:
    if isinstance(value, ContentType):
        return value
    if isinstance(value, str):
        try:
            return ContentType(value)
        except ValueError:
            return None
    return None


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _as_float(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0
