"""Stage 3 candidate discovery, scoring, novelty, and semantic evaluation."""

from app.candidates.proposals import generate_proposals
from app.candidates.providers import (
    DeterministicSemanticProvider,
    SemanticEvaluationRequest,
    SemanticEvaluationResult,
    SemanticProvider,
)
from app.candidates.service import CandidateAnalysisService
from app.candidates.types import (
    CandidateAnalysisOutcome,
    CandidateDraft,
    CandidateScores,
    ContentClassification,
    HookRecord,
    Proposal,
    UncertaintyEvidence,
)

__all__ = [
    "CandidateAnalysisOutcome",
    "CandidateAnalysisService",
    "CandidateDraft",
    "CandidateScores",
    "ContentClassification",
    "DeterministicSemanticProvider",
    "HookRecord",
    "Proposal",
    "SemanticEvaluationRequest",
    "SemanticEvaluationResult",
    "SemanticProvider",
    "UncertaintyEvidence",
    "generate_proposals",
]
