"""Deterministic meaning-oriented novelty and duplicate control.

Deliberately small: TF-IDF/cosine over stable Arabic/English-compatible
unigrams/bigrams with canonical idea/topic signatures. No vector database,
embedding service, or sentence-transformer model is introduced. Cross-source
comparison is repository-wide only because account/channel/history models do not
exist yet; recurring channel-output diversity and publication-history dedup are
deferred to Stage 7.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.candidates.policy import DEFAULT_CONFIG, Stage3Config
from app.candidates.text import token_feature_counts, tokenize


@dataclass(frozen=True)
class NoveltyItem:
    key: str
    source_id: str
    idea_text: str
    topic_text: str
    clip_score: float


@dataclass(frozen=True)
class NoveltyResult:
    idea_novelty_score: float
    topic_novelty_score: float
    recent_semantic_similarity_risk: float
    redundant: bool


def idea_signature(text: str) -> str:
    keys = sorted(token_feature_counts(text))
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def topic_signature(text: str) -> str:
    keys = sorted({token for token in tokenize(text)})
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def score_novelty(
    items: Sequence[NoveltyItem],
    corpus: Sequence[NoveltyItem] = (),
    *,
    config: Stage3Config = DEFAULT_CONFIG,
) -> list[NoveltyResult]:
    """Compute idea/topic novelty and recent-corpus duplication risk per item."""

    if not items:
        return []
    bounded_corpus = list(corpus)[: config.novelty_corpus_limit]
    idea_vectors = _tfidf([*items, *bounded_corpus], bigrams=True)
    topic_vectors = _tfidf([*items, *bounded_corpus], bigrams=False)
    offset = len(items)
    results: list[NoveltyResult] = []
    for index, item in enumerate(items):
        same_idea = [
            _cosine(idea_vectors[index], idea_vectors[other])
            for other in range(len(items))
            if other != index
        ]
        same_topic = [
            _cosine(topic_vectors[index], topic_vectors[other])
            for other in range(len(items))
            if other != index
        ]
        recent = [
            _cosine(idea_vectors[index], idea_vectors[offset + position])
            for position in range(len(bounded_corpus))
        ]
        max_idea = max(same_idea, default=0.0)
        max_topic = max(same_topic, default=0.0)
        max_recent = max(recent, default=0.0)
        redundant = _is_redundant(index, item, items, same_idea, max_recent, config)
        results.append(
            NoveltyResult(
                idea_novelty_score=_clamp(1.0 - max_idea),
                topic_novelty_score=_clamp(1.0 - max_topic),
                recent_semantic_similarity_risk=_clamp(max_recent),
                redundant=redundant,
            )
        )
    return results


def _is_redundant(
    index: int,
    item: NoveltyItem,
    items: Sequence[NoveltyItem],
    same_source_similarities: Sequence[float],
    max_recent: float,
    config: Stage3Config,
) -> bool:
    if max_recent >= config.cross_source_duplicate_threshold:
        return True
    for other, similarity in enumerate(same_source_similarities):
        actual = other if other < index else other + 1
        if similarity < config.same_source_duplicate_threshold:
            continue
        candidate = items[actual]
        if item.clip_score < candidate.clip_score:
            return True
        if item.clip_score == candidate.clip_score and index > actual:
            return True
    return False


def _tfidf(texts: Sequence[NoveltyItem], *, bigrams: bool) -> list[dict[str, float]]:
    if not texts:
        return []
    documents = [
        _feature_counts(item.idea_text if bigrams else item.topic_text, bigrams=bigrams)
        for item in texts
    ]
    document_frequency: dict[str, int] = {}
    for document in documents:
        for key in document:
            document_frequency[key] = document_frequency.get(key, 0) + 1
    total = len(documents)
    vectors: list[dict[str, float]] = []
    for document in documents:
        vector: dict[str, float] = {}
        for key, count in document.items():
            idf = math.log((1.0 + total) / (1.0 + document_frequency[key])) + 1.0
            vector[key] = (1.0 + math.log(count)) * idf
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm > 0:
            vector = {key: value / norm for key, value in vector.items()}
        vectors.append(vector)
    return vectors


def _feature_counts(text: str, *, bigrams: bool) -> dict[str, int]:
    counts = token_feature_counts(text)
    if bigrams:
        return counts
    return {key: value for key, value in counts.items() if key.startswith("1:")}


def _cosine(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    if not first or not second:
        return 0.0
    if len(first) > len(second):
        first, second = second, first
    return _clamp(sum(value * second.get(key, 0.0) for key, value in first.items()))


def _clamp(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))
