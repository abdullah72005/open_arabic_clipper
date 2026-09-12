# Stage 3 candidate analysis operations and design

Stage 3 finds the best potential clip moments cheaply from an imperfect
INDEX-quality transcript, preserves promising moments when transcription is
uncertain, and defers expensive audio/transcript refinement to Stage 3.5.

## Scope

Stage 3 owns source provenance evidence, content understanding, coarse candidate
discovery, candidate scoring, hook analysis, semantic duplicate/novelty control,
and the decision that a promising region deserves later refinement.

Stage 3 does **not** own targeted audio extraction or retranscription,
omitted-English recovery, publication-grade transcript refinement, exact final
clip boundaries, transformation planning, rendering, publishing, review UI, or
analytics learning. There is no vector database, embedding service, account or
channel subsystem, or global Gemini budget manager.

## Pipeline position

```
AUDIO_ANALYSIS -> READY_FOR_ANALYSIS -> CANDIDATE_ANALYSIS -> READY_FOR_REFINEMENT
```

`AUDIO_ANALYSIS` still records `READY_FOR_ANALYSIS`. Worker orchestration then
queues `CANDIDATE_ANALYSIS`. Candidate-analysis failure or cancellation leaves
the source at `READY_FOR_ANALYSIS`; success advances it to
`READY_FOR_REFINEMENT`. Existing sources already at `READY_FOR_ANALYSIS` are
analyzable through `POST /api/sources/{id}/candidate-analysis?force=` or
`python -m app.cli candidate-analysis <source-id> [--force]`.

Stage 3 consumes imperfect INDEX text. Candidate quality and transcript
confidence are separate judgments. A strong moment with material uncertainty
survives as `CANDIDATE_NEEDS_REFINEMENT`; Stage 3.5 owns targeted audio and
transcript refinement, exact boundaries, and publication-quality text.
Omitted-English audio recovery is not implemented in Stage 3.

## Source provenance

`SourceVideo.rights_status` is reused. `media_origin`
(`YOUTUBE_CREATOR_VIDEO`, `PODCAST_INTERVIEW`, `MOVIE_TV`, `NEWS_CLIP`,
`SPORTS_BROADCAST`, `OTHER`) defaults to `OTHER`, which means unclassified, not
operator-declared ownership. A small bounded `provenance_metadata` JSON object
records evidence such as creator, license/permission reference, platform
reference, and notes. API validation bounds key count, key length, and value
length. Provenance notes are never logged and are never provider prompt content.

Provenance can be set at source creation (URL JSON or multipart) and through
`PATCH /api/sources/{id}/provenance`. Duplicate source ingest never silently
mutates an existing source's provenance. Provenance changes change the Stage 3
input fingerprint and therefore invalidate Stage 3 only; they never retranscribe
or renormalize the source.

Unknown or third-party provenance never blocks local analysis, proposal
generation, scoring, or persistence. Rights/provenance risk and
originality/transformation risk are separate. Owned content is normally
`NOT_INDICATED`; licensed/permission/public-domain/third-party-reuse material and
movie/TV/news/sports material can require substantive transformation (presentation
operations such as captions, crops, zooms, cuts, or gameplay do not count).
Unknown/undeclared provenance is `UNDETERMINED` and reviewable. Stage 3 makes no
publishing, infringement, or monetization decision.

## Deterministic proposal generation

A pure, deterministic, one-pass service selects analysis text per segment in the
order operator text, final text, corrected text, then raw text (manual overrides
are authoritative and transcript fields are never modified). Coarse windows are
formed from segment/word timestamps, silence/gap midpoints, sentence punctuation,
speaker changes, question/answer patterns, contrast/topic transitions, story
build-up/payoff cues, and RMS energy changes. Fixed fallback windows are used
only when meaningful boundaries are unavailable.

Proposals are generated over a flat sequence of bounded atoms. A normal segment
is one atom; a segment longer than the maximum coarse window is split at
deterministic word/timestamp boundaries, or, when only text is available, at a
deterministic proportional-character fallback that preserves approximate
text/time correspondence. Every atom is then bounded so none exceeds the maximum
coarse window: a pathological single word/timestamp span larger than the cap is
split safely by time with its text distributed proportionally, and no oversized
word atom is retained. When the source duration is known it is the hard outer
bound, even if transcript segment/word timestamps extend beyond it. This is still
coarse discovery, never exact boundary refinement (Stage 3.5 owns final
boundaries). Each atom range has a stable span identity; `candidate_key` combines
the source UUID and the atom span, so multiple bounded windows from one oversized
transcript segment get stable, non-colliding identities across reruns. Segment
indexes, timestamps, and word/span evidence remain traceable on each persisted
candidate.

Safety caps (configurable): minimum coarse window 15 s, preferred window
35–75 s, maximum coarse window 120 s, at most 8 s surrounding context, and a
loose raw safety cap of at most 96 discovered proposals per source hour / 960
per source that exists only to protect CPU/memory during discovery. The tight
shortlist caps (at most 24 proposals per source hour, at most 240 per source) are
applied only after full deterministic scoring, classification, and novelty,
ranked by the resulting `clip_score`; clearly redundant candidates rank last so
duplicates never crowd distinct moments out of the shortlist. At most 60 retained
non-rejected candidates are persisted. These are caps, not output targets. Every
persisted proposal/candidate satisfies the configured bounds, and boundaries
always satisfy `0 <= start < end <= source_duration`. Highly overlapping,
textually similar proposals are merged; distinct ideas are not merged merely
because they share a source. Zero valid moments produces zero accepted
candidates.

## Scoring and uncertainty

Persisted independent normalized scores: `clip_score`, `short_form_score`,
`moment_density_score`, `boredom_risk_score`, `ending_quality_score`,
`loopability_score`, `engagement_confidence`, `transcript_confidence`,
`audio_confidence`, `boundary_confidence`, `uncertainty_severity`,
`idea_novelty_score`, `topic_novelty_score`, and
`recent_semantic_similarity_risk`.

`clip_score` is a content-quality aggregate of moment strength, short-form
suitability, density, ending quality, inverse boredom, and loopability. It is
computed by one shared aggregate function used by both deterministic scoring and
provider-enriched recomputation, so an accepted provider response with no score
adjustments preserves the deterministic `clip_score` exactly. `clip_score` and
the content-quality scores (`moment_density_score`, `short_form_score`,
`ending_quality_score`, `loopability_score`, `boredom_risk_score`) never include
transcript confidence, unresolved/deferred INDEX status, word confidence,
code-switch uncertainty, audio confidence, boundary confidence, or uncertainty
severity. Those remain separate fields and may only drive
`CANDIDATE_NEEDS_REFINEMENT`, `engagement_confidence`, and evidence display. A
strong moment with identical text/boundaries keeps the same content score whether
its transcript is clean or INDEX-deferred. A low-quality filler moment with
perfect transcript confidence stays low quality. Transcript confidence is derived
separately from bounded candidate evidence (word probabilities, acoustic
evidence, unresolved reconstruction state, low-confidence spans, manual
overrides); `engagement_confidence` describes evidence coverage, not quality.
After a long segment is split, word/acoustic/low-confidence and
protected/code-switch evidence is filtered to each candidate's actual coarse
time span, so evidence from one sub-window never contaminates another.

Deterministic cue/classification/hook detection runs against an analysis-only
normalized matching view: safe Unicode normalization, English case-folding,
Arabic diacritic/tatweel removal, and conservative alif/ya unification. It never
rewrites stored transcript text, corrected/final text, timestamps, numbers, names,
URLs, technical forms, protected code-switch tokens, or hook display text, and it
never changes dialect evidence or target-audience behavior. Cue vocabularies stay
small and focus on general discourse/structural signals with common Egyptian,
Gulf/Saudi, Levantine, MSA/Fusha, and English variants.

## Refinement-needed behavior

After content scoring and novelty handling: content below the retention
threshold becomes `DO_NOT_CLIP` regardless of transcript cleanliness; strong
non-redundant content with adequate transcript evidence becomes `CANDIDATE`;
strong non-redundant content with material uncertainty becomes
`CANDIDATE_NEEDS_REFINEMENT`. Bounded reason codes: `LOW_TRANSCRIPT_CONFIDENCE`,
`UNRESOLVED_INDEX_TEXT`, `LOW_CONFIDENCE_WORD_SPAN`, `CODE_SWITCH_UNCERTAINTY`,
`PROTECTED_ENTITY_UNCERTAINTY`, `LOW_BOUNDARY_CONFIDENCE`. Code switching alone
is never an error and only contributes when it overlaps low-confidence,
unresolved, or protected-token uncertainty.

## Content types and hooks

Closed content ontology: `EDUCATIONAL`, `CONTROVERSIAL_OPINION`, `FUNNY`,
`STORY`, `SURPRISING_FACT`, `EMOTIONAL`, `NEWS_CURRENT_EVENT`,
`INTERVIEW_INSIGHT`, `DEBATE`, `MOTIVATIONAL`, `TUTORIAL`, `ANALYSIS`,
`REACTION_WORTHY`, `OTHER`. A deterministic Arabic/English cue classifier picks
one primary and a small deduplicated secondary set; provider values may only be
declared enum members.

Hook types: `CURIOSITY`, `CONTRADICTION`, `QUESTION`, `DIRECT_CLAIM`,
`EMOTIONAL`, `PAYOFF_FIRST`, `CONTEXTUAL`, `SEARCH_LED`. At most three hooks per
retained proposal. Deterministic hooks use exact source-faithful sentences or
validated directions, never invented marketing claims. Provider hooks are
strictly validated against the bounded candidate/context: candidate ID must
match, the type must be declared, numerics must be finite and bounded, source
evidence must exist in the candidate/context, and every number, date, name,
Latin token, abbreviation, URL, or technical token must be supported by source
evidence. Malformed provider hooks are dropped; deterministic hooks are kept.

## Novelty and duplicates

Proposals are merged only when temporal overlap **and** transcript similarity
indicate the same moment. Same-source proposals are compared by canonical
idea/topic signatures and a repeated idea can be redundant even without overlap.
A bounded recent corpus (most recent 500 current non-rejected candidates from
other sources) provides cross-source duplication risk; a stable digest of the
corpus snapshot participates in the Stage 3 input fingerprint. Provider-produced
canonical idea/topic summaries are used when available. Provider-free mode uses
stable Arabic/English tokenization, stopword handling, unigrams/bigrams, and
TF-IDF cosine. Idea novelty, topic novelty, and recent-similarity risk are
separate. The weaker of strongly redundant candidates is marked
`DO_NOT_CLIP_RECENTLY_REDUNDANT`. Recurring channel-output diversity and
publication-history dedup are deferred to Stage 7.

Novelty is two-phase. Deterministic novelty is the cheap first pass and the
initial redundancy filter; clearly redundant candidates are excluded from
provider selection and never consume provider quota. After accepted provider
enrichment, novelty/disposition is recomputed for eligible retained candidates
using the improved summaries. Provider enrichment may refine novelty scores or
mark a newly duplicate candidate redundant, but it never revives a weak candidate
solely because it received a provider response, and an already-redundant
candidate stays redundant.

## Semantic provider strategy

Modes: `deterministic` (default; zero Gemini and zero Qwen calls), `adaptive`
(selective Gemini batches only when a key is configured; never routes to Qwen as
a fallback), and `local_only` (explicit local Qwen/Ollama only when
`CLIPFACTORY_LOCAL_QWEN_ENABLED=true`; never Gemini). Missing or misconfigured
providers always degrade to deterministic output. The Stage 3 request carries a
stable candidate ID, bounded transcript, tiny context, timing, deterministic
feature and uncertainty summaries, dialect evidence, code-switch/protected
tokens, and a `MEDIUM` priority. Hard caps: at most 32 provider-evaluated
candidates per source, 8 candidates per request, 4 actual calls per source, and
bounded input/output tokens at temperature 0. Cancellation is polled before and
after every actual request and before final success.

Provider results are reusable: a per-candidate provider-input fingerprint covers
bounded text/context, deterministic features, uncertainty evidence,
dialect/code-switch evidence, provider/model/prompt/schema identity, and
validation/scoring versions. If any requested candidate is missing a valid
accepted result (malformed/partial provider output), the analysis is
non-cache-eligible/retryable and reports `PROVIDER_PARTIAL`; already accepted
evaluations are persisted on their candidate rows and a later rerun reuses them,
calling a provider only for the missing/invalid candidates whose provider input
fingerprint still matches. Malformed output never corrupts rows: unaccepted
candidates keep their deterministic fallback data. Rate limits stop later calls
safely. The semantic provider mode is part of the persistent input fingerprint, so
switching modes invalidates Stage 3 correctly without storing secrets or
transient availability.

## Persistence and API

Tables: `candidate_analyses` (one-to-one source summary: fingerprints,
policy/scoring versions, semantic mode, stable provider identity, sanitized
status, cache eligibility, bounded metrics, duration) and `clip_candidates`
(deterministic `candidate_key`, current/stale marker, bounds, contiguous segment
indexes, bounded excerpt and evidence snapshot, content types, disposition,
queryable score columns, uncertainty fields, refinement reasons/evidence,
provenance snapshot, separate rights and originality risk, dialect/code-switch
snapshot, bounded hooks JSON, idea/topic summaries and signatures, provider
fingerprint/evidence, analysis fingerprint, timestamps). Database constraints
enforce time/segment ranges, score bounds, and `candidate_key` uniqueness.

Accepted and rejected proposals are both persisted for explainability. Rejected
proposals are not returned as accepted candidates unless explicitly requested.
Upsert on `candidate_key` preserves UUIDs for unchanged intervals; candidates no
longer emitted are marked stale only after a successful finalization, and
historical candidates are never deleted. A successful deterministic-only run
(including an absent optional Gemini key) is cache-eligible; transient provider
failure, malformed/partial output, or rate exhaustion finalizes valid
deterministic candidates with non-cache-eligible metadata so a later normal
request retries.

The Stage 3 input fingerprint covers every output-affecting input: source
identity/rights/provenance, transcript/audio/quality fingerprints, dialect
evidence, all segment evidence, semantic provider mode, stable provider identity,
novelty corpus digest, and every `Stage3Config` field that affects proposal
generation, scoring, novelty, hooks, provider request construction, provenance
bounds, or persistence. The output fingerprint covers the complete persisted
current candidate representation in stable key order: bounds/spans, disposition,
every score, classifications, hooks, uncertainty/refinement evidence,
provenance/originality risks, dialect/code-switch handoff, idea/topic summaries
and signatures, and provider evidence identity. Stage 3 fingerprints never
invalidate raw ASR, Stage 2.5, Stage 2.7, or audio analysis.

The downgrade path is safe after genuine Stage 3 use: it removes Stage-3-only
candidate/analysis rows and Stage-3-only pipeline-run/job history, maps any
`CANDIDATE_ANALYSIS`/`READY_FOR_REFINEMENT` source lifecycle back to
`READY_FOR_ANALYSIS` before narrowing enum/check constraints, and preserves all
pre-existing Stage 1/2/2.5/2.7/2.7.1 source, transcript, and audio data.

API:
- `POST /api/sources/{id}/candidate-analysis?force=`
- `GET /api/sources/{id}/candidate-analysis`
- `GET /api/sources/{id}/candidates?offset=&limit=&include_rejected=`
- `GET /api/candidates/{id}`
- `PATCH /api/sources/{id}/provenance`

CLI: `python -m app.cli candidate-analysis <source-id> [--force]` and
`python -m app.cli candidates <source-id> [--limit] [--include-rejected]`.

Responses expose scores, dispositions, refinement reasons, hooks,
provenance/originality risk, dialect/code-switch evidence, and sanitized provider
metadata without secrets or huge transcript bodies. The source detail page shows a
minimal read-only "Clip candidates" card (summary line, compact candidate rows
with time-range seek, score, disposition, content type, excerpt, and refinement
reasons, plus a show-rejected toggle) that reads the same candidate-analysis and
candidates endpoints. There is no candidate editing or refinement UI.

## Stage 3.5 handoff

Every accepted/refinement-needed candidate persists/returns: stable candidate
UUID/key, source ID, coarse start/end, target segment indexes, authoritative
INDEX excerpt, immutable raw segment/timestamp references (via the source
transcript), word/acoustic confidence summary, exact low-confidence spans,
unresolved/needs-refinement evidence, refinement reason codes and uncertainty
severity, source dialect profile/confidence, candidate-local code-switch
flags/tokens, protected name/number/date/technical-token evidence, boundary
confidence and evidence, analysis/provider/policy fingerprints, and a provenance
and transformation-risk snapshot. Stage 3.5 extracts candidate audio, refines
transcription, recovers omitted mixed-language speech, refines boundaries, and
reaches publication-quality text. See
[Stage 3.5 operations](STAGE_3_5_OPERATIONS.md) for the refinement lifecycle,
API/CLI, fingerprints, admission gate, and the typed Stage 4 handoff (Stage 4 is
not implemented).

## Configuration

See `.env.example`. Key variables: `CLIPFACTORY_CANDIDATE_SEMANTIC_MODE`,
`CLIPFACTORY_CANDIDATE_RETENTION_THRESHOLD`,
`CLIPFACTORY_CANDIDATE_UNCERTAINTY_THRESHOLD`,
`CLIPFACTORY_CANDIDATE_MAX_RETAINED`,
`CLIPFACTORY_CANDIDATE_MAX_RAW_PROPOSALS_PER_HOUR`,
`CLIPFACTORY_CANDIDATE_MAX_RAW_PROPOSALS_PER_SOURCE`,
`CLIPFACTORY_CANDIDATE_MAX_PROPOSALS_PER_HOUR`,
`CLIPFACTORY_CANDIDATE_MAX_PROPOSALS_PER_SOURCE`,
`CLIPFACTORY_CANDIDATE_MAX_PROVIDER_CANDIDATES`,
`CLIPFACTORY_CANDIDATE_PROVIDER_CANDIDATES_PER_REQUEST`,
`CLIPFACTORY_CANDIDATE_MAX_PROVIDER_CALLS`,
`CLIPFACTORY_CANDIDATE_PROVIDER_MAX_INPUT_CHARACTERS`,
`CLIPFACTORY_CANDIDATE_PROVIDER_MAX_OUTPUT_TOKENS`, and
`CLIPFACTORY_CANDIDATE_NOVELTY_CORPUS_LIMIT`. Stage 3 semantic mode defaults to
`deterministic`; Stage 2.7 reconstruction routing defaults remain unchanged.
