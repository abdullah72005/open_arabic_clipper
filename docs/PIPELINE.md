# Stage 1, Stage 2, Stage 2.5, Stage 2.7, and Stage 3 pipeline

1. An operator submits a permitted public URL or uploads a local file.
2. The API creates a durable source and queued ingest job.
3. The worker acquires the source into managed storage and records its hash.
4. The worker calls `ffprobe` with an argument array and stores typed media
   metadata.
5. The worker extracts cached mono 16 kHz WAV audio and runs local faster-whisper.
6. Raw source/segment text, segment ordering, timestamps, and word timestamps are persisted.
7. Stage 2.5 derives conservative dialect-aware correction into separate
   corrected/final fields. A pure deterministic detector (no network, no LLM, no
   model loading, no audio decoding) classifies the source from immutable raw
   segment text into `EGYPTIAN`, `SAUDI`, `GULF`, `LEVANTINE`, `MSA`, or
   `UNKNOWN_ARABIC` (`None` = no Arabic evidence). The Egyptian lexicon and its
   optional provider apply only for confidently/explicitly EGYPTIAN sources;
   other profiles pass through unchanged. Exact Latin/technical/number tokens
   are preserved and `code_switch_suspected` evidence is persisted. It never
   realigns audio or overwrites raw evidence.
8. Stage 2.7 derives bounded contextual reconstruction through the managed local
   Ollama provider. It preserves raw text, segment timestamps, and word
   timestamps; it never creates, removes, merges, splits, or retimes segments.
   Local and Gemini providers share one dialect-neutral, preservation-first
   instruction plus validated profile-specific addenda, and the shared request
   carries the target segment's inherited effective dialect profile.
9. Timestamp-aware chunks use final operator text when present, otherwise
   corrected text; silence/quality signals are persisted separately.
10. The source reaches `READY_FOR_ANALYSIS`.
11. Stage 3 candidate analysis runs from the imperfect INDEX transcript, derives
    coarse deterministic proposals, scores content separately from transcript
    confidence, classifies content types, generates bounded source-faithful
    hooks, enforces same-source/cross-source novelty, and persists accepted and
    rejected candidates. It then advances the source to `READY_FOR_REFINEMENT`.
    Failure or cancellation leaves it at `READY_FOR_ANALYSIS`.

Stage 3 consumes imperfect INDEX-quality text. Candidate quality and transcript
confidence are separate; a strong moment with material uncertainty survives as
`CANDIDATE_NEEDS_REFINEMENT` for Stage 3.5, which owns targeted audio/transcript
refinement and exact boundaries. Source dialect is source evidence, not target
audience; code-switched text is preserved and omitted-English recovery is not
implemented in Stage 3. Unknown/third-party provenance never blocks local
analysis. Stage 3 semantic mode defaults to deterministic; `adaptive` uses
Gemini selectively when configured and `local_only` uses Qwen/Ollama only when
`CLIPFACTORY_LOCAL_QWEN_ENABLED=true`. See
`docs/STAGE_3_OPERATIONS.md` for the full design.

Each stage is persisted and idempotent. Stage runs persist canonical input and
output fingerprints; a changed upstream evidence reruns downstream derived
stages. Completed stages are skipped on resume; failures retain job and pipeline
error data for an operator retry. Unknown rights are allowed through local
ingest/probe only. Any future candidate generation, rendering, or publishing
must first pass an explicit authorization policy; it must reject `UNKNOWN`
rights by default.

Correction uses at most two neighboring segments on either side, but emits one
result per target segment only. Default operation is the local versioned lexicon.
An optional configured OpenAI-compatible local provider receives bounded batches
with stable IDs and may only approve a declared lexicon candidate; invalid,
missing, unsafe, or low-confidence output falls back to raw/lexicon text. Manual
operator text is feedback data only and does not train a model online.

Reconstruction defaults to the managed local Ollama provider (`qwen3.5:4b`) with
routing-driven, schema-validated two-pass candidates and per-candidate scores.
An unavailable or misconfigured provider persists a truthful status and leaves
Stage 2.5 final; it never blocks `READY_FOR_ANALYSIS`.

## Stage 3.5 candidate-scoped refinement

Stage 3.5 is explicit, candidate-scoped work after the source reaches
`READY_FOR_REFINEMENT`. It is **not** part of the automatic `_NEXT_STAGE` chain
and never advances every source; a source may stay `READY_FOR_REFINEMENT` while
individual candidates refine independently. It processes **candidate audio only**
— it never rediscovers clips, retranscribes a whole source, or uploads a whole
source — and extends the existing Celery/`ProcessingJob` system with a
`CANDIDATE_REFINEMENT` job kind plus a `candidate_refinements` row per
`(clip_candidate_id, priority)`. Only `CANDIDATE` (semantic quality) and
`FINAL_CLIP` (publication/caption quality) are valid; `INDEX` is rejected.

It extracts a bounded context window (candidate 5/5 s, final 8/8 s, max 150 s)
from the original source, runs targeted local faster-whisper as the mandatory
backbone (word timestamps, automatic language, no VAD, no
condition-on-previous-text, beam 5/8), converts clip-relative times to source
time once, and validates them inside the context window. Omitted English/code
switching is recovered only from actual targeted audio transcription; a
text-only model or `local_only` Qwen pass can never add omitted English.
Optional hosted `gemini-3.5-transcribe` (Interactions API, verbatim, word
timestamps) and `gemini-3.8-flash` adjudication are selective and gated by a
shared Redis priority/budget controller. Deterministic entity/ambiguity and
boundary refinement produce `CANDIDATE_REFINED`, `FINAL_TRANSCRIPT_READY`,
`NEEDS_MANUAL_TRANSCRIPT_REVIEW`, `PROVIDER_DEGRADED`, `REFINEMENT_FAILED`, or
`CANCELLED`. `FINAL_TRANSCRIPT_READY` is transcript readiness, not publishing
readiness. Stage 4 receives a typed read-only handoff but is not implemented. See
`docs/STAGE_3_5_OPERATIONS.md`.
