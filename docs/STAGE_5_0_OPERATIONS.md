# Stage 5.0 — Execution preflight and render contract

Stage 5.0 is deterministic execution preflight plus a durable, evidence-bound
execution/render contract. It does **not** render, generate content, or use AI.
All Stage 3/3.5/3.7 and Stage 4.0–4.3 behavior is frozen; Stage 5.0 only adds the
`app/render` package, one enum group, one table, one migration, API/CLI
endpoints, and configuration.

## Boundaries

Stage 5.0 must never:

- render, cut/join, encode, normalize loudness, produce a final MP4, or run
  render QC/black-frame detection;
- detect faces, track speakers, crop/reframe, smooth crops, lay out
  multi-speaker frames, or blur backgrounds;
- generate captions/ASS/libass/subtitles, overlay text, or manipulate stored
  text with Unicode/BiDi controls;
- generate narration wording, write scripts, run TTS, or choose a voice,
  provider, or model;
- select B-roll/gameplay/supporting visuals;
- publish, schedule, or produce metadata;
- call Gemini, Qwen, Whisper, Ollama, or any network service;
- enqueue `FINAL_CLIP` refinement, replan/re-govern/re-select Stage 4, patch a
  plan, or advance the source lifecycle;
- add a `PipelineStage`, an `_NEXT_STAGE` entry, a Celery task, a `ProcessingJob`
  kind, or a `PipelineRun`.

The only external process Stage 5.0 may invoke is one bounded read-only ffprobe
metadata probe, and only through the injectable `prober` seam (tests never
require ffprobe). Gemini calls = 0; Qwen loads = 0.

## Input and position

Stage 5.0 consumes the read-only Stage 4.3 execution handoff
(`build_execution_handoff`) plus live rows, validates publication-quality
`FINAL_CLIP` evidence, performs bounded source-media preflight, and persists one
current `RenderContract` per candidate/input fingerprint. It is synchronous,
transaction-safe, provider-free, and candidate-scoped. No Celery, no jobs.

`policy.py` owns versions, closed statuses/outcomes/reason codes, tolerances,
protected semantic operators (negation/exclusivity/modality, English + Arabic),
contraction expansions, filler tokens, the render-profile registry, and
`Stage50Config`/`stage50_config_payload()`.

## Statuses

| Status | Meaning |
| --- | --- |
| `BLOCKED` | No executable selection (missing/not-selected/stale selection, missing/not-current plan, unresolved required verification). |
| `FINAL_CLIP_REFINEMENT_REQUIRED` | Selected plan exists but no usable `FINAL_CLIP` (`FINAL_TRANSCRIPT_READY` + non-empty transcript). Never enqueues refinement. |
| `UPSTREAM_REVALIDATION_REQUIRED` | `FINAL_CLIP` exists but compatibility is material/unresolved. Never replans/re-governs/re-selects. |
| `INVALID_SOURCE_BINDING` | A plan excerpt no longer binds safely to the `FINAL_CLIP` or media bounds. |
| `SOURCE_MEDIA_UNAVAILABLE` | Managed source artifact missing/unmanaged/corrupt/no video/no audio/invalid duration/changed during preflight. |
| `READY_FOR_RENDER_PLANNING` | Executable contract, all materialization slots already materialized. |
| `MATERIALIZATION_REQUIRED` | Executable contract, but at least one authored/narration slot is not materialized. |

Only `READY_FOR_RENDER_PLANNING` and `MATERIALIZATION_REQUIRED` persist
`contract_ready=true` and a non-empty `contract_payload`. Every other status
persists `contract_ready=false` and `contract_payload={}`, and is still a
deterministic reusable preflight row. There is no persisted
`COMPATIBILITY_CHECK_REQUIRED`; `build_execution_handoff()["compatibility_recheck_required"]`
remains a live read-only flag.

## Preflight order

1. Candidate exists, else `None` (API 404).
2. Selection must be `PLAN_SELECTED`/`PLAN_SELECTED_WITH_CAUTION`, current,
   effective, and `VERIFIED_CURRENT`; a newer usable `FINAL_CLIP`
   (`REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK`) is reconciled through
   compatibility instead of blocking. Otherwise `BLOCKED`.
3. Required verification must be resolved; otherwise `BLOCKED`.
4. A usable `FINAL_CLIP` must exist; otherwise `FINAL_CLIP_REFINEMENT_REQUIRED`.
5. Exact match = stored planning refinement identity + output fingerprint equal
   the live `FINAL_CLIP`; still runs structural binding validation.
6. Full compatibility runs otherwise.
7. Source-media identity + cached probe reuse + bounded ffprobe.
8. Rebound spans validated against media duration (0.75 s tolerance).
9. Contract payload, slots, caption/framing/profile, readiness, fingerprints,
   then persistence.

## FINAL_CLIP compatibility

Deterministic and pure over persisted evidence; provider timestamps are never
trusted. Per `SOURCE_EXCERPT`: structural validation, bounded token alignment
(`difflib`, 8 s window, max 600 words, ≥ 0.5 coverage), wording comparison over
analysis-normalized tokens (NFKC, diacritics/tatweel stripped, alif/ya unified,
Latin casefolded, typographic/alternate apostrophes folded to ASCII, contractions
expanded), protected semantic operators
(negation/exclusivity/modality; English + Arabic), digit/numeric-entity change,
entity change, recovered code-switch tokens, change-ratio bands
(≤0.20 compatible, ≤0.50 unresolved, >0.50 material), complete-thought and
window-clipping boundary checks, timing-drift bands (≤1.5 bounded, ≤3.0
boundary-adjusted, >3.0 material), payoff/hook coverage, grounding-quote
preservation, meaning-critical unresolved spans, and word-evidence sufficiency.

An added Latin token is admitted as a recovered code-switch **only** when it is
listed in `FINAL_CLIP` code-switch evidence (case-folded) or the planning excerpt
itself contains Arabic-script tokens. An arbitrary inserted English
intensifier/hedge in an English excerpt is never silently recovered; it flows
through the normal operator/number/entity/change-ratio path.

Outcome precedence (most severe first): `SOURCE_SPAN_NO_LONGER_VALID` >
`UNRESOLVED_COMPATIBILITY` > `MATERIAL_SEMANTIC_CHANGE` >
`MATERIAL_TIMING_CHANGE` > `COMPATIBLE_NON_MATERIAL_CHANGE` > `EXACT_MATCH`.
Compatible outcomes proceed; `SOURCE_SPAN_NO_LONGER_VALID` maps to
`INVALID_SOURCE_BINDING`; the rest map to `UPSTREAM_REVALIDATION_REQUIRED`.

## Source-span rebinding

Each `SOURCE_EXCERPT` becomes a `BoundSourceSpan` (planning bounds/text, rebound
`FINAL_CLIP` bounds/word indexes/text, role, hero flag, verdict,
preservation constraints). The frozen Stage 4.1 plan is never mutated; the
binding lives only in the contract. Caption/quote material downstream must use
`final_clip_text`/`FINAL_CLIP` words, never the plan's copied `source_text`.

## Source media preflight

The source path must resolve inside `StorageService.source_directory(source_id)`;
empty/remote/other paths are `SOURCE_MEDIA_NOT_INGESTED`/`SOURCE_MEDIA_UNMANAGED_PATH`.
`SourceMediaIdentity` is a cheap stat identity (source id, stored content hash,
storage-relative path, size, mtime-ns) — never a full-file hash. Cached probe
facts are reused when the live identity matches a prior row
(`probe_reuse_enabled`). Probe failures map to `SOURCE_MEDIA_CORRUPT`,
`SOURCE_MEDIA_NO_VIDEO_STREAM`, `SOURCE_AUDIO_STREAM_MISSING`,
`SOURCE_MEDIA_INVALID_DURATION`, and re-stat changes fail closed as
`SOURCE_MEDIA_CHANGED_DURING_PREFLIGHT`.

## Contract blocks and materialization

Blocks preserve exact Stage 4.1 order and `plan_block_index`. Slot mapping:
`SOURCE_EXCERPT` → `SOURCE_MEDIA` (no materialization, carries the binding);
`ORIGINAL_VALUE` delivered `NARRATION` → `AUTHORED_NARRATION` (required);
`ORIGINAL_VALUE` delivered `ON_SCREEN_TEXT`/`FLEXIBLE`/none and
`TEXTUAL_ANNOTATION` → `AUTHORED_TEXT` (required); `TRANSITION` → `TRANSITION`
(no materialization); `FACT_VERIFICATION_PLACEHOLDER` → `VERIFICATION_EVIDENCE`
(no materialization once resolved). Narration `need != NONE` adds one
contract-level `AUTHORED_NARRATION` slot (`OPTIONAL` not required;
`RECOMMENDED`/`REQUIRED`/essential required). No narration slot is added for
`NONE`. A Stage 4.1 draft line is only ever an `authoring_reference`
(`draft_only=true`, `authoritative=false`), never final wording.

## Caption input and logical order

`caption_input` carries the exact `FINAL_CLIP` `final_transcript` byte-for-byte,
exact word timestamps, language (from `FINAL_CLIP` evidence else the source
transcript language, never invented), dialect profile/confidence, code-switch
and entity evidence, protected tokens, `logical_order_preserved=true`, and
`rendered_assets=null`. **No ASS/libass/subtitle rendering happens in Stage 5.0.**
Stage 5.0 performs no BiDi manipulation: stored transcript bytes are never
reversed or injected with U+202A–U+202E/U+2066–U+2069.

> **Stage 5.1 requirement:** Stage 5.1 MUST include real rendered ASS/libass
> regression testing for mixed Arabic–English captions. Earlier `<bdi>`
> frontend work does not solve subtitle rendering.

## Fingerprints

`input_fingerprint` covers candidate/source identity, live `SourceMediaIdentity`,
selection and plan identity/fingerprints, planning and `FINAL_CLIP` refinement
identity/output fingerprints, live caption-source fingerprint, live governance
verification state/unresolved, render profile, the full versioned
`stage50_config_payload()` (tolerances, operator sets, contraction expansions,
filler tokens, render profiles, compatibility policy version), and
policy/schema/fingerprint versions. `output_fingerprint` covers the full contract
payload + status.
`caption_source_fingerprint`, `source_media_fingerprint`, and `probe_fingerprint`
are separate. TTS provider/model/voice, generated narration audio, caption
font/animation, face-tracking output, crop path, B-roll, final codec tuning,
publishing metadata, and any final render artifact hash are **not** inputs and
can never invalidate a contract.

## Persistence, API, CLI

`render_contracts`: unique `(clip_candidate_id, input_fingerprint)`, one
database-current row per candidate via a partial unique index, bool checks, a
`contract_ready`/status coupling check, and FK/status indexes. Migration
`20260918_0020` (`down_revision=20260918_0019`) adds only this table.

- `POST /api/candidates/{candidate_id}/render-contract`
- `GET /api/candidates/{candidate_id}/render-contract`
- `GET /api/render-contracts/{contract_id}`
- `GET /api/candidates/{candidate_id}/stage5-1-handoff`
- `python -m app.cli render-contract <candidate_id>`
- `python -m app.cli render-contract-status <candidate_id>`
- `python -m app.cli stage5-1-handoff <candidate_id>`

## Concurrency and currentness

POST locks the candidate row (PostgreSQL `FOR UPDATE`), re-reads inputs,
reuses a matching current/historical row, marks prior rows non-current, and
recovers concurrent inserts by fingerprint. GET recomputes the **full** persisted
`input_fingerprint` from live rows (database reads plus the stat-only
`_best_effort_identity`; never ffprobe, never a provider) and returns `CURRENT`
only on exact equality, `STALE` on mismatch, and `UNVERIFIABLE` on any
recomputation failure; anything other than `CURRENT` yields `effective=false`
(fail closed). The Stage 5.1 handoff surfaces `contract.live_freshness` and
`contract.effective` and never presents a non-effective contract as ready.

## Versions

- policy `stage5.0-v3`
- schema `stage5.0-schema-v1`
- fingerprint `1`
- compatibility policy `stage5.0-compatibility-v3`
- render profile `stage5.0-render-profile-v1`
- profile `SHORTS_1080X1920` (9:16, 1080×1920, `SOURCE_COMPATIBLE` fps, 30 fallback)

## Tests

`tests/stage50_support.py` plus `test_stage50_{readiness,compatibility,binding,materialization,fingerprints,contract,api,migration,scope}.py`.
All ffprobe is faked; no provider, rendering, caption, TTS, or face-tracking
path is exercised.

Sealing patch completion: the required-test list is fully covered, including the
boundary/timing bands (`TIMING_DRIFT_BOUNDED`, `BOUNDARY_ADJUSTED`, material
timing drift, `EXCERPT_CUTS_THOUGHT`, `EXCERPT_CLIPPED_BY_WINDOW`), zero-byte and
corrupt managed source media, negative/reversed planning spans, a
`TEXTUAL_ANNOTATION` block mapped to a required `AUTHORED_TEXT` slot (no
ASS/caption file), and the unresolved-required-verification `BLOCKED` gate (unit
matrix plus a post-selection governance-snapshot mutation). These outcomes are
reachable only when wording is unchanged; a no-wording-change comparison defers
to the deterministic boundary/timing classification instead of being absorbed as
a minor wording change. Focused Stage 5.0 verification: 103 tests pass (101
default plus the 2 PostgreSQL-gated tests). `test_stage50_postgres.py` was run
against the compose PostgreSQL with `CLIPFACTORY_TEST_POSTGRES_URL` pointing at a
disposable database; both the migration upgrade/downgrade test and the
concurrent-current-row test pass.
