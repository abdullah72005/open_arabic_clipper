# Stage 6 V4 offline remediation — FAILED human acceptance

Date: 2026-10-20. Reviewed baseline: `04e1caacf171de2083b0256db28739c66e76bbc8`.

**Disposition:** the genuine 2026-10-20 package for candidate
`1827c524-e893-4769-8b22-31411c10b6e8` **FAILED human acceptance**. It remains
failed. Stage 6 is **UNFROZEN**, Stage 7 is **UNSTARTED**,
`publication_ready=false`. This note records bounded offline remediation, not a
new acceptance attempt. No hosted/provider/TTS/Qwen call, new genuine candidate,
production submission, acceptance retry, worker run, publication, or Stage 7
work was performed. Stage 5.2 was not modified.

## Four-issue before/after matrix

| Issue | Before (confirmed failure) | After (offline correction) | Class |
| --- | --- | --- | --- |
| 1. Arabic caption order | ASS carried a manual visual permutation; libass reordered it again. First line `ألماص قوي كتير`, tail `كتير طلباتهم`. Active color states moved words. | ASS carries logical Unicode text; libass owns BiDi. Static render matches an independent logical reference and differs from the old permutation. RTL/mixed active emphasis degrades to one static state with `ACTIVE_EMPHASIS_UNSAFE_BIDI`; LTR emphasis still advances without reflow. | Renderer fidelity (serialization) |
| 2. Subject-preserving framing | `(x,y,w,h)` boxes were fed to edge-coordinate geometry: self-IoU 0, recorded jump .08137 vs threshold .13346 (false join), saved face self-IoU 0. Smoothed hero path left the face outside the crop (source 585.66 s: crop x .6436–.7547 vs face x .5224–.6141). Output 1.16–3.04 s showed hand/curtain. | One internal `(x1,y1,x2,y2)` convention; self-IoU 1; recorded jump .20467 (rejected). Final smoothed/interpolated path is checked against contemporaneous subject boxes; an unsafe scene becomes `BACKGROUND_FILL` (`CROP_LOSES_SUBJECT`). Replayed saved evidence shows the subject instead of the hand/curtain. | Renderer fidelity (geometry) |
| 3. Silent ending card | 48 px DejaVu bottom-inset card (10% bottom margin), 4.00–7.72 s (48.19%), 52 chars/14 = 3.72 s. Layout findings discarded; `readable_copy` defaulted true; no measured pacing evidence; dominant silent ending passed. | Safe-area-centred `Noto Sans Arabic` 88 px card, max 3 lines, no truncation, complete copy, measured layout/reading/silence/fraction/delivery-rationale evidence. Missing/overlong/unsafe layout fails; dominant silent ending without `LENGTH_NECESSITY` defers. | Product quality (delivery/assessment) |
| 4. Promised value not delivered | Selected 8-word list fragment; authored line merely restated excess. Stage 4.2 approved with full transcript available; Stage 6 critique marked all dimensions PASS and echoed the plan intent as `adds_beyond_source`. Paraphrase detector was evaded by changed wording. | Planning/governance requests separate rendered selected evidence from nonrendered context and require content sufficiency. New critical `INTENT_FULFILLMENT` requires an exact copy quote, offered/resolvable grounding refs, and a bounded added relation. Missing/invalid evidence cannot PASS; `REDUNDANCY` is consequential for substantive blocks; `adds_beyond_source` is the reviewed relation. | Product quality (semantics) |

## Contract/version changes

- Stage 5.1: `stage5.1-framing-v2`, `stage5.1-ass-v6`,
  `stage5.1-caption-bidi-v3`; new `ACTIVE_EMPHASIS_UNSAFE_BIDI` and
  `CROP_LOSES_SUBJECT`.
- Stage 6: `stage6-authored-ass-v2`, `stage6-timeline-v2`,
  `stage6-gemini-schema-v2`; new critical `INTENT_FULFILLMENT` dimension and
  structured intent evidence.
- Stage 4.1: `stage4.1-v13`, `stage4.1-validation-v13`.
- Stage 4.2: `stage4.2-v10`, `stage4.2-validation-v12`.

Existing fingerprints invalidate affected outputs/checkpoints naturally. Source
transcripts, FINAL_CLIP word identities, historical budgets, and original
packages were not modified.

## Offline verification

Docker Python 3.12 (`oac-stage6-test`), network disabled, bind-mounted checkout,
`PYTHONPATH=/app`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`.

- `tests/test_stage51_ass.py tests/test_stage51_bidi_render.py
  tests/test_stage51_tracking.py tests/test_stage51_framing.py`: **63 passed**
  (includes the new MP4 encode order checks, recorded jump, containment fallback).
- `tests/test_stage51_analysis.py tests/test_stage51_planner.py
  tests/test_stage51_fingerprints.py tests/test_stage51_captions.py
  tests/test_stage51_bidi_order.py tests/test_stage51_geometry.py`:
  **142 passed**.
- `tests/test_stage6_providers.py tests/test_stage6_gemini_boundary.py
  tests/test_stage6_adapter_regressions.py tests/test_stage6_materialization_core.py`:
  **58 passed**.
- `tests/test_stage6_overlays.py tests/test_stage6_v4_delivery_regressions.py
  tests/test_stage6_v4_intent_regressions.py`: **19 passed**.
- `tests/test_stage6_finalization_gates.py tests/test_stage6_finalizer_evidence.py`:
  **34 passed**.
- `tests/test_stage6_results_boundary.py`: **7 passed**.
- `tests/test_stage6_cache_validation.py`: **13 passed**.
- `tests/test_stage6_rc_corrections.py tests/test_stage6_rc_v4_corrections.py
  tests/test_stage6_runtime_corrections.py tests/test_stage6_api_cli_acceptance.py`:
  **48 passed**.
- `tests/test_stage6_milestone.py`: **7 passed** (real FFmpeg authored card,
  narration, repair chain, core coexistence).
- `tests/test_stage41_providers.py tests/test_stage41_planning.py
  tests/test_stage41_contract_v4.py tests/test_stage42_providers.py
  tests/test_stage42_governance.py`: **160 passed**.
- `ruff check app tests`: clean.

These are offline contract/renderer checks. Recorded/fake model responses test
contracts, not improved live semantic judgment. No genuine acceptance is implied.

## Diagnostic artifacts (offline, explicit non-acceptance)

Directory:
`storage/benchmarks/stage-6/manual-acceptance/review-notes/2026-10-20-v4-offline-remediation/`
(`manifest.json` lists every file).

- `captions-fixed.mp4` `0bbcae5670433c3387d96759d907764c9f180543fd895cc08a860d5873984212`
  — corrected logical captions, actual acceptance lines.
- `captions-old-permutation.mp4`
  `681f78898719dc70c1123546d1da4de560f7bdb5b0a1f685a7e5ed906c766f47`
  — old double-reordered ablation.
- `framing-old-crop-585.66.png`
  `a65e4088a89c8b6047096121fdf6f43664d4c65349f592b96cd658d5e3f87d33`
  — hand/curtain, no face.
- `framing-fallback-585.66.png`
  `1d4643854cab6eed43ab2124673eaa8567285aa26fd5bda3b3395db1cb249c71`
  and `framing-fallback.mp4`
  `91999716f64f93d2fba441e5716ca3b58c763f14319ad82f6c594360be909857`
  — subject visible across the former failure window.
- `card-fixed.mp4` `c7e746b5e1a124cd7c4a1a0b32223a2b7b0de9514a5199e434135fb3db02c531`
  and `card-fixed-1.50.png`
  `3c95710b867b8e287a60948f951c9a33005fdcc47ae5a8825008f25bb42df702`
  — measured safe-area card.
- `provenance-probe.json`
  `ca18264de3328e72b99c48eac025e625936cc3c73ae27bb50bd4f940d854dfa8`.

## Follow-up corrections after the independent SOL review (2026-10-20)

Reviewed BASE: `39560e3b6c898d3bb60d4f132750ebc1821405a3`. Disposition was
**CHANGES REQUESTED** for R1–R5 in
`docs/superpowers/handoffs/2026-10-20-stage-6-v4-independent-sol-review.md`.
The offline corrections below are implemented and locally verified. They do not
rehabilitate the failed MP4 and are not acceptance evidence.

| Finding | Before (reproduced) | After (offline correction) |
| --- | --- | --- |
| R1 substantive intent bypass | `...` in a required `ORIGINAL_VALUE/SYNTHESIS` block passed deterministic validation, skipped the critique, and reached `READY_FOR_HUMAN_REVIEW` with supplied QC. | Content-free copy (punctuation, whitespace, stopwords only) is a deterministic `CONTENT_FREE` hard failure. Every essential substantive block now requires the existing semantic intent/redundancy review even if lexical checks pass. Explicit `SOURCE_AS_EVIDENCE` handling remains. No critique means honest defer. |
| R2 boolean-only positive replay | A checkpoint/provider response with `accepted:true`, empty `blocks` produced PASS for every dimension, an invented intent quote/reference and the relation `Fake critique accepted the supplied relation.` | Boolean-only entries are never promoted. Checkpoint parsing drops them and defers (`MALFORMED_CRITIQUE_EVIDENCE`); replay defers without a provider resend. Complete categorical evidence is required. Fake test adapters now supply complete evidence. |
| R3 length necessity | A PASS for block 1 authorized block 2; `FLEXIBLE` delivery was labelled `AUTHORED_TEXT`/non-silent, so a real silent card was accepted with `LENGTH_NECESSITY UNKNOWN`. | Provisional delivery follows the executable policy (`NARRATION` only when narration will actually run; otherwise a silent `STANDALONE_CARD`). Its executable duration is derived from the actual copy. `LENGTH_NECESSITY` is required for silent/optional-narration delivery and is bound to its own block, reviewed copy, delivery and duration. Sibling/stale evidence cannot authorize a card; UNKNOWN/FAIL defers before rendering. |
| R4 layout safety | A fixed `0.55 * font_size * characters` estimate declared 17 uppercase `W` readable; the encoded MP4 ink was `[0,814,995,859]` against the 54–918 safe area and clipped at the frame edge. | `build_authored_ass` rasterizes the emitted one-event ASS locally through FFmpeg/libass and measures actual glyph ink, including fontconfig fallback and outline padding. Unsafe/unbreakable/overwide copy records `AUTHORED_TEXT_EXCEEDS_SAFE_AREA`; unavailable measurement records `AUTHORED_LAYOUT_MEASUREMENT_UNAVAILABLE`. Complete copy is preserved, never truncated or shrunk. Findings reach `readable_copy`, render admission and final assessment. |
| R5 subject absence | Detections only at 0, 0.5 and 1 s still produced `TRACKED_CROP` through a 6 s hero; five unobserved seconds were treated as contained. | `_path_preserves_subjects` now rejects leading/trailing/interior intervals without subject evidence beyond the detection hold. The scene uses the existing full-source `BACKGROUND_FILL` fallback with `CROP_LOSES_SUBJECT`; recentering is never containment proof. Detection/sampling policy was not changed. |

Version changes: `stage5.1-framing-v3`, `stage6-authored-ass-v3`,
`stage6-validation-v3`. Existing fingerprints invalidate affected checkpoints and
assessments. Stage 5.2 (`backend/app/render/execution/`) was not modified.

### Follow-up verification

Docker Python 3.12, `PYTHONPATH=/app`, `PYTHONDONTWRITEBYTECODE=1`, network
disabled (`--network none`) except an isolated internal bridge for Postgres-backed
tests (no internet). Exact results:

- `tests/test_stage6_v4_followup_regressions.py
  tests/test_stage6_v4_intent_regressions.py
  tests/test_stage6_v4_delivery_regressions.py tests/test_stage6_overlays.py
  tests/test_stage6_providers.py tests/test_stage51_framing.py`: **87 passed**.
- `tests/test_stage51_ass.py tests/test_stage51_bidi_render.py
  tests/test_stage51_tracking.py tests/test_stage51_framing.py
  tests/test_stage51_bidi_order.py tests/test_stage51_fingerprints.py
  tests/test_stage51_analysis.py tests/test_stage51_planner.py
  tests/test_stage51_captions.py tests/test_stage51_geometry.py`: **212 passed**.
- `tests/test_stage6_providers.py tests/test_stage6_gemini_boundary.py
  tests/test_stage6_adapter_regressions.py tests/test_stage6_materialization_core.py
  tests/test_stage6_overlays.py tests/test_stage6_v4_delivery_regressions.py
  tests/test_stage6_v4_intent_regressions.py
  tests/test_stage6_v4_followup_regressions.py`: **99 passed**.
- `tests/test_stage6_component_recovery.py tests/test_stage6_runtime_corrections.py
  tests/test_stage6_service.py`: **19 passed**.
- `tests/test_stage6_finalization_gates.py`: **10 passed**;
  `tests/test_stage6_finalizer_evidence.py`: **24 passed**;
  `tests/test_stage6_results_boundary.py`: **7 passed**;
  `tests/test_stage6_cache_validation.py`: **13 passed** (Postgres on an isolated
  internal network).
- `tests/test_stage6_milestone.py tests/test_stage6_authored_render.py
  tests/test_stage6_authored_spec.py`: **27 passed**;
  `tests/test_stage6_real_executor.py tests/test_stage6_worker_pipeline.py`:
  **3 passed**.
- `ruff check app tests`: clean.

### Follow-up diagnostic artifacts (diagnostic-only)

Directory:
`storage/benchmarks/stage-6/manual-acceptance/review-notes/2026-10-20-v4-followup-corrections/`
(`README.md`, `manifest.json`, `before/`, `after/`). The probes are copied from
the SOL review unchanged. `before/` reproduces the reviewed failures; `after/`
shows the corrected contract outcomes. Hashes:

- `manifest.json` `ca0588c693bf82f14f7508b4f7e3cf0272c2876f77d62612ed971f5d515137b4`
  (19 files; `before/review-findings.json` is
  `5caaf54221da9061ec2e19306184d72ed1ac3e44408f9bf5c4dffb8c88d41c55`, matching
  the reviewed SOL evidence).
- `after/review-findings.json`
  `74ac6be279745ddd4e640b6e7181d0020cc08b648fa2c489b8e811e708c281b5`:
  `boolean_replay` deferred/empty; `unrelated_length_pass` `UNKNOWN`;
  wide-card `layout_failed=true`, measured ink `[0,814,996,859]`; unobserved gap
  `BACKGROUND_FILL` + `CROP_LOSES_SUBJECT`.
- `after/punctuation-probe.json`
  `cd52f5a96bd8633041e1f090bf99e1ec6c2ec60fa7469c1e77d02943e1442179`:
  `script_accepted=false`, `NO_OUTPUT`.
- `after/flexible-delivery-probe.json`
  `581e2faf6a9812ca7cb8d638f92a15eecf809d07ac99cb2e1d5518d8207036ac`:
  `STANDALONE_CARD`, `silent=true`, `length_required=true`, deferred.
- `after/wide-card.ass/mp4/png` are byte-identical to `before/` (same emitted
  text); the correction is the refusal finding, not a different asset.

These files are diagnostic reproductions only. They are not human acceptance,
not genuine semantic evidence, and do not prove execution of the R5 fallback on
the original media. The existing Stage 5.2 engine test
(`test_stage52_engine.py::test_all_six_framing_modes_compile`) remains the
frozen-renderer `BACKGROUND_FILL` execution check; no new media claim is made
here.

## Replay-binding corrections after SOL follow-up review (2026-10-21)

Reviewed BASE/HEAD: `86d6dc5632781690e2b9ebbed095a205e9fc5017`.
Disposition: **CHANGES REQUESTED — F1/F2**, both offline contract defects in
`docs/superpowers/handoffs/2026-10-21-stage-6-v4-followup-sol-review.md`.

| Finding | Before (reproduced) | After (offline correction) |
| --- | --- | --- |
| F1 legacy verdict | `CritiqueResult(accepted={1: False}).decision(1)` was `False` but `.verdict(1)` returned `ACCEPTED`. | An explicit negative boolean returns `REJECTED`, matching `decision`; positive/missing boolean evidence remains `DEFERRED`/unknown. Structured categorical evidence still takes precedence. |
| F2 replay binding | The real replay wrapper checked only the SCRIPT dependency. With identical inputs and only narration availability changed, it reused an old silent-card `LENGTH_NECESSITY PASS` with zero critic calls and relabelled it `reviewed_delivery=NARRATION`, `reviewed_duration_seconds=0.0`. | Accepted critique checkpoints persist the exact bounded request fingerprint (`stage6-critique-request-v1`) and the exact reviewed target snapshot. Reuse requires an exact request match; mismatched, legacy-unbound or malformed stored judgments defer without a provider resend. Persisted findings use the reviewed snapshot, never the current target. Exact-match replay still makes zero calls; valid writing checkpoints remain reusable. |

No new semantic pass, provider call, orchestration branch, Stage 4/5 contract or
Stage 5.2 change was added. The SCRIPT dependency and validation versions are
unchanged, so the new binding is limited to the critique checkpoint evidence.

### Verification

Docker Python 3.12 (`oac-stage6-test`), read-only backend mount, network disabled,
`PYTHONPATH=/app:/app/tests`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`.

- `tests/test_stage6_v4_replay_binding_regressions.py`: **11 passed**. With only
  the two application fixes reverted, the same file had **7 failed / 4 passed**
  (the 4 passes are the already-correct positive/missing/structured F1 cases and
  exact-match replay).
- `tests/test_stage6_v4_replay_binding_regressions.py
  tests/test_stage6_component_recovery.py
  tests/test_stage6_v4_followup_regressions.py
  tests/test_stage6_v4_delivery_regressions.py
  tests/test_stage6_v4_intent_regressions.py tests/test_stage6_providers.py
  tests/test_stage6_overlays.py tests/test_stage6_gemini_boundary.py
  tests/test_stage6_adapter_regressions.py tests/test_stage6_materialization_core.py`:
  **121 passed**.
- `tests/test_stage6_finalization_gates.py`: **10 passed**;
  `tests/test_stage6_finalizer_evidence.py`: **24 passed**;
  `tests/test_stage6_results_boundary.py`: **7 passed**.
- `ruff check app tests`: clean. Changed files are `ruff format --check` clean.

### Diagnostic artifacts (diagnostic-only)

Directory:
`storage/benchmarks/stage-6/manual-acceptance/review-notes/2026-10-21-v4-replay-binding-corrections/`
(`README.md`, `manifest.json`, `before/`, `after/`). SOL's `replay_probe.py` is
copied unchanged. `before/` reproduces the reviewed defects; `after/` shows the
bound/deferred outcomes. The probes use local provider stubs and simulated QC;
they are not semantic, video or human acceptance evidence.

- `manifest.json` hashes every archived file. Key results:
  `before/replay-probe.json`
  `6f08cfdb4ebc494236834b6cd1ff4fe38981d1e8a885b58ba4a414f38129c76f`
  (negative verdict `ACCEPTED`; stale replay accepted and relabelled);
  `after/replay-probe.json`
  `bbebc389106acf13c1663ae620d89d5d7841ab997fc692015fa3018a72e5b5e2`
  (verdict `REJECTED`; replay used false, one critic call, no stale PASS);
  `before/regressions.txt`
  `b8f204466ae4aa59ca236e3979762f35d7725915477c7432aaac55b15419cd2d`
  (7 failed / 4 passed) and `after/regressions.txt`
  `f8063f4f430824669dd389e9bbb4f6fcf9647802dca0de5b9bfdd9a56b2ea4ff`
  (11 passed).

## F3 — finalization/readiness critique binding (2026-10-21)

Reviewed BASE/HEAD `e49dfb1d46d202a2a290f459bc26af5c84456dd5`. SOL's remaining
finding was that `_accepted_script_blocks` trusted an accepted SCRIPT whose
critique checkpoint no longer carried `request_fingerprint`/`reviewed_targets`;
the downstream finalizer then produced `MATERIALIZED_WITH_CAUTION`,
human-review-required, cache-eligible, and the shared readiness view returned
`review_ready=true` without passing through the e49dfb1 replay guard.

The correction carries the existing binding into the no-prepare trust path:

- Accepted-script validation identifies blocks whose value is authorized by a
  critique: essential substantive blocks always, and any block whose accepted
  `validation_evidence` actually records a critique. Deterministic
  `SOURCE_AS_EVIDENCE`/no-critique blocks remain exempt.
- It requires the critique checkpoint's SCRIPT dependency match, canonical
  reviewed target snapshots, the exact `stage6-critique-request-v1` fingerprint
  recomputed from those snapshots plus the live source context, and complete
  categorical evidence for every reviewed block.
- The reviewed copy, claim class, substantive kind, executable delivery and
  standalone-card duration must match the accepted blocks and assembled timeline.
  A sibling/current fact cannot stand in for the reviewed fact.
- Missing, legacy-unbound, malformed or stale proof defers in finalization and
  readiness. It is never review-ready or a verified completed cache. Existing
  writing/accepted-copy checkpoints, budgets and artifacts are preserved; no
  provider is called automatically and no semantic pass or Stage 4/5 contract is
  changed. Stage 5.2 remains untouched.

Versioning reuses `stage6-critique-request-v1`; no new evidence version was
required because the binding format is unchanged and old unbound checkpoints are
refused.

Verification (offline Docker Python 3.12; network disabled except an isolated
local Postgres bridge for the existing cache suite):

- `tests/test_stage6_v4_finalization_binding_regressions.py`: **7 passed**
  (authored before the fix; observed failing).
- Combined new + e49dfb1 replay/follow-up/intent/delivery/provider/component/
  finalization-gate/finalizer-evidence batch: **112 passed**.
- Isolated-Postgres cache subset including new POST and resume cases:
  **6 passed**.
- `ruff check app tests`: clean; changed files `ruff format --check` clean.

Diagnostic artifacts (diagnostic-only):
`storage/benchmarks/stage-6/manual-acceptance/review-notes/2026-10-21-v4-finalization-binding/`.
`before/` is SOL's reproduced pre-fix output; `after/` shows the unbound case as
`accepted_after_binding_removed=false`, `finalization_outcome=DEFERRED`,
`semantic_outcome=DEFERRED_EVIDENCE`, `cache_eligible=false` and
`read_view_review_ready=false`, while the current-bound control remains
review-ready. The fixture uses SQLite, fake semantic/render-QC evidence and a real
transformed spec/authored ASS; it is **not** semantic, video or human acceptance.

## Unresolved repeated-face-box provenance (open blocker)

The original run's saved right-hand two-shot detections repeat identically at
583.16, 583.66, 584.16, 584.66 and 585.16 s. A fresh current-checkout replay
does **not** reproduce them: it already returns the close-up at 584.16 s. In a
bounded probe the `fps`-filter sample at 584.16 s
(`ac2190e5b496f6e0`) differs from the exact input-seek frame at 584.16 s
(`df49e38109839d8d`), proving the `span_start + index/fps` label is not the
identity of the input frame selected by FFmpeg near a cut. No consecutive
byte-identical frames appeared in fresh sampling.

The production run retained no per-sample frame digest/decode identity. The
surviving evidence therefore cannot attribute the repeated boxes to the
detector, a cache, decoder state, scene assignment, or fps-filter frame
selection. Detector policy, sampling policy, and scene assignment were **not**
changed without a reproduced cause; full genuine sampling/framing remediation is
not claimed. This is an explicit open blocker for SOL.

## Remaining blockers

1. **Unresolved sampling provenance** above — needs per-sample frame digest /
   actual decode identity capture and a reproduced cause before claiming full
   sampling correction.
2. **Semantic acceptance** — the Stage 4 promise/Stage 6 intent contracts are
   implemented, but offline fake/recorded responses cannot prove a live model now
   judges correctly. The recorded semantically invalid plan should be rejected
   before rendering; genuine semantic acceptance still needs real review.
3. **Silent card policy** — a dominant silent card is allowed only with an
   assessed `LENGTH_NECESSITY`; this plan's recorded 52-character copy and
   no-free-zone overlay remain unsuitable for a positive card.
4. **Genuine acceptance** — none is authorized. Only after SOL closes these
   blockers may one separately authorized attempt be proposed; human acceptance
   is mandatory.
