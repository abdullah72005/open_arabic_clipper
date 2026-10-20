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
