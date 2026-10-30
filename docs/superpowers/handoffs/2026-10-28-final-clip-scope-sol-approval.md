# FINAL_CLIP scope correction — Architect approval

Date: 2026-10-28
Disposition: **APPROVED WITH KNOWN LIMITATIONS — offline code batch only**

Brief: [R2 correction](2026-10-28-final-clip-scope-v4-fix-r2.md).
Implementer: [R2 result](2026-10-28-final-clip-scope-v4-fix-r2-result.md).
History: [original review](2026-10-28-final-clip-scope-sol-review.md),
[R1 re-review](2026-10-28-final-clip-scope-sol-review-r1.md).

## Scope and repository identity

Live branch `p5`; original BASE and current HEAD both
`d326f426a7bf6d19c0fe47442172add706b755ee`. No new commits. The approved code
remains in the shared modified/untracked worktree; unrelated work was preserved.

Compared current hashes with the archived R1 review: exactly the four declared
Python files changed among the ten reviewed files: refinement service/executor,
transformation inputs, and the refinement tests. Re-reviewed those changes and
their tests. Previously closed S1, pure scope derivation, F1 projection and other
unchanged components were not reopened for optional improvements.

Only `p5` was checked. Functional components exercised in isolated tests are not
a second checked-out `main` branch, production deployment, or real service test.

## Independent findings closed

**S2/B1 residual: PASS.** `_effective_final_view` keeps the complete raw sequence
and projects it before any filtering on the scoped/manual path. The archived
68-record probe now produces Stage 4 refusal and Stage 5 unsupported projection
(`manual_transcript_invalid_word_evidence`), with manual text/raw records intact.
The existing valid selected/manual consumer chain continues to pass.

**S3 residual: PASS.** The production executor now carries `asdict` of actual
`targeted_transcription_options` for both priorities. Rejected-observation matching
uses every field from the `TranscriptionOptions` schema, including compute and
fallback precision, temperature and source-evidenced hotword overlay. Missing
identity refuses reuse; tuple/list JSON round-trips compare consistently.

The actual production-factory probe was repeated independently:

| Requested change | Production identity changes | Old rejected observations reused |
| --- | --- | --- |
| `compute_type`: `int8` -> `float32` | yes | no |
| `temperature`: `(0.0,)` -> `(0.9,)` | yes | no |

The same preserved raw observations still derive a valid selected view. Thus the
cache refusal is due to the changed decoder request, not malformed test evidence.
Real factory regressions also prove unchanged-configuration reuse and CPU fallback
precision mismatch through a fake backbone. The prior model/beam/device and
CANDIDATE retry fixes remain passing.

**S1: stays closed.** The fresh selection includes non-waivable refusal across
manual resolution and normal rerun. The archived boundary probe also confirms
optimistic-ready refused rows remain unusable by the final resolver, input
assembly and Stage 4.1 handoff. No new change to S1 was needed.

## Fresh verification

Network-disabled cached Python 3.12 image, backend mounted read-only, isolated
SQLite and temporary storage; fake ASR boundaries only:

```bash
docker run --rm --pull never --network none \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app:/app/tests \
  -v "$PWD/backend:/app:ro" \
  -v /tmp/clipfactory-final-scope-r2-sol-review:/audit \
  -w /app --entrypoint python oac-stage6-test:latest -m pytest \
  tests/test_stage35_refinement.py \
  -k 'scope_refusal or supported_manual_correction or rejected_observation or rejected_candidate_retry or raw_sequence_with_nonrecord or production_' \
  -q -p no:cacheprovider --basetemp=/audit/pytest --junitxml=/audit/junit.xml
```

Parsed fresh JUnit: **14 tests, 0 failures, 0 errors, 0 skips**.
V4's existing affected-selection JUnit was inspected and reports **122/0/0/0**;
SOL did not duplicate that five-minute selection. Scoped Ruff check and format
check passed on all four changed Python files. `git diff --check` passed;
`git diff --exit-code -- backend/app/render/execution/` was empty.

[Archived probes, results, JUnit and reviewed-file hashes](2026-10-28-final-clip-scope-r2-sol-evidence/).

No production code edits, actual model/ASR/provider calls, operational
PostgreSQL/Redis/storage access, real-row changes, recovery/import, migration,
commit or push occurred during this review. Compose services remain stopped.

## Known limitations and next gates

- **Operator:** selected spoken wording and captions are not human-verified.
  Use the [existing selected preview and provisional transcript packet](2026-10-28-selected-window-local-recovery-v4-result.md).
  A technical 67-word projection is not an accepted transcript or good Short.
- **Architect/Implementer:** no real refinement was recovered by this batch.
  The legacy operational row lacks complete stored observations in the new
  production shape. The separately archived decoder response is an observation,
  not an approved production import. A supported identity-validated recovery path
  must be defined and authorized before any real-row use; do not launch another
  ASR pass or assume a normal retry will be provider-free.
- **Architect/Operator:** if verified wording needs edits outside F1's supported
  one-token projection, that exact alignment requirement remains a separate
  decision. General multi-token alignment was not introduced.
- **QA:** evidence is SQLite/offline; no PostgreSQL or live provider acceptance
  is claimed. Broader hardening remains deferred to Stage 13.
- **Architect/Operator:** substantive transformation suitability, actual Stage
  4–6 request and finite hosted budget still require their own preflight and
  explicit spending approval. Originality/publication/media acceptance is not
  granted by this code review.

**S1–S3 and this correction cycle are closed. No further correction batch is
requested.** Retain the reviewed scope without optional re-audit/refactoring.
Stage 6 remains unfrozen and unaccepted; Stage 7 remains gated. No new operational
recovery or model/provider authorization is implied by this approval.
