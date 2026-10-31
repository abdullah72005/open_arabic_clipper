# Local observation recovery — one bounded implementation batch

Status: READY FOR IMPLEMENTER LAUNCH — implementation/tests only
Baseline: `p5`, HEAD `d326f426a7bf6d19c0fe47442172add706b755ee` plus the shared
uncommitted worktree. Preserve all existing/unrelated work. No commit or push.

Use Abdullah Implementer. Follow the
[approved scope correction](2026-10-28-final-clip-scope-sol-approval.md) and
[actual captured run](2026-10-28-selected-window-local-recovery-v4-result.md).
The scope correction cycle is closed; do not reopen it for optional improvements.

## Outcome

Add the smallest operator-only service and thin CLI needed to admit the existing,
identity-verified local decoder observation as a raw Stage 3.5 checkpoint. A normal
subsequent refinement job can then derive the approved clip view without a new
model call. Do not directly set final text, ready status or executable clip words
as an import shortcut. This is one candidate's recovery capability, not a new
job platform, capture framework or general data migration.

The existing reconstruction benchmark capture schema is a different contract.
Reuse suitable existing validation/hash/storage patterns, but do not relabel the
operational observation as a historical capture with invented metadata.

## Operator wording approval

On 2026-10-29, after receiving the selected preview and provisional transcript
packet, the operator said: **`the captions are good, now what`**. Record that as
approval of the presented selected wording, not acceptance of a transformed video.

The presented selected text is the whitespace-joined `selected_words` from the
archived `result.json` below. Its UTF-8 SHA-256 is
`6cc761f9da32f3469c0cd0eca1554fd3a59fd47bf6afc7e088ce5332e74c7f9e`.
It has 67 tokens and matches the derived clip view exactly. SOL checked the
actual raw response through the current pure scope helper and F1 projection:
`UNCHANGED`, with context-only invalid record `[105]` preserved.

No manual endpoint has been submitted and no real-row approval flag changed.
Keep manual submission separate from raw checkpoint admission; the existing
endpoint can later receive this exact approved text after supported recovery.
Do not invent new wording, normalize away differences, or expand alignment.

## Pinned target and evidence

- Source `cafc529e-b0d8-401b-b439-2721a57de73c`.
- Candidate `b3fcfc05-638a-49f6-89f2-cdab89291871` (OPERATOR).
- FINAL refinement `b2ee68f5-1ec1-40c8-a9b0-3fb5b1d7c345`.
- Captured successful job `fc9ded35-74b0-46ab-b3d0-bee91b38e37c`.
- Coarse `384.280–408.660`; context `376.280–416.660`;
  actual refined `384.300–408.680` (retain stored exact float values).
- Candidate key
  `9ebcd6855ab02d1017b5f7e7e45ae1948a2d4623363339c5bce4fcaf3f96fd3e`.
- Raw-span fingerprint
  `60c630dc55c1e7d41b6024c45effb2d245e79e7182ccd7a4fd9cc66be929e4b7`.
- Source SHA-256
  `77043cae0ceb6d15c27a73ea16842503e3e8e56044bc28a284772305ac63f706`.
- Exact extracted context WAV SHA-256
  `7ea64c4ac920a1ce8e308d02bdad7c32a119361617fdc8c537cb8a2ce1bce96e`.
- Storage-owned observation directory:
  `temporary/diagnostics/local-recovery-b3fcfc05-20261028T044442Z-5edff6f75ef1/`.

| File | Pinned SHA-256 |
| --- | --- |
| `decoder-result.json` | `d3c4a3090b1f820b96b0bef19f1994b1123f9912270088f14b4f749c39b82fbc` |
| `result.json` | `db4e7193e70357971c29510f5b9d11962073b1d88f57461df01c89ef8aeaf466` |
| `prior-evidence.json` | `c78633884b7a0bb96683178395c269af818c0a83be5cbfd4344f32654f15994e` |

This is an explicitly operator-trusted, hash-pinned legacy observation bundle.
Do not accept an arbitrary new JSON object just because it names the candidate.
Preserve missing historical metadata as unknown; never synthesize a past model
run, library version, approval or job. Current source/audio/options validation
must be independent of assertions made by the bundle itself.

## Service and CLI contract

1. Add one small recovery module under `app/refinement/` and an operator CLI
   entry point in the existing CLI. Default to read-only validation/preview;
   provide explicit apply mode for later authorized execution. No HTTP route,
   schema change, automatic source-wide processing, or worker dispatch.
2. Validate all pinned file hashes; candidate/source/refinement/job identities;
   exact coarse/context bounds; operator marker/current raw-span identity;
   source bytes and context WAV bytes; complete raw response; and the current
   effective decoder/context contract including source-evidenced hotwords.
   Use StorageService-owned paths. No arbitrary-path import or guessed identity.
   Reject mismatches with a precise reason before mutation.
3. Reconstruct the raw local result through the same production conversion and
   fingerprint logic used after `WhisperEngine` returns. If needed, extract that
   conversion as one pure helper reused by both callers; do not duplicate the
   transformation, load a model, or introduce a fake provider. Preserve all 107
   original records, their text/times/probabilities and record 105's zero duration.
4. Preview reports the exact proposed checkpoint delta, provenance and derived
   clip validation without database/file writes, job creation, model loading or
   provider construction. It must not claim the real row is ready.
5. Apply must revalidate under existing source/refinement ownership/locking
   conventions, refuse active/conflicting jobs or changed evidence, and archive
   the current selected checkpoint before replacing it. Mutations belong to the
   caller's transaction; no partial admission on failure. A snapshot failure
   prevents mutation. Use StorageService for the uniquely named audit snapshot.
6. Admit only the verified raw local checkpoint/evidence and dependency
   fingerprints needed for existing rejected-observation reuse. Keep its context
   state **REJECTED**, with the real nonpositive-duration reason and recovery
   provenance (bundle hashes, source/audio/decoder identity and original job).
   Keep automatic/raw source evidence, manual authority, existing jobs and source
   lifecycle intact. Do not copy a hand-built 67-word result into final fields,
   mark FINAL_TRANSCRIPT_READY/cache eligible, or change publishing approvals.
7. Make identical admission idempotent. A changed bundle or conflicting current
   checkpoint/manual/evidence identity must refuse, not overwrite blindly.
8. Apply does **not** dispatch or run refinement. Supply a runbook for a later
   normal, exact-candidate job with a hard zero-model-call guard: if normal reuse
   misses, stop before Whisper/provider work. Do not assume a normal retry is
   automatically free. Preserve normal job claim/cancellation/resource handling;
   no broad worker/backlog consumer or manual-ready patch.

The later operational sequence is checkpoint admission -> one normal guarded
refinement job -> supported manual endpoint with the exact approved text ->
readiness/currentness checks. That sequence is **not authorized for execution by
this implementation-only brief**. Code must be independently reviewed first.

## Focused acceptance and verification

- Positive pinned fixture: preview has zero mutations; applying in isolated
  SQLite admits a rejected complete raw checkpoint with traceable provenance.
- Normal production executor over that isolated admitted checkpoint uses zero
  actual/fake-ASR calls (a fail-on-call spy proves this), derives the accepted
  67-word view at actual refined bounds, and keeps rejected context truthful.
- Existing manual endpoint with the exact operator-approved 67-word text then
  satisfies F1 unchanged projection in the isolated fixture. Ready status must
  come from existing rules; do not force it. Verify Stage 4 effective input and
  caption payload agreement with no context-word leakage.
- Tampered file/source/audio hash, wrong candidate/refinement/window, changed
  decoder settings, incomplete raw data, selected invalid timing, stale source
  or active job all refuse before writes. Do not bypass the approved scope
  validator because the bundle is trusted.
- Repeated identical apply is idempotent; failure/rollback preserves old evidence,
  manual text, jobs and source state. No replay can manufacture consensus or count
  as a new model run. Report recovery separately from actual ASR usage.

Use network-disabled cached Python 3.12 containers, isolated SQLite/storage and
the recorded observation fixture only. Run the focused new recovery tests and
affected Stage 3.5 scope/manual identity checks, scoped Ruff/format,
`git diff --check` and frozen renderer diff. No repeated full audit/suite.

## Permissions, parallelism and delivery

Zero actual ASR/Gemini/hosted/Qwen/TTS calls, downloads, operational DB/Redis access,
real-row writes/imports, Stage 4–6 execution, migrations, commits or pushes in this
batch. Read the pinned observation bundle only to prepare test fixtures; leave
the originals unchanged. Frozen Stage 5.2 and closed scope/F1 algorithms stay
unchanged. No Stage 6 freeze or Stage 7 work.

One owner for validation/admission. Independent negative-fixture/runbook work may
run in parallel with explicit ownership; no overlapping shared-file edits. Stop
only for a concrete consequential conflict with this contract; do not design a
new recovery platform or silently broaden trusted inputs.

Save `docs/superpowers/handoffs/2026-10-29-local-observation-recovery-v4-result.md`
and return a copyable report: actual base/HEAD/state, implementation locations,
exact focused verification, proposed dry-run/apply commands, admitted field delta,
zero-model guarded continuation, limitations and unresolved blockers.

Stop for Architect independent review. The next deliverable is a safe recovery
capability and concrete runbook, not another diagnostic video or an assertion of
Stage 6 acceptance. Hosted production spending remains separately gated.
