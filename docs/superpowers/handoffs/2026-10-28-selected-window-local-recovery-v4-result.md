# Selected-window additional local pass — result

Date: 2026-10-28
Live HEAD: `d326f426a7bf6d19c0fe47442172add706b755ee`, with preserved existing changes.
Task: [approved one-pass recovery scope](2026-10-28-selected-window-local-recovery-v4.md).
Prerequisite: [independently closed R3 correction](2026-10-28-local-evidence-checkpoint-sol-verification.md).

## Outcome

**One additional local model invocation completed. Transcript recovery did not
succeed.** The job succeeded, but the refinement remains
`NEEDS_MANUAL_TRANSCRIPT_REVIEW`, `cache_eligible=false`. No Stage 4–6 work follows.

The new diagnostic and preserved raw decoder response identify the exact original
failure shape: ASR word record **105** (zero-based), `الجو؟`, has relative
`start=end=40.02`, source `start=end≈416.300` seconds. Its probability is
`0.30443238218625385`. It is inside the context `376.28–416.66`, after the
selected clip `384.28–408.66`. It is **zero duration**, not a context escape.

The strict adapter refuses that record and rejects the local evidence with
`nonpositive word duration`. This reproduces the old failure with an accurate
diagnostic. No epsilon change, clamping, timing invention or token removal to
force readiness occurred. The 67 selected words remain present, including the
later denial; their presence does not satisfy the existing whole-refinement gate.

## Actual execution and identities

| Field | Measured value |
| --- | --- |
| Source | `cafc529e-b0d8-401b-b439-2721a57de73c` |
| OPERATOR candidate | `b3fcfc05-638a-49f6-89f2-cdab89291871` |
| FINAL_CLIP refinement | `b2ee68f5-1ec1-40c8-a9b0-3fb5b1d7c345` |
| New job | `fc9ded35-74b0-46ab-b3d0-bee91b38e37c` |
| Job status | `SUCCEEDED` |
| Started / completed UTC | `2026-10-28 04:44:43.427862+00:00` / `2026-10-28 04:46:01.206207+00:00` |
| Coarse / context | `384.28–408.66` / `376.28–416.66` |
| Refined | `384.300–408.680` |
| Raw-span fingerprint | `60c630dc55c1e7d41b6024c45effb2d245e79e7182ccd7a4fd9cc66be929e4b7` |
| Actual engine invocations in this task | `1` |
| Persisted local metrics | `{"local_asr_runs": 1}` |
| Model / hardware | Cached `large-v3-turbo`, CPU/int8 |
| FINAL_CLIP decoder | beam 8, automatic language, word timestamps, no VAD, no previous-text conditioning |
| Temperature sequence | Existing `[0, .2, .4, .6, .8, 1]` |
| Child elapsed / peak RSS | `76.099 s` / `2142400512` bytes |
| Raw decoder / retained / selected words | `107 / 106 / 67` |
| Final whitespace tokens | `107` |
| Local evidence | `REJECTED` / `nonpositive word duration` |
| Display projection | `manual_transcript_token_count_mismatch` |

The normal decoder settings were preserved. Targeted options have `cpu_threads=0`
under the existing production implementation; the container had a six-CPU limit.
The global thread setting was not substituted into targeted options. The cached
model snapshot is `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`; all required files
were present. Model weights were mounted read-only, with downloads disabled.

## Execution path and controls

This was the operator-approved additional pass, executed by the coordinating
agent through the existing `queue_candidate_refinement(... FINAL_CLIP,
force=False)` and `clipfactory.run_candidate_refinement` Celery eager task.
No backlog worker or new application execution path was introduced. Fresh
sessions and a held source `FOR NO KEY UPDATE NOWAIT` guard were used.

Before queueing, the actual source file SHA-256, cached refinement WAV hash,
operator ownership, exact key/bounds/indexes, current immutable raw-span
fingerprint, no competing active source jobs, and R3 checkpoint refusal were
checked. The prior evidence and two earlier job identities were archived first.
The completed result was read from a new session. Source/transcript update
timestamps and the raw-span fingerprint were asserted unchanged.

An observation wrapper around the existing engine counted invocations, refused
a second invocation before model work, and archived the exact returned decoder
response before adapter validation. It did not change that response, the
decoder, provider selection, job ownership, lease, cancellation or readiness
rules. The archive is an operational observation; no new reusable ASR capture
schema or replay authority is claimed.

A parallel read-only agent checked cached model/dependencies and harness APIs
before execution. It did not load a model, access databases or start services.
After execution it independently checked the archived raw record, persisted
diagnostic, counts, retained denial, projection refusal and recorded cleanup;
it found no contradiction. Its cleanup assessment used the recorded evidence,
while the coordinator also checked live Compose/network state after completion.

## Evidence paths and integrity

Archive directory, host:

`/mnt/c/users/abdul/onedrive/documents/vscode/oac/storage/temporary/diagnostics/local-recovery-b3fcfc05-20261028T044442Z-5edff6f75ef1`

Windows:

`C:\Users\abdul\OneDrive\Documents\VScode\oac\storage\temporary\diagnostics\local-recovery-b3fcfc05-20261028T044442Z-5edff6f75ef1`

- [Prior evidence snapshot](../../../storage/temporary/diagnostics/local-recovery-b3fcfc05-20261028T044442Z-5edff6f75ef1/prior-evidence.json) — full previous selected
  refinement, fingerprints, timed words, job identities and audio hash.
- [Raw decoder response](../../../storage/temporary/diagnostics/local-recovery-b3fcfc05-20261028T044442Z-5edff6f75ef1/decoder-result.json) — all 107 original relative
  word records, including the zero-duration record, plus options/audio identity.
- [Actual model-start record](../../../storage/temporary/diagnostics/local-recovery-b3fcfc05-20261028T044442Z-5edff6f75ef1/model-start.json).
- [Fresh persisted result](../../../storage/temporary/diagnostics/local-recovery-b3fcfc05-20261028T044442Z-5edff6f75ef1/result.json) — full resulting transcript,
  retained words, selected words, evidence, diagnostic, job and projection result.
- [Analysis, cleanup state and observed execution harness](2026-10-28-local-recovery-evidence/).

SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| `decoder-result.json` | `d3c4a3090b1f820b96b0bef19f1994b1123f9912270088f14b4f749c39b82fbc` |
| `model-start.json` | `b120ea085b59b85682c65800d3c1565666ba3143880e0af750b93869af6fc2e7` |
| `prior-evidence.json` | `c78633884b7a0bb96683178395c269af818c0a83be5cbfd4344f32654f15994e` |
| `result.json` | `db4e7193e70357971c29510f5b9d11962073b1d88f57461df01c89ef8aeaf466` |

The archived prior snapshot hash still matches its pre-execution hash. The new
raw text exactly matches the prior automatic text, and all 106 retained words
exactly match the prior retained sequence. Thus this pass supplied precise
failure evidence, not improved transcription.

## Concrete review packet — provisional ASR, not human-verified captions

Existing source-only preview (not a Stage 6 artifact):

`C:\Users\abdul\OneDrive\Documents\VScode\oac\storage\temporary\diagnostics\selected-money-timing-384280-408660-20261027-e201734f.mp4`

Existing context WAV, `376.28–416.66` source seconds:

`C:\Users\abdul\OneDrive\Documents\VScode\oac\storage\sources\cafc529e-b0d8-401b-b439-2721a57de73c\candidate-refinements\b3fcfc05-638a-49f6-89f2-cdab89291871\FINAL_CLIP.wav`

Selected ASR text (67 timed words), requiring listening/correction:

> طيب انا اسألك سؤال لو حبيت واحدة واتجوزته وهي طلعت اغنى منك هل في الحالة دي انت هتسيبها؟ مش هحط نفسي في الكيس دي يعني هقولك على حاجة خلينا متفعين ان الموضوع بيبقى عيلة كاملة مش مجرد اتنين اه فالعيلة بقى هتبص نظرة انا مش هحبها فهتسيبها يعني لا مش هاسبه أنت كلامك بيقول لك هاسبه اللي هو أنت هتيجي هتجي بشانتة هدومك مش هتيجي بحاجة حمش

The question, first reply, follow-up, denial and reaction are represented in the
timed evidence. These are ASR observations, not a claim of accurate wording.
The meaning-critical denial remains:

| Token | Start | End | Probability |
| --- | --- | --- | --- |
| لا | 402.420 | 402.720 | 0.7381 |
| مش | 402.720 | 402.960 | 0.3225 |
| هاسبه | 402.960 | 403.360 | 0.5933 |

`مش` is still low-confidence; the raw INDEX and local evidence both retain
negation. No human listening was performed during this execution. Do not treat
the ASR spelling `هاسبه`, other obvious-looking transcription errors, or the
post-context `الجو؟` token as operator-confirmed speech. No manual text or
resolution was submitted. F1 still refuses the 107-token/106-word sequence.

## Cleanup and stop

- Exactly one actual new local engine call; zero hosted transcription,
  adjudication, Gemini, Qwen, TTS or other provider calls.
- The one-off ran on a verified internal Docker network; no external gateway,
  no hosted credentials, and offline model flags. The network and container
  were removed.
- Redis `/1` reported no unsafe marker or active heavy-model lease after work.
  Nothing was cleared by hand.
- PostgreSQL and Redis were restored to their prior stopped state. No running
  Compose services remain. The one-off and cleanup exited 0.
- Source/raw transcript/manual authority were preserved. Only normal selected
  refinement/job updates and diagnostic archives were made. No application
  code, migrations, approvals, publication states, Stage 4–6 rows, frozen
  renderer, commit or push changes were made by this operation.

**STOP.** The approved additional pass is consumed. No third pass, model change,
hosted fallback or automatic correction batch is authorized. Stage 4 remains
blocked by the unusable FINAL_CLIP status under the existing prerequisite rules;
no candidate-grade refinement was added. Stage 6 remains unfrozen and Stage 7
remains gated.

The next architectural decision is how to handle invalid timing in context
outside the selected clip without discarding evidence or weakening selected
caption validation. No change to that contract or proposed remedy is implemented
or approved by this result. Another identical ASR retry has no demonstrated
benefit after the identical reproduction. Human review alone cannot supply the
missing timed word through the current manual projection endpoint.
