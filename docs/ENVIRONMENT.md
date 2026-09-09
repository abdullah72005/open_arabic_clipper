# Development Environment Audit

Audited: 2026-09-04. These facts describe the current development machine only;
the application remains portable and config-driven.

| Check | Finding |
| --- | --- |
| Environment | WSL2: Linux `6.18.33.2-microsoft-standard-WSL2`, workspace mounted from Windows (`/mnt/c`) |
| CPU | Intel Core Ultra 9 185H, 22 logical CPUs |
| RAM | 7.4 GiB total; 4.7 GiB available at audit time |
| GPU / VRAM / CUDA | No `nvidia-smi` or `nvcc` present; no GPU capability assumed |
| Docker | Docker Desktop 28.3.3 available |
| Docker Compose | v2.39.2-desktop.1 available |
| Python | Host Python 3.10.12 (below project runtime requirement) |
| Node / npm / pnpm | Node v24.13.1; npm 11.17.0; pnpm unavailable |
| FFmpeg / ffprobe | Not installed on host PATH |
| Git | 2.34.1 |
| Free disk | 259 GiB free on the mounted Windows volume (952 GiB total) |
| Network | HTTPS checks to PyPI and npm registry succeeded |
| Ollama | `ollama/ollama` runs under the `reconstruction` profile; `qwen3:8b` pulled with digest `500a1f067a9f…b41` |

## Hosted Gemini cloud configuration (2026-09-09)

Stage 2.7 supports an optional hosted Gemini reconstruction provider through the
official Google Gen AI SDK. It is configured by environment variables only; no
frontend settings system exists. The Gemini API key is read by application
settings from `GEMINI_API_KEY` or `CLIPFACTORY_GEMINI_API_KEY` in the local
`.env`, held as a Pydantic `SecretStr`, and unwrapped only when constructing the
SDK client. It is masked in settings repr/serialization/validation errors and is
never printed, logged, fingerprinted, returned by any API, or committed.

**Cloud snippet disclosure.** The `adaptive` and `gemini_only` routing modes may
send short transcript snippets and bounded context to Google Gemini. The
default mode is `adaptive`; set `CLIPFACTORY_RECONSTRUCTION_ROUTING_MODE` to
`local_only` for a fully local pipeline. Cloud-processing configuration is
separate from rights/provenance eligibility. Free-tier quotas are shared per
Google project, so all Gemini consumers draw from the same project allowance.

Model, timeout, retry, and budget settings mirror the local provider pattern:

```bash
CLIPFACTORY_RECONSTRUCTION_ROUTING_MODE=adaptive
CLIPFACTORY_GEMINI_MODEL=gemini-3.8-flash
CLIPFACTORY_GEMINI_THINKING_LEVEL=low
CLIPFACTORY_GEMINI_TEMPERATURE=0
CLIPFACTORY_GEMINI_API_VERSION=v1
CLIPFACTORY_GEMINI_TIMEOUT_SECONDS=30
CLIPFACTORY_GEMINI_RETRY_ATTEMPTS=1
CLIPFACTORY_GEMINI_RETRY_BACKOFF_SECONDS=1.5
CLIPFACTORY_GEMINI_MAX_TARGETS_PER_JOB=5
CLIPFACTORY_GEMINI_MAX_OUTPUT_TOKENS=1024
```

The SDK client is constructed lazily only when a generation request runs, and
availability is configuration-level: there are no per-job `models.get` metadata
probes, so a cache hit, an all-`NO_LLM` job, and `LOCAL_ONLY` work make zero
Gemini network calls. Reconstruction generation is deterministic
(`temperature=0`, explicit API version `v1`).

## Bounded local reconstruction

Local Qwen work is selected, ranked, batched, and hard-bounded so a multi-hour
source can never produce unbounded inference:

```bash
CLIPFACTORY_LOCAL_RECONSTRUCTION_MAX_TARGETS_PER_JOB=64
CLIPFACTORY_LOCAL_RECONSTRUCTION_MAX_WALL_SECONDS=1200
CLIPFACTORY_RECONSTRUCTION_PROVIDER_BATCH_WINDOWS=8
CLIPFACTORY_RECONSTRUCTION_PROVIDER_BATCH_CHARACTERS=24000
```

Clean, well-covered unchanged Stage 2.5 segments route to `NO_LLM` and use
neither LLM; local candidates are attempted strongest-first in micro-batches;
when a ceiling is reached the remaining targets are marked unresolved/manual
review and never auto-escalate to Gemini. Accepted per-target work survives
cancellation and restart through checkpointed per-target fingerprints.

## Ollama hardware safeguards (Compose)

The Ollama service pins operator-tunable safety limits sized for the documented
~10.7 GiB WSL host:

```bash
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_MAX_QUEUE=4
OLLAMA_CONTEXT_LENGTH=4096
OLLAMA_CPUS=6
OLLAMA_MEM_LIMIT=6g
OLLAMA_MEMSWAP_LIMIT=8g
```

These caps primarily protect machine responsiveness; they can increase
individual inference latency. Routing, batching, caching, and the local work
ceilings provide the main wall-time improvement. If the ceiling prevents Qwen
from loading or completing, the job records a truthful provider failure,
preserves Stage 2.5 output, escalates only within the Gemini policy and budget,
and otherwise marks the target unresolved/manual review without retrying
indefinitely.

A missing key never blocks startup or local operation. Gemini is treated as a
scarce resource: the per-job target budget spends on the strongest eligible
targets first, only transient failures get one bounded retry, and a 429/quota
exhaustion stops further calls for that job. The per-job cap is not an
account-wide billing/quota manager. A temporary provider outage does not
invalidate accepted output: fingerprints are stable identity only and cache
eligibility is tracked separately. See `docs/STAGE_2_7_OPERATIONS.md` for the
full routing policy.

## Development implications

The Docker services use Python 3.12 and install FFmpeg, so they are the
supported path on this machine. Native execution is still supported after the
operator installs Python 3.12+ and FFmpeg/ffprobe. CPU-only operation is the
default; later GPU acceleration is an optional enhancement.

The 7.4 GiB RAM limit makes the provisional `qwen3:8b` reconstruction model
infeasible for the live pipeline benchmark: loading it (~5.5 GiB) alongside
faster-whisper and the running services triggers an out-of-memory kill.
`qwen3.5:4b` (~3.4 GiB) loads but did not apply a reconstruction during the
diagnostic because its unbatched request exceeded the model context; see
`docs/BENCHMARKS.md` for measured results.

## Stage 2.7 memory ceiling (measured 2026-09-07)

`scripts/diagnose-memory.sh` is a read-only diagnostic that prints these facts.
The measured values below distinguish host RAM, the WSL VM limit, the container
cgroup limit, process/container usage, and swap.

| Layer | Measurement | Value |
| --- | --- | --- |
| Host physical RAM | `Win32_ComputerSystem.TotalPhysicalMemory` | 16,506,011,648 bytes ≈ 15.37 GiB |
| WSL config | `%UserProfile%\.wslconfig` | absent (no explicit limit) |
| WSL default | `wsl --status` | WSL2, Ubuntu-22.04 |
| Linux total | `/proc/meminfo MemTotal` | 7,803,048 kB ≈ 7.44 GiB |
| Linux available | `/proc/meminfo MemAvailable` | ≈ 4.34 GiB at measurement |
| Swap total / used | `swapon --show --bytes` | 2 GiB total, ≈ 1.3 GiB used |
| cgroup limit | `/sys/fs/cgroup/memory.max` | `max` (no container cgroup limit) |
| cgroup current / peak | `memory.current` / `memory.peak` | ≈ 383 MB / 504 MB |
| Docker host view | `docker info MemTotal` | 7,990,321,152 bytes ≈ 7.44 GiB |
| Worker container limit | `docker inspect .HostConfig.Memory` | 0 (unlimited) |
| Ollama residency | `ollama ps` | no model loaded at measurement |

**Conclusion:** the narrowest active ceiling is the WSL2 virtual-machine
allocation. With no `.wslconfig`, WSL2 defaults to 50% of host RAM (about 8 GB),
which the guest reports as 7.44 GiB of `MemTotal`. Container cgroup limits are
unlimited and are not the cause. This is a WSL/Docker VM capacity limit, not an
application limit.

To raise the ceiling, the operator writes `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=11GB
swap=4GB
```

then runs `wsl --shutdown` from Windows, restarts Docker Desktop, and reruns
`scripts/diagnose-memory.sh`. Eleven GiB leaves roughly five GiB of host
headroom while allowing a practical quantized 8B trial. Twelve GiB is acceptable
only as an operator-selected alternative after measuring that Windows pressure
stays safe. The repository never edits `.wslconfig` or restarts WSL itself.

### After applying the WSL configuration (measured 2026-09-07)

The operator applied the `memory=11GB swap=4GB` configuration. `diagnose-memory`
now reports:

| Layer | Value |
| --- | --- |
| Linux `MemTotal` | 11,212,972 kB ≈ 10.69 GiB |
| Docker `MemTotal` | 11,482,083,328 bytes ≈ 10.69 GiB |
| Linux `MemAvailable` | ≈ 8.2 GiB at rest |
| Swap | 4 GiB total, ≈ 0 used at rest |
| cgroup limit | `max` (unlimited) |

The heavy-model lifecycle was then measured over three sequential
transcription-plus-reconstruction trials. In every trial the Whisper model ran
in a spawned child process that exited and was reaped before the Ollama
reconstruction began (the Redis `clipfactory:heavy-model` lease serializes
them). `ollama ps` was empty before and after each reconstruction, and the
`unload_outcome` metadata confirmed the unload within a second. See
`docs/STAGE_2_7_OPERATIONS.md` for the per-trial memory table.

## Unsafe heavy-model state and operator recovery (2026-09-08)

Unsafe model residency is recorded in a separate persistent Redis marker
(`clipfactory:heavy-model:unsafe`) that carries no TTL. It is written when an
Ollama unload fails or when a heavy-model lease is lost while work is active.
Because it has no expiry, the marker survives worker restart, CLI exit, and the
lease TTL, so no new heavy-model work may start until an operator clears it.

Recovery is explicit and conservative. `recover-heavy-model` only clears the
marker after confirming the configured model is no longer resident (it polls
Ollama `/api/ps` and treats an unreadable listing as resident):

```bash
docker compose exec backend python -m app.cli recover-heavy-model
```

A worker or CLI that finds the marker refuses to acquire the heavy-model lease
(`HeavyModelUnsafe`) until recovery succeeds. Acquisition checks the unsafe
marker and takes the lease inside one atomic Redis script, and recovery clears
the stale lease and the marker inside one atomic script, so no acquisition can
slip into the gap between the two operations and recovery never deletes a
valid newly acquired lease. This behavior is covered by the heavy-model lease
lifecycle tests.
