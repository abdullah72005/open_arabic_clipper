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
