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
