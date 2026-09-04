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

## Development implications

The Docker services use Python 3.12 and install FFmpeg, so they are the
supported path on this machine. Native execution is still supported after the
operator installs Python 3.12+ and FFmpeg/ffprobe. CPU-only operation is the
default; later GPU acceleration is an optional enhancement.
