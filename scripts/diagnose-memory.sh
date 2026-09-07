#!/usr/bin/env bash
# Read-only memory diagnostics for the Stage 2.7 heavy-model lifecycle.
#
# Prints labeled facts about Linux, cgroup, Docker, container, and Ollama
# memory. It never modifies state. Optional inputs that are missing or unreadable
# are reported as unavailable instead of failing the whole run.
set -eu

echo "== Linux /proc/meminfo =="
if [ -r /proc/meminfo ]; then
  sed -n '1,25p' /proc/meminfo
else
  echo "unavailable: /proc/meminfo not readable"
fi

echo "== free =="
if command -v free >/dev/null 2>&1; then
  free -h
else
  echo "unavailable: free not installed"
fi

echo "== swap =="
if command -v swapon >/dev/null 2>&1; then
  swapon --show --bytes || echo "unavailable: swapon failed"
else
  echo "unavailable: swapon not installed"
fi

echo "== cgroup v2 =="
for file in memory.max memory.current memory.peak memory.high; do
  path="/sys/fs/cgroup/${file}"
  if [ -r "${path}" ]; then
    printf '%s: %s\n' "${file}" "$(cat "${path}")"
  else
    printf '%s: unavailable\n' "${file}"
  fi
done

echo "== Docker host memory =="
if command -v docker >/dev/null 2>&1; then
  docker info --format 'MemTotal={{.MemTotal}}'
else
  echo "unavailable: docker CLI not installed"
fi

echo "== container memory limit =="
if command -v docker >/dev/null 2>&1; then
  docker inspect "$(hostname)" \
    --format 'HostConfig.Memory={{.HostConfig.Memory}} HostConfig.MemorySwap={{.HostConfig.MemorySwap}}' \
    2>/dev/null || echo "unavailable: cannot inspect this container"
else
  echo "unavailable: docker CLI not installed"
fi

echo "== docker stats (all containers) =="
if command -v docker >/dev/null 2>&1; then
  docker stats --no-stream 2>/dev/null || echo "unavailable: docker stats failed"
else
  echo "unavailable: docker CLI not installed"
fi

echo "== ollama residency =="
if command -v ollama >/dev/null 2>&1; then
  ollama ps 2>/dev/null || echo "unavailable: ollama ps failed"
elif [ -n "${OLLAMA_CONTAINER:-}" ]; then
  docker exec "${OLLAMA_CONTAINER}" ollama ps 2>/dev/null \
    || echo "unavailable: ollama ps failed inside ${OLLAMA_CONTAINER}"
else
  echo "unavailable: ollama binary not installed and OLLAMA_CONTAINER not set"
fi