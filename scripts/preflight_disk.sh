#!/usr/bin/env bash
set -Eeuo pipefail
# ~9.84 GiB compressed base plus unpacked CUDA libraries, layers, venv and
# transient build storage. 40 GiB is a conservative admission floor, not a guarantee.
required=$((40 * 1024 * 1024 * 1024))
available=$(df --output=avail -B1 /var/lib/docker | tail -1 | tr -d ' ')
printf 'Docker filesystem available bytes: %s; required minimum: %s\n' "$available" "$required"
if (( available < required )); then
  echo '::error::Insufficient disk on the free standard runner. Stopping without pulling the base, removing runner software, or selecting paid compute.'
  exit 1
fi
