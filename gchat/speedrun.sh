#!/usr/bin/env bash

# Train gchat on a TPU VM using tokenized ArrayRecord shards in GCS.
#
# Example:
#   bash gchat/speedrun.sh
#   bash gchat/speedrun.sh gchat/conf/my_run.yaml

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

CONFIG_PATH="${1:-${SCRIPT_DIR}/conf/speedrun.yaml}"

UV_BIN="${UV_BIN:-}"
if [[ -z "${UV_BIN}" ]]; then
  if command -v uv >/dev/null; then
    UV_BIN=$(command -v uv)
  elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    if command -v uv >/dev/null; then
      UV_BIN=$(command -v uv)
    elif [[ -x "${HOME}/.local/bin/uv" ]]; then
      UV_BIN="${HOME}/.local/bin/uv"
    else
      echo "uv installation completed, but uv was not found in PATH or ${HOME}/.local/bin" >&2
      exit 1
    fi
  fi
fi

# Use the Gaia workspace root so local workspace packages such as gaia-metrics
# resolve correctly.
"${UV_BIN}" sync --package gchat --extra tpu

"${UV_BIN}" run --package gchat gchat-train "${CONFIG_PATH}"
