#!/usr/bin/env bash
# Build the gchat dataset locally, generate token_bytes.npy, and upload it to GCS.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GCHAT_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
GAIA_ROOT=$(cd -- "${GCHAT_ROOT}/.." && pwd)

cd "${GAIA_ROOT}"

export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

# Dataset generation defaults.
export GCHAT_TOKENS="${GCHAT_TOKENS:-7B}"
export GCHAT_SOURCE="${GCHAT_SOURCE:-${GCHAT_ROOT}/climbmix_tokens}"
export GCHAT_TOKENIZER="${GCHAT_TOKENIZER:-gpt2}"
export GCHAT_TOKEN_BYTES="${GCHAT_TOKEN_BYTES:-${GCHAT_SOURCE}/token_bytes.npy}"
export GCHAT_NUM_PRODUCERS="${GCHAT_NUM_PRODUCERS:-4}"
export GCHAT_NUM_CONSUMERS="${GCHAT_NUM_CONSUMERS:-4}"
export GCHAT_SHARD_CACHE="${GCHAT_SHARD_CACHE:-${GCHAT_SOURCE}/shard_cache}"
export GCHAT_ARRAYRECORD_GROUP_SIZE="${GCHAT_ARRAYRECORD_GROUP_SIZE:-1}"

# Upload defaults. GCS bucket names cannot contain slashes, so this stores data at
# gs://gchat/data/.
export GCHAT_BUCKET="${GCHAT_BUCKET:-gchat}"
export GCHAT_DEST_PREFIX="${GCHAT_DEST_PREFIX:-data}"
export GCHAT_LOCATION="${GCHAT_LOCATION:-us-central2}"
export GCHAT_STORAGE_CLASS="${GCHAT_STORAGE_CLASS:-STANDARD}"

if [[ -z "${UV_BIN:-}" ]]; then
  if command -v uv >/dev/null; then
    UV_BIN=$(command -v uv)
  elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
  else
    echo "uv is required. Install it from https://docs.astral.sh/uv/ or set UV_BIN." >&2
    exit 1
  fi
fi

echo "Preparing gchat dataset"
echo "  tokens:                 ${GCHAT_TOKENS}"
echo "  output:                 ${GCHAT_SOURCE}"
echo "  ArrayRecord group size: ${GCHAT_ARRAYRECORD_GROUP_SIZE}"
echo "  token bytes:            ${GCHAT_TOKEN_BYTES}"
echo "  upload:                 gs://${GCHAT_BUCKET}/${GCHAT_DEST_PREFIX%/}/"

"${UV_BIN}" sync --package gchat

"${UV_BIN}" run --package gchat python -m gchat.data.download \
  --tokens "${GCHAT_TOKENS}" \
  --output "${GCHAT_SOURCE}" \
  --num-producers "${GCHAT_NUM_PRODUCERS}" \
  --num-consumers "${GCHAT_NUM_CONSUMERS}" \
  --shard-cache "${GCHAT_SHARD_CACHE}" \
  --arrayrecord-group-size "${GCHAT_ARRAYRECORD_GROUP_SIZE}"

"${UV_BIN}" run --package gchat python -m gchat.data.token_bytes \
  --tokenizer "${GCHAT_TOKENIZER}" \
  --output "${GCHAT_TOKEN_BYTES}"

"${SCRIPT_DIR}/upload_dataset.sh" \
  --bucket "${GCHAT_BUCKET}" \
  --source "${GCHAT_SOURCE}" \
  --dest-prefix "${GCHAT_DEST_PREFIX}" \
  --token-bytes "${GCHAT_TOKEN_BYTES}" \
  --location "${GCHAT_LOCATION}" \
  --storage-class "${GCHAT_STORAGE_CLASS}"
