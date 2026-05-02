#!/usr/bin/env bash

# Train gchat on a TPU VM using tokenized ArrayRecord shards in GCS.
#
# Example:
#   GCHAT_DATA_DIR=gs://my-bucket/gchat/climbmix_tokens \
#   GCHAT_CHECKPOINT_BUCKET=my-bucket \
#   bash gchat/speedrun.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export GCHAT_BASE_DIR="${GCHAT_BASE_DIR:-${HOME}/.cache/gchat}"
export GCHAT_HOST_CACHE="${GCHAT_HOST_CACHE:-${GCHAT_BASE_DIR}/host_cache}"
mkdir -p "${GCHAT_BASE_DIR}" "${GCHAT_HOST_CACHE}"

if [[ -z "${GCHAT_DATA_DIR:-}" ]]; then
  cat >&2 <<'EOF'
Set GCHAT_DATA_DIR to the GCS prefix containing:
  tokens-*.arrayrecord
  test-tokens*.arrayrecord
  token_bytes.npy

Example:
  GCHAT_DATA_DIR=gs://my-bucket/gchat/climbmix_tokens bash gchat/speedrun.sh
EOF
  exit 2
fi

export GCHAT_TOKEN_BYTES_PATH="${GCHAT_TOKEN_BYTES_PATH:-${GCHAT_DATA_DIR%/}/token_bytes.npy}"

# Training defaults chosen to mirror nanochat/runs/speedrun.sh as closely as
# this single-device JAX trainer currently supports.
GCHAT_SEQUENCE_LENGTH="${GCHAT_SEQUENCE_LENGTH:-2048}"
GCHAT_BATCH_SIZE="${GCHAT_BATCH_SIZE:-16}"
GCHAT_LEARNING_RATE="${GCHAT_LEARNING_RATE:-3e-4}"
GCHAT_SEED="${GCHAT_SEED:-0}"
GCHAT_LOG_EVERY="${GCHAT_LOG_EVERY:-1}"
GCHAT_EVAL_EVERY="${GCHAT_EVAL_EVERY:-250}"
GCHAT_EVAL_BATCH_SIZE="${GCHAT_EVAL_BATCH_SIZE:-16}"
GCHAT_EVAL_SPLIT_TOKENS="${GCHAT_EVAL_SPLIT_TOKENS:-41943040}"

# By default, run one pass over the available token shards so the final
# checkpoint is saved. Set GCHAT_NO_REPEAT=0 to train indefinitely.
GCHAT_NO_REPEAT="${GCHAT_NO_REPEAT:-1}"
GCHAT_NO_SHUFFLE="${GCHAT_NO_SHUFFLE:-0}"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Use the Gaia workspace root so local workspace packages such as gaia-metrics
# resolve correctly.
uv sync --package gchat --extra tpu

TRAIN_ARGS=(
  --data-dir "${GCHAT_DATA_DIR}"
  --token-bytes-path "${GCHAT_TOKEN_BYTES_PATH}"
  --host-cache-dir "${GCHAT_HOST_CACHE}"
  --sequence-length "${GCHAT_SEQUENCE_LENGTH}"
  --batch-size "${GCHAT_BATCH_SIZE}"
  --learning-rate "${GCHAT_LEARNING_RATE}"
  --seed "${GCHAT_SEED}"
  --log-every "${GCHAT_LOG_EVERY}"
  --eval-every "${GCHAT_EVAL_EVERY}"
  --eval-batch-size "${GCHAT_EVAL_BATCH_SIZE}"
  --eval-split-tokens "${GCHAT_EVAL_SPLIT_TOKENS}"
)

if [[ "${GCHAT_NO_REPEAT}" != "0" ]]; then
  TRAIN_ARGS+=(--no-repeat)
fi

if [[ "${GCHAT_NO_SHUFFLE}" != "0" ]]; then
  TRAIN_ARGS+=(--no-shuffle)
fi

if [[ -n "${GCHAT_CHECKPOINT_BUCKET:-}" ]]; then
  TRAIN_ARGS+=(
    --gcs-checkpoint-bucket "${GCHAT_CHECKPOINT_BUCKET}"
    --gcs-checkpoint-prefix "${GCHAT_CHECKPOINT_PREFIX:-gchat/base_checkpoints/d24}"
  )
else
  TRAIN_ARGS+=(--checkpoint-dir "${GCHAT_CHECKPOINT_DIR:-${GCHAT_BASE_DIR}/checkpoints}")
fi

echo "Starting gchat training"
echo "  data:        ${GCHAT_DATA_DIR}"
echo "  token bytes: ${GCHAT_TOKEN_BYTES_PATH}"
echo "  cache:       ${GCHAT_HOST_CACHE}"

uv run --package gchat python -m gchat.training.train "${TRAIN_ARGS[@]}"
