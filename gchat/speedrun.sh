#!/usr/bin/env bash

# Train gchat on a TPU VM using tokenized ArrayRecord shards in GCS.
#
# Example:
#   bash gchat/speedrun.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export GCHAT_BASE_DIR="${GCHAT_BASE_DIR:-${HOME}/.cache/gchat}"
export GCHAT_HOST_CACHE="${GCHAT_HOST_CACHE:-${GCHAT_BASE_DIR}/host_cache}"
mkdir -p "${GCHAT_BASE_DIR}" "${GCHAT_HOST_CACHE}"

# Edit this block to point the run at a different dataset or metrics collector.
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT="${OTEL_EXPORTER_OTLP_METRICS_ENDPOINT:-https://otel.gerardosalazar.com/v1/metrics}"
export OTEL_COLLECTOR_USERNAME="${OTEL_COLLECTOR_USERNAME:-otel}"
export OTEL_COLLECTOR_PASSWORD="${OTEL_COLLECTOR_PASSWORD:-<your collector password>}"
export GCHAT_DATA_DIR="${GCHAT_DATA_DIR:-gs://gchat-climbmix-7b/data}"
export GCHAT_TOKEN_BYTES_PATH="${GCHAT_TOKEN_BYTES_PATH:-${GCHAT_DATA_DIR%/}/token_bytes.npy}"
export GCHAT_METRICS_RUN_ID="${GCHAT_METRICS_RUN_ID:-gchat-test}"

# Training defaults chosen to mirror nanochat/runs/speedrun.sh as closely as
# this single-device JAX trainer currently supports.
GCHAT_SEQUENCE_LENGTH="${GCHAT_SEQUENCE_LENGTH:-1024}"
GCHAT_N_LAYER="${GCHAT_N_LAYER:-12}"
GCHAT_BATCH_SIZE="${GCHAT_BATCH_SIZE:-16}"
GCHAT_LEARNING_RATE="${GCHAT_LEARNING_RATE:-3e-4}"
GCHAT_SEED="${GCHAT_SEED:-0}"
GCHAT_LOG_EVERY="${GCHAT_LOG_EVERY:-1}"
GCHAT_EVAL_EVERY="${GCHAT_EVAL_EVERY:-250}"
GCHAT_EVAL_AT_END="${GCHAT_EVAL_AT_END:-0}"
GCHAT_EVAL_BATCH_SIZE="${GCHAT_EVAL_BATCH_SIZE:-16}"
GCHAT_EVAL_SPLIT_TOKENS="${GCHAT_EVAL_SPLIT_TOKENS:-41943040}"
GCHAT_TOKEN_SHARD_COUNT="${GCHAT_TOKEN_SHARD_COUNT:-54}"

# By default, run one pass over the available token shards so the final
# checkpoint is saved. Set GCHAT_NO_REPEAT=0 to train indefinitely.
GCHAT_NO_REPEAT="${GCHAT_NO_REPEAT:-1}"
GCHAT_NO_SHUFFLE="${GCHAT_NO_SHUFFLE:-0}"

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

TRAIN_ARGS=(
  --data-dir "${GCHAT_DATA_DIR}"
  --token-bytes-path "${GCHAT_TOKEN_BYTES_PATH}"
  --host-cache-dir "${GCHAT_HOST_CACHE}"
  --sequence-length "${GCHAT_SEQUENCE_LENGTH}"
  --n-layer "${GCHAT_N_LAYER}"
  --batch-size "${GCHAT_BATCH_SIZE}"
  --learning-rate "${GCHAT_LEARNING_RATE}"
  --seed "${GCHAT_SEED}"
  --log-every "${GCHAT_LOG_EVERY}"
  --eval-every "${GCHAT_EVAL_EVERY}"
  --eval-batch-size "${GCHAT_EVAL_BATCH_SIZE}"
  --eval-split-tokens "${GCHAT_EVAL_SPLIT_TOKENS}"
)

if [[ "${GCHAT_EVAL_AT_END}" != "0" ]]; then
  TRAIN_ARGS+=(--eval-at-end)
fi

TRAIN_ARGS+=(
  --gcs-token-shard-count "${GCHAT_TOKEN_SHARD_COUNT}"
)

if [[ "${GCHAT_NO_REPEAT}" != "0" ]]; then
  TRAIN_ARGS+=(--no-repeat)
fi

if [[ "${GCHAT_NO_SHUFFLE}" != "0" ]]; then
  TRAIN_ARGS+=(--no-shuffle)
fi

GCHAT_DATA_URI="${GCHAT_DATA_DIR%/}"
if [[ "${GCHAT_DATA_URI}" == gs://* ]]; then
  GCHAT_DATA_PATH="${GCHAT_DATA_URI#gs://}"
  GCHAT_CHECKPOINT_BUCKET="${GCHAT_DATA_PATH%%/*}"
  if [[ "${GCHAT_DATA_PATH}" == */* ]]; then
    GCHAT_CHECKPOINT_PREFIX="${GCHAT_DATA_PATH#*/}/checkpoint"
  else
    GCHAT_CHECKPOINT_PREFIX="checkpoint"
  fi
  TRAIN_ARGS+=(
    --gcs-checkpoint-bucket "${GCHAT_CHECKPOINT_BUCKET}"
    --gcs-checkpoint-prefix "${GCHAT_CHECKPOINT_PREFIX}"
  )
else
  TRAIN_ARGS+=(--checkpoint-dir "${GCHAT_CHECKPOINT_DIR:-${GCHAT_BASE_DIR}/checkpoints}")
fi

echo "Starting gchat training"
echo "  run id:      ${GCHAT_METRICS_RUN_ID}"
echo "  data:        ${GCHAT_DATA_DIR}"
echo "  token bytes: ${GCHAT_TOKEN_BYTES_PATH}"
echo "  layers:      ${GCHAT_N_LAYER} transformer blocks"
echo "  shards:      ${GCHAT_TOKEN_SHARD_COUNT} training token shards"
echo "  cache:       ${GCHAT_HOST_CACHE}"
if [[ "${GCHAT_DATA_URI}" == gs://* ]]; then
  echo "  checkpoint:  gs://${GCHAT_CHECKPOINT_BUCKET}/${GCHAT_CHECKPOINT_PREFIX}"
fi

"${UV_BIN}" run --package gchat python -m gchat.training.train "${TRAIN_ARGS[@]}"
