#!/usr/bin/env bash
# Create a GCS bucket (if missing) and upload tokenized gchat shards to it.
#
# Designed to run once from the machine that produced ./climbmix_tokens with
# `gchat-download`. Requires: gcloud, and an authenticated user
# (`gcloud auth login` + `gcloud config set project <PROJECT>`).
#
# All configuration is via flags or env vars; nothing is hardcoded.

set -euo pipefail

SCRIPT_NAME=$(basename "$0")

usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} --bucket BUCKET [options]

Required:
  -b, --bucket NAME           GCS bucket name (without gs:// prefix).

Options:
  -p, --project ID            GCP project (default: gcloud config's core/project).
  -l, --location REGION       Bucket location, single region recommended
                              (default: us-central2). Must match your TPU region.
  -c, --storage-class CLASS   Storage class (default: STANDARD).
  -s, --source DIR            Local source directory to upload
                              (default: ./climbmix_tokens).
  -d, --dest-prefix PREFIX    Object prefix inside the bucket
                              (default: gchat/climbmix_tokens).
  -t, --token-bytes PATH      Path to a token_bytes.npy file to upload
                              explicitly (default: <source>/token_bytes.npy).
                              Uploaded to <dest-prefix>/token_bytes.npy.
      --grant-reader PRINCIPAL
                              Optional IAM principal to grant
                              roles/storage.objectViewer on the bucket, e.g.:
                                serviceAccount:foo@bar.iam.gserviceaccount.com
                                user:alice@example.com
                                group:ml-team@example.com
                              May be passed multiple times.
      --grant-writer PRINCIPAL
                              Same as --grant-reader but grants
                              roles/storage.objectUser (read+write+delete).
                              May be passed multiple times.
      --hierarchical-namespace
                              Create the bucket with hierarchical namespace
                              enabled (better for high metadata QPS).
      --dry-run               Print actions without executing them.
  -h, --help                  Show this help and exit.

Environment overrides (flags win):
  GCHAT_BUCKET, GCHAT_PROJECT, GCHAT_LOCATION,
  GCHAT_STORAGE_CLASS, GCHAT_SOURCE, GCHAT_DEST_PREFIX,
  GCHAT_TOKEN_BYTES

Examples:
  ${SCRIPT_NAME} --bucket my-gchat-data --location us-central2
  ${SCRIPT_NAME} -b my-bucket -l europe-west4 \\
      --grant-reader serviceAccount:123-compute@developer.gserviceaccount.com
  ${SCRIPT_NAME} -b my-bucket --token-bytes /tmp/token_bytes.npy
EOF
}

BUCKET=${GCHAT_BUCKET:-}
PROJECT=${GCHAT_PROJECT:-}
LOCATION=${GCHAT_LOCATION:-us-central2}
STORAGE_CLASS=${GCHAT_STORAGE_CLASS:-STANDARD}
SOURCE_DIR=${GCHAT_SOURCE:-./climbmix_tokens}
DEST_PREFIX=${GCHAT_DEST_PREFIX:-gchat/climbmix_tokens}
TOKEN_BYTES_OVERRIDE=${GCHAT_TOKEN_BYTES:-}
HIERARCHICAL_NS=false
DRY_RUN=false
READER_PRINCIPALS=()
WRITER_PRINCIPALS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--bucket)                BUCKET=$2;                  shift 2 ;;
    -p|--project)               PROJECT=$2;                 shift 2 ;;
    -l|--location)              LOCATION=$2;                shift 2 ;;
    -c|--storage-class)         STORAGE_CLASS=$2;           shift 2 ;;
    -s|--source)                SOURCE_DIR=$2;              shift 2 ;;
    -d|--dest-prefix)           DEST_PREFIX=$2;             shift 2 ;;
    -t|--token-bytes)           TOKEN_BYTES_OVERRIDE=$2;    shift 2 ;;
    --grant-reader)             READER_PRINCIPALS+=("$2");  shift 2 ;;
    --grant-writer)             WRITER_PRINCIPALS+=("$2");  shift 2 ;;
    --hierarchical-namespace)   HIERARCHICAL_NS=true;       shift ;;
    --dry-run)                  DRY_RUN=true;               shift ;;
    -h|--help)                  usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

log()  { printf '[%s] %s\n'   "${SCRIPT_NAME}" "$*"; }
warn() { printf '[%s] WARN: %s\n' "${SCRIPT_NAME}" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "${SCRIPT_NAME}" "$*" >&2; exit 1; }

run() {
  if $DRY_RUN; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

command -v gcloud >/dev/null || die "gcloud is not installed or not on PATH."

[[ -n ${BUCKET} ]] || { usage; die "--bucket is required."; }
[[ ${BUCKET} != gs://* ]] || die "--bucket must be a name, not a gs:// URI."

if [[ -z ${PROJECT} ]]; then
  PROJECT=$(gcloud config get-value core/project 2>/dev/null || true)
  [[ -n ${PROJECT} ]] || die "No project set. Use --project or 'gcloud config set project PROJECT'."
fi

[[ -d ${SOURCE_DIR} ]] || die "Source directory not found: ${SOURCE_DIR}"

shopt -s nullglob
SHARDS=( "${SOURCE_DIR}"/tokens-*.arrayrecord "${SOURCE_DIR}"/test-tokens*.arrayrecord )
shopt -u nullglob
[[ ${#SHARDS[@]} -gt 0 ]] || die "No tokens-*.arrayrecord files found under ${SOURCE_DIR}."

BUCKET_URI="gs://${BUCKET}"
DEST_URI="${BUCKET_URI}/${DEST_PREFIX%/}"

log "Project:        ${PROJECT}"
log "Bucket:         ${BUCKET_URI}  (${LOCATION}, ${STORAGE_CLASS})"
log "Source:         ${SOURCE_DIR}  (${#SHARDS[@]} shard file(s))"
log "Destination:    ${DEST_URI}"
$HIERARCHICAL_NS && log "Hierarchical namespace: enabled"
$DRY_RUN && log "Dry-run mode: no changes will be made."

if gcloud storage buckets describe "${BUCKET_URI}" --project "${PROJECT}" >/dev/null 2>&1; then
  log "Bucket already exists; skipping creation."
  EXISTING_LOCATION=$(gcloud storage buckets describe "${BUCKET_URI}" \
    --project "${PROJECT}" --format='value(location)' 2>/dev/null | tr '[:upper:]' '[:lower:]')
  REQUESTED_LOCATION=$(echo "${LOCATION}" | tr '[:upper:]' '[:lower:]')
  if [[ -n ${EXISTING_LOCATION} && ${EXISTING_LOCATION} != "${REQUESTED_LOCATION}" ]]; then
    warn "Existing bucket is in '${EXISTING_LOCATION}', but --location is '${REQUESTED_LOCATION}'."
    warn "Continuing, but uploads will land in '${EXISTING_LOCATION}'."
  fi
else
  log "Creating bucket ${BUCKET_URI} ..."
  CREATE_ARGS=(
    storage buckets create "${BUCKET_URI}"
    --project "${PROJECT}"
    --location "${LOCATION}"
    --default-storage-class "${STORAGE_CLASS}"
    --uniform-bucket-level-access
  )
  $HIERARCHICAL_NS && CREATE_ARGS+=( --enable-hierarchical-namespace )
  run gcloud "${CREATE_ARGS[@]}"
fi

grant_role() {
  local role=$1 principal=$2
  log "Granting ${role} to ${principal} on ${BUCKET_URI}"
  run gcloud storage buckets add-iam-policy-binding "${BUCKET_URI}" \
    --project "${PROJECT}" \
    --member="${principal}" \
    --role="${role}" \
    --condition=None \
    >/dev/null
}

for p in "${READER_PRINCIPALS[@]:-}"; do [[ -n ${p} ]] && grant_role roles/storage.objectViewer "${p}"; done
for p in "${WRITER_PRINCIPALS[@]:-}"; do [[ -n ${p} ]] && grant_role roles/storage.objectUser   "${p}"; done

TOKEN_BYTES_DEFAULT="${SOURCE_DIR}/token_bytes.npy"
TOKEN_BYTES_LOCAL=${TOKEN_BYTES_OVERRIDE:-${TOKEN_BYTES_DEFAULT}}
TOKEN_BYTES_DEST="${DEST_URI}/token_bytes.npy"
TOKENIZER_DIR="${SOURCE_DIR}/tokenizer"

if [[ ! -f ${TOKEN_BYTES_LOCAL} ]]; then
  die "token_bytes.npy not found at ${TOKEN_BYTES_LOCAL}. Generate one with: \
gchat-token-bytes --tokenizer gpt2 --output ${TOKEN_BYTES_LOCAL}"
fi

log "Uploading ${#SHARDS[@]} shard(s) to ${DEST_URI}/ ..."
run gcloud storage rsync --recursive \
  --exclude='^shard_cache/' \
  --project "${PROJECT}" \
  "${SOURCE_DIR%/}" "${DEST_URI}/"

if [[ ${TOKEN_BYTES_LOCAL} != "${TOKEN_BYTES_DEFAULT}" ]]; then
  log "Uploading token_bytes.npy from ${TOKEN_BYTES_LOCAL} to ${TOKEN_BYTES_DEST}"
  run gcloud storage cp \
    --project "${PROJECT}" \
    "${TOKEN_BYTES_LOCAL}" "${TOKEN_BYTES_DEST}"
else
  log "token_bytes.npy uploaded as part of rsync from ${TOKEN_BYTES_LOCAL}"
fi

if [[ -d ${TOKENIZER_DIR} ]]; then
  log "Confirmed tokenizer/ directory present in upload set."
else
  warn "No tokenizer/ directory under ${SOURCE_DIR}; consider pinning a tokenizer.json next to the shards."
fi

log "Done. To read from training code set:"
log "  --data-dir         ${DEST_URI}"
log "  --token-bytes-path ${TOKEN_BYTES_DEST}"
