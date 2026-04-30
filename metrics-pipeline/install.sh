#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METRICS_NAMESPACE="gaia-metrics"
CLICKHOUSE_NAME="gaia-clickhouse"
OTEL_COLLECTOR_NAME="gaia-otel-collector"
METRICS_SECRET_NAME="gaia-metrics-secrets"
CLICKHOUSE_MANIFEST="${SCRIPT_DIR}/clickhouse.yaml"
CLICKHOUSE_SCHEMA="${SCRIPT_DIR}/schema.sql"
OTEL_COLLECTOR_VALUES="${SCRIPT_DIR}/otel-collector.yaml"
OTEL_COLLECTOR_INGRESS="${SCRIPT_DIR}/otel-collector-ingress.yaml"
CLICKHOUSE_POD_SELECTOR="clickhouse.altinity.com/chi=${CLICKHOUSE_NAME},clickhouse.altinity.com/cluster=${CLICKHOUSE_NAME},clickhouse.altinity.com/shard=0,clickhouse.altinity.com/replica=0"

ALTINITY_OPERATOR_MANIFEST="https://raw.githubusercontent.com/Altinity/clickhouse-operator/master/deploy/operator/clickhouse-operator-install-bundle.yaml"

require_command() {
  local command_name="$1"

  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 1
  fi
}

require_command kubectl
require_command helm
require_command yq
require_command shasum

prompt_required() {
  local variable_name="$1"
  local prompt="$2"
  local secret="${3:-false}"
  local value=""

  if [[ ! -t 0 ]]; then
    echo "Missing required environment variable: ${variable_name}" >&2
    echo "Run interactively or provide ${variable_name} in the environment." >&2
    exit 1
  fi

  while [[ -z "${value}" ]]; do
    if [[ "${secret}" == "true" ]]; then
      read -rsp "${prompt}: " value
      echo
    else
      read -rp "${prompt}: " value
    fi

    if [[ -z "${value}" ]]; then
      echo "${variable_name} cannot be empty." >&2
    fi
  done

  printf -v "${variable_name}" "%s" "${value}"
}

if [[ -z "${CLICKHOUSE_OTEL_PASSWORD:-}" ]]; then
  prompt_required CLICKHOUSE_OTEL_PASSWORD "ClickHouse password for otel user" true
fi

if [[ -z "${OTEL_COLLECTOR_HTPASSWD:-}" ]]; then
  require_command htpasswd

  if [[ -z "${OTEL_COLLECTOR_USERNAME:-}" ]]; then
    prompt_required OTEL_COLLECTOR_USERNAME "OTLP Basic Auth username"
  fi

  if [[ -z "${OTEL_COLLECTOR_PASSWORD:-}" ]]; then
    prompt_required OTEL_COLLECTOR_PASSWORD "OTLP Basic Auth password" true
  fi

  OTEL_COLLECTOR_HTPASSWD="$(htpasswd -nbB "${OTEL_COLLECTOR_USERNAME}" "${OTEL_COLLECTOR_PASSWORD}")"
fi

if [[ ! -f "${CLICKHOUSE_SCHEMA}" ]]; then
  echo "Missing ClickHouse schema: ${CLICKHOUSE_SCHEMA}" >&2
  exit 1
fi

if [[ ! -f "${OTEL_COLLECTOR_VALUES}" ]]; then
  echo "Missing OpenTelemetry Collector values: ${OTEL_COLLECTOR_VALUES}" >&2
  exit 1
fi

if [[ ! -f "${OTEL_COLLECTOR_INGRESS}" ]]; then
  echo "Missing OpenTelemetry Collector ingress: ${OTEL_COLLECTOR_INGRESS}" >&2
  exit 1
fi

echo "Installing Altinity ClickHouse operator..."
kubectl apply -f "${ALTINITY_OPERATOR_MANIFEST}"

echo "Creating metrics namespace..."
kubectl create namespace "${METRICS_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

echo "Creating metrics secrets..."
kubectl create secret generic "${METRICS_SECRET_NAME}" \
  --namespace "${METRICS_NAMESPACE}" \
  --from-literal=clickhouse-password="${CLICKHOUSE_OTEL_PASSWORD}" \
  --from-literal=otel-collector-htpasswd="${OTEL_COLLECTOR_HTPASSWD}" \
  --dry-run=client \
  -o yaml | kubectl apply -f -

CLICKHOUSE_OTEL_PASSWORD_SHA256="$(printf "%s" "${CLICKHOUSE_OTEL_PASSWORD}" | shasum -a 256 | awk '{print $1}')"

echo "Installing ClickHouse..."
CLICKHOUSE_NAME="${CLICKHOUSE_NAME}" METRICS_NAMESPACE="${METRICS_NAMESPACE}" CLICKHOUSE_OTEL_PASSWORD_SHA256="${CLICKHOUSE_OTEL_PASSWORD_SHA256}" yq eval \
  '.metadata.name = env(CLICKHOUSE_NAME) | .metadata.namespace = env(METRICS_NAMESPACE) | .spec.configuration.users."otel/password_sha256_hex" = env(CLICKHOUSE_OTEL_PASSWORD_SHA256)' \
  "${CLICKHOUSE_MANIFEST}" | kubectl apply -f -

echo "Waiting for ClickHouse pod to be created..."
CLICKHOUSE_POD=""
for _ in {1..60}; do
  CLICKHOUSE_POD="$(kubectl get pod --namespace "${METRICS_NAMESPACE}" \
    --selector "${CLICKHOUSE_POD_SELECTOR}" \
    --output jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"

  if [[ -n "${CLICKHOUSE_POD}" ]]; then
    break
  fi

  sleep 5
done

if [[ -z "${CLICKHOUSE_POD}" ]]; then
  echo "Timed out waiting for ClickHouse pod to be created." >&2
  exit 1
fi

echo "Waiting for ClickHouse pod to become ready..."
kubectl wait --namespace "${METRICS_NAMESPACE}" \
  --for=condition=Ready "pod/${CLICKHOUSE_POD}" \
  --timeout=300s

echo "Applying ClickHouse schema..."
kubectl exec --stdin --namespace "${METRICS_NAMESPACE}" "${CLICKHOUSE_POD}" -- \
  clickhouse-client --multiquery <"${CLICKHOUSE_SCHEMA}"

echo "Installing OpenTelemetry Collector Helm chart..."
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo update open-telemetry

helm upgrade --install "${OTEL_COLLECTOR_NAME}" open-telemetry/opentelemetry-collector \
  --namespace "${METRICS_NAMESPACE}" \
  --values "${OTEL_COLLECTOR_VALUES}"

echo "Installing OpenTelemetry Collector ingress..."
METRICS_NAMESPACE="${METRICS_NAMESPACE}" yq eval \
  '.metadata.namespace = env(METRICS_NAMESPACE)' \
  "${OTEL_COLLECTOR_INGRESS}" | kubectl apply -f -

echo "Metrics pipeline install complete."
