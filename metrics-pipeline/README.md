## Gaia Metrics Pipeline
This directory contains configuration files and an installation script
to install an OTel Collector and ClickHouse instance in a Kubernetes cluster
and expose it at a subdomain. 

## Dependencies

The install script requires `kubectl`, `helm`, `yq`, and `htpasswd`.

## Installation

The install script deploys metrics pipeline components into the `gaia-metrics` namespace as `gaia-clickhouse` and `gaia-otel-collector`.

It also creates an ingress for `otel.gerardosalazar.com` that routes authenticated OTLP/HTTP traffic to the collector service on port `4318`.

## Connectivity Test

Send OTLP/HTTP metrics with Basic Auth:

```bash
curl -u "${OTEL_COLLECTOR_USERNAME}:${OTEL_COLLECTOR_PASSWORD}" \
  https://otel.gerardosalazar.com/v1/metrics \
  -H 'Content-Type: application/json' \
  -d @payload.json
```

## Local gchat Smoke Run

Port-forward the collector service:

```bash
kubectl -n gaia-metrics port-forward svc/gaia-otel-collector 4318:4318
```

In another terminal, point `gchat` at the local collector endpoint and provide
the OTLP Basic Auth credentials:

```bash
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT="http://localhost:4318/v1/metrics"
export OTEL_COLLECTOR_USERNAME="otel"
export OTEL_COLLECTOR_PASSWORD="<your collector password>"

uv run python -m gchat.training.train \
  --data-dir climbmix_tokens \
  --token-bytes-path /path/to/token_bytes.npy \
  --batch-size 1 \
  --eval-batch-size 1 \
  --sequence-length 128 \
  --log-every 1 \
  --eval-every 2 \
  --eval-split-tokens 128 \
  --no-repeat
```

The run uses the fixed metrics run id `gchat-test`. `--eval-every 2` performs a
validation BPB evaluation at step 2, and `--eval-split-tokens 128` keeps the
evaluation to one tiny batch.
