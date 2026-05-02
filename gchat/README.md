# gchat

My implementation of [nanochat](https://github.com/karpathy/nanochat) in JAX trained on TPUs.

## Speedrun

Train on a TPU VM using tokenized ArrayRecord shards in GCS:

```bash
export OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://otel.gerardosalazar.com/v1/metrics
export OTEL_COLLECTOR_USERNAME=otel
export OTEL_COLLECTOR_PASSWORD="<your collector password>"

GCHAT_DATA_DIR=gs://my-bucket/gchat/climbmix_tokens \
GCHAT_CHECKPOINT_BUCKET=my-bucket \
bash gchat/speedrun.sh
```

`speedrun.sh` avoids listing the GCS data prefix during startup. It expects
`tokens-00000.arrayrecord` through `tokens-00053.arrayrecord` by default; set
`GCHAT_TOKEN_SHARD_COUNT` if you upload more training shards.

The `OTEL_*` variables are optional unless exporting metrics to the authenticated collector.

