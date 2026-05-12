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

## Scaling dashboard

Install the optional dashboard dependencies and launch the FastHTML memory
estimator:

```bash
uv run --extra dashboard gchat-scaling-dashboard
```

Then open `http://127.0.0.1:5001`.

The dashboard is a rough sizing tool for comparing model shape, training data
size, HBM capacity, and idealized TPU training time. The sliders update the
estimate in the browser; `Training tokens` is entered in billions, and `Data
parallelism` is treated as chips per model replica for the fit checks.

The parameter estimate is:

```text
head_dim = n_embd / n_head
embedding_params = vocab_size * n_embd
lm_head_params = vocab_size * n_embd
qkv_params = n_embd * ((n_head + 2 * n_kv_head) * head_dim)
attention_projection_params = n_embd * n_embd
mlp_params = 8 * n_embd * n_embd
block_params = qkv_params + attention_projection_params + mlp_params
total_params = embedding_params + lm_head_params + n_layer * block_params
```

The memory estimate adds:

```text
parameter_memory = total_params * parameter_bytes
gradient_memory = total_params * parameter_bytes
adamw_state = total_params * 2 * 4
hidden_activations = batch_size * sequence_len * n_embd * activation_bytes * (n_layer + 1)
attention_workspace = n_layer * batch_size * n_head * sequence_len * ((sequence_len + sequence_len / 2) / 2) * activation_bytes
total_memory = parameter_memory + gradient_memory + adamw_state + hidden_activations + attention_workspace
```

The HBM grid checks whether one model replica fits on each TPU type:

```text
replica_hbm = hbm_per_chip * data_parallelism
fits = column_chip_count >= data_parallelism and replica_hbm >= total_memory
```

The training time estimate uses the Chinchilla-style training FLOPs rule and
assumes 50% MFU:

```text
total_flops = 6 * total_params * training_tokens
seconds = total_flops / (bf16_peak_tflops_per_chip * 1e12 * column_chip_count * 0.5)
```

Cells are marked red when the selected model replica needs more chips than the
column provides, or when that TPU type does not have enough per-replica HBM.

