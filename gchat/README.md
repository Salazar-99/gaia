# gchat

My implementation of [nanochat](https://github.com/karpathy/nanochat) in JAX trained on TPUs.

## Speedrun

Train on a TPU VM using tokenized ArrayRecord shards in GCS:

```bash
bash gchat/speedrun.sh
```

The data directory must contain `token_bytes.npy` and the token shards
`tokens-00000.arrayrecord` through `tokens-00053.arrayrecord`. Checkpoints are
written under the same GCS path at `checkpoint/<timestamp>/`. Edit
`gchat/speedrun.sh` directly to change the dataset or metrics collector.

### Speedrun configuration

Edit the defaults in `gchat/speedrun.sh` before starting a run:

| Setting | Default | What it controls | How to change it |
| --- | --- | --- | --- |
| `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` | `https://otel.gerardosalazar.com/v1/metrics` | Metrics collector endpoint. | Replace the URL in the editable config block. |
| `OTEL_COLLECTOR_USERNAME` | `otel` | Metrics collector username. | Replace the default username in the editable config block. |
| `OTEL_COLLECTOR_PASSWORD` | `<your collector password>` | Metrics collector password. | Replace the placeholder password in the editable config block. |
| `GCHAT_DATA_DIR` | `gs://gchat-climbmix-7b/data` | GCS directory containing `token_bytes.npy` and token shards. | Replace the default `gs://.../data` path. |
| `GCHAT_TOKEN_BYTES_PATH` | `${GCHAT_DATA_DIR%/}/token_bytes.npy` | Vocabulary byte metadata file. | Leave derived from `GCHAT_DATA_DIR` unless the file lives elsewhere. |
| `GCHAT_SEQUENCE_LENGTH` | `2048` | Tokens per training sequence. | Edit the numeric default in the training defaults block. |
| `GCHAT_BATCH_SIZE` | `16` | Per-step batch size. | Edit the numeric default in the training defaults block. |
| `GCHAT_LEARNING_RATE` | `3e-4` | AdamW learning rate. | Edit the default learning-rate value. |
| `GCHAT_SEED` | `0` | Random seed for training. | Edit the integer seed value. |
| `GCHAT_LOG_EVERY` | `1` | Training log frequency in steps. | Edit the step interval. |
| `GCHAT_EVAL_EVERY` | `250` | Eval frequency in steps. | Edit the step interval. |
| `GCHAT_EVAL_BATCH_SIZE` | `16` | Eval batch size. | Edit the numeric default. |
| `GCHAT_EVAL_SPLIT_TOKENS` | `41943040` | Number of tokens reserved for eval. | Edit the token count. |
| `GCHAT_TOKEN_SHARD_COUNT` | `54` | Number of training shards to read. | Match this to the uploaded `tokens-*.arrayrecord` shard count. |
| `GCHAT_NO_REPEAT` | `1` | Whether training stops after one pass over the shards. | Keep `1` for one pass; set to `0` to repeat indefinitely. |
| `GCHAT_NO_SHUFFLE` | `0` | Whether training disables dataset shuffle. | Keep `0` to shuffle; set to `1` to disable shuffle. |

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

