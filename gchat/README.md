# gchat

My implementation of [nanochat](https://github.com/karpathy/nanochat) in JAX trained on TPUs.

## Speedrun

Train on a TPU VM using the default YAML config:

```bash
bash gchat/speedrun.sh
```

To use another config:

```bash
bash gchat/speedrun.sh gchat/conf/my_run.yaml
```

`gchat/conf/speedrun.yaml` is the source of truth for training settings. The
default data directory must contain `token_bytes.npy` and
`tokens-00000.arrayrecord` through `tokens-00053.arrayrecord`. For `gs://` data,
checkpoints default to the same bucket and prefix under `checkpoint/<timestamp>/`.

### Speedrun Configuration

| YAML key | Default | Meaning |
| --- | --- | --- |
| `runtime.omp_num_threads` | `1` | Sets `OMP_NUM_THREADS` before JAX starts. |
| `runtime.base_dir` | `~/.cache/gchat` | Local base directory for profile logs. |
| `metrics.enabled` | `true` | Enables OTLP metric export through `gaia-metrics`. |
| `metrics.run_id` | `gchat-test` | Run label attached to metrics. |
| `metrics.endpoint` | `https://otel.gerardosalazar.com/v1/metrics` | OTLP metrics endpoint; set to `null` for console metrics. |
| `metrics.username` / `metrics.password` | `otel` / placeholder | Basic-auth credentials for the metrics endpoint. |
| `data.data_dir` | `gs://gchat-climbmix-7b/data` | Directory containing ArrayRecord token shards and `token_bytes.npy`. |
| `data.batch_size` | `16` | Training batch size. |
| `data.sequence_length` | `1024` | Tokens per training sequence. |
| `data.shuffle` | `true` | Shuffles the Grain dataset. |
| `data.gcs_token_shard_count` | `54` | Number of expected `tokens-*.arrayrecord` shards for GCS data. |
| `model.n_layer` | `12` | Number of transformer blocks. |
| `model.n_head` / `model.n_kv_head` | `8` / `8` | Query and key/value attention head counts. |
| `model.n_embd` | `1536` | Transformer hidden width. |
| `model.window_pattern` | `SSSL` | Sliding attention pattern; `S` is half-context, `L` is full-context. |
| `training.learning_rate` | `3.0e-4` | AdamW learning rate. |
| `training.seed` | `0` | Model initialization seed. |
| `training.log_every` | `1` | Training loss log interval in steps, written through the delayed `RecordWriter`. |
| `eval.every` | `250` | Runs BPB eval periodically after step 0. |
| `eval.batch_size` | `16` | Eval batch size. |
| `eval.split_tokens` | `1310720` | Number of test tokens consumed per BPB eval run. |
| `profiling.enabled` | `false` | Runs only the short JAX profiler loop when `true`. |
| `profiling.log_dir` | `null` | Profile output directory; defaults to `<runtime.base_dir>/profiles`. |
| `profiling.warmup_steps` / `profiling.steps` | `5` / `3` | Warmup steps before tracing and traced training steps. |

## This Branch

The `gchat/main` branch trains a compact GChat transformer on TPU with bf16
parameters and activations. The default speedrun shape is 12 layers, width 1536,
8 query heads, 8 key/value heads, and 1024-token sequences. Attention uses the
`SSSL` sliding-window pattern, tiled across layers, with the final layer forced
to full-context attention. The model includes RoPE, QK RMS normalization, a
squared-ReLU MLP, and a tied-shape but separate output projection head.

## Scaling dashboard

The scaling dashboard lives at `src/gchat/scaling/dashboard.py`. Install the
optional dashboard dependencies and launch the FastHTML memory estimator:

```bash
cd gchat
uv run --extra dashboard gchat-scaling-dashboard
```

Or run the module directly:

```bash
cd gchat
uv run --extra dashboard python -m gchat.scaling.dashboard
```

Then open `http://127.0.0.1:5001`.

To run it in Docker, build from the `gchat` directory. If deploying to an
amd64 cluster from Apple Silicon, build for that platform:

```bash
cd gchat
docker build --platform linux/amd64 -f src/gchat/scaling/Dockerfile -t gimages.azurecr.io/gchat-scaling-dashboard:1.0 .
docker run --rm -p 5001:5001 gimages.azurecr.io/gchat-scaling-dashboard:1.0
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

