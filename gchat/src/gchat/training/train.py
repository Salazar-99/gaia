from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from gchat.training.checkpoint import save_model
from gchat.training.data import (
    DatasetConfig,
    build_training_dataset,
    list_arrayrecord_files,
)
from gchat.training.evals import evaluate_bpb
from gchat.training.model import GPTConfig, GPT
from gaia_metrics import initialize_metrics_from_env


METRICS_RUN_ID_ENV = "GCHAT_METRICS_RUN_ID"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain GPT on token batches (single-device JAX; no sharding)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="climbmix_tokens",
        help=(
            "Directory containing tokens-*.arrayrecord files. "
            "Accepts a local path or a gs://bucket/prefix URI; gs:// is "
            "streamed directly by Grain/ArrayRecord."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help=(
            "Local directory to write the final checkpoint into. Ignored "
            "when --gcs-checkpoint-bucket is set."
        ),
    )
    parser.add_argument(
        "--gcs-checkpoint-bucket",
        type=str,
        default=None,
        help=(
            "If set, write the final checkpoint to "
            "gs://<bucket>/<gcs-checkpoint-prefix>/<timestamp>/ instead of "
            "--checkpoint-dir. Requires write access to the bucket."
        ),
    )
    parser.add_argument(
        "--gcs-checkpoint-prefix",
        type=str,
        default="gchat",
        help=(
            "Object prefix inside --gcs-checkpoint-bucket under which each "
            "run's <timestamp>/ directory is written (default: gchat)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument(
        "--n-layer",
        type=int,
        default=24,
        help="Number of transformer blocks.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable dataset shuffle.",
    )
    parser.add_argument(
        "--no-repeat",
        action="store_true",
        help="Do not repeat the dataset; stop when the input data is exhausted.",
    )
    parser.add_argument(
        "--gcs-token-shard-count",
        type=int,
        default=None,
        help=(
            "For gs:// data dirs, construct tokens-00000.arrayrecord through "
            "tokens-<N-1>.arrayrecord instead of listing the GCS prefix with "
            "gcloud. Local data dirs still use filesystem globbing."
        ),
    )

    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Run BPB evaluation every N steps (0 = disabled unless --eval-at-end).",
    )
    eval_group.add_argument(
        "--eval-at-end",
        action="store_true",
        help="Run BPB evaluation once after training finishes (no periodic eval).",
    )
    eval_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Batch size for BPB evaluation.",
    )
    eval_group.add_argument(
        "--eval-split-tokens",
        type=int,
        default=40 * 524_288,
        help="Total tokens to evaluate per BPB run (steps derived automatically).",
    )
    eval_group.add_argument(
        "--token-bytes-path",
        type=str,
        default=None,
        help=(
            "Local path or gs:// URI to a .npy file of shape (vocab_size,) "
            "with byte lengths per token. gs:// URIs are downloaded once to "
            "--host-cache-dir before use."
        ),
    )
    parser.add_argument(
        "--host-cache-dir",
        type=Path,
        default=Path(os.environ.get("GCHAT_HOST_CACHE", "/tmp/gchat")),
        help=(
            "Local directory to cache small assets downloaded from gs:// "
            "(e.g. token_bytes.npy). Default: /tmp/gchat or "
            "$GCHAT_HOST_CACHE."
        ),
    )
    return parser.parse_args()


def _fetch_to_host(uri: str, cache_dir: Path) -> Path:
    """Materialize a local path for a local filesystem path or gs:// URI.

    For gs:// URIs, downloads the object to cache_dir/<basename> using
    `gcloud storage cp` (no extra Python deps). The copy is skipped if a
    non-empty local file already exists. Safe to call from every host in a
    multi-host TPU job: each host maintains its own copy in its own /tmp.
    """
    if not uri.startswith("gs://"):
        return Path(uri).expanduser().resolve()

    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / Path(uri).name

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    print(f"[host {jax.process_index()}] downloading {uri} -> {local_path}")
    subprocess.run(
        ["gcloud", "storage", "cp", uri, str(local_path)],
        check=True,
    )
    return local_path


def _load_token_bytes(path: Path) -> jax.Array:
    return jnp.asarray(np.load(path))


TEST_SHARD_GLOB = "test-tokens*.arrayrecord"


def _build_eval_state(args: argparse.Namespace) -> dict | None:
    """Prepare everything needed for BPB evaluation, or return None."""
    if args.eval_every <= 0 and not args.eval_at_end:
        return None
    if args.token_bytes_path is None:
        raise ValueError(
            "--token-bytes-path is required when evaluation is enabled "
            "(--eval-every > 0 or --eval-at-end)."
        )

    expected_test_files = None
    if args.data_dir.startswith("gs://"):
        expected_test_files = ("test-tokens.arrayrecord",)
    test_files = list_arrayrecord_files(
        args.data_dir,
        TEST_SHARD_GLOB,
        expected_gcs_file_names=expected_test_files,
    )

    token_bytes_local = _fetch_to_host(args.token_bytes_path, args.host_cache_dir)
    token_bytes = _load_token_bytes(token_bytes_local)
    tokens_per_step = args.eval_batch_size * args.sequence_length
    split_tokens = (args.eval_split_tokens // tokens_per_step) * tokens_per_step
    eval_steps = split_tokens // tokens_per_step

    eval_config = DatasetConfig(
        data_dir=args.data_dir,
        token_file_glob=TEST_SHARD_GLOB,
        batch_size=args.eval_batch_size,
        sequence_length=args.sequence_length,
        shuffle=False,
        repeat=True,
        seed=None,
        expected_gcs_file_names=expected_test_files,
    )
    test_file_names = [f.rsplit("/", 1)[-1] for f in test_files]
    if args.eval_at_end and args.eval_every <= 0:
        schedule = "once at end of training"
    elif args.eval_at_end:
        schedule = f"every {args.eval_every} steps and at end of training"
    else:
        schedule = f"every {args.eval_every} steps"
    print(
        f"BPB eval: {schedule}, "
        f"{eval_steps} steps/run ({split_tokens:,} tokens), "
        f"batch_size={args.eval_batch_size}, "
        f"test files: {test_file_names}"
    )
    return {
        "config": eval_config,
        "steps": eval_steps,
        "token_bytes": token_bytes,
    }


def _run_eval(model: GPT, eval_state: dict) -> float:
    eval_batches = build_training_dataset(eval_state["config"])
    bpb = evaluate_bpb(
        model, eval_batches, eval_state["steps"], eval_state["token_bytes"]
    )
    return bpb


def main() -> None:
    args = _parse_args()

    data_config = DatasetConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        shuffle=not args.no_shuffle,
        repeat=not args.no_repeat,
        expected_gcs_token_file_count=args.gcs_token_shard_count,
    )
    batches = build_training_dataset(data_config)

    gpt_config = GPTConfig(
        sequence_len=args.sequence_length,
        n_layer=args.n_layer,
    )
    rngs = nnx.Rngs(jax.random.key(args.seed))
    model = GPT(gpt_config, rngs)
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(args.learning_rate, mu_dtype=jnp.float32),
        wrt=nnx.Param,
    )

    eval_state = _build_eval_state(args)
    metrics_run_id = os.environ.get(METRICS_RUN_ID_ENV, "gchat-test")
    training_loss_gauge, validation_loss_gauge, _ = initialize_metrics_from_env(
        run_id=metrics_run_id
    )
    print(f"Metrics initialized with run_id={metrics_run_id}")

    @nnx.jit
    def train_step(
        model: GPT,
        optimizer: nnx.Optimizer,
        inputs: jax.Array,
        targets: jax.Array,
    ) -> jax.Array:
        def loss_fn(m: GPT) -> jax.Array:
            logits = m(inputs)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            ).astype(jnp.float32).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    print("Starting training")
    for step, batch in enumerate(batches):
        inputs = jnp.asarray(batch["inputs"])
        targets = jnp.asarray(batch["targets"])
        loss = train_step(model, optimizer, inputs, targets)

        if step % args.log_every == 0:
            loss_val = float(jax.block_until_ready(loss))
            print(f"step {step:6d}  loss {loss_val:.4f}")
            training_loss_gauge.set(loss_val)

        if (
            eval_state
            and args.eval_every > 0
            and step > 0
            and step % args.eval_every == 0
        ):
            bpb = _run_eval(model, eval_state)
            print(f"step {step:6d}  bpb  {bpb:.6f}")
            validation_loss_gauge.set(bpb)

    if eval_state and args.eval_at_end:
        final_step = step
        bpb = _run_eval(model, eval_state)
        print(f"step {final_step:6d}  bpb  {bpb:.6f}")
        validation_loss_gauge.set(bpb)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.gcs_checkpoint_bucket:
        prefix = args.gcs_checkpoint_prefix.strip("/")
        checkpoint_dir = f"gs://{args.gcs_checkpoint_bucket}/{prefix}/{timestamp}"
    else:
        checkpoint_dir = str((args.checkpoint_dir / timestamp).resolve())
    print(f"Saving final model checkpoint to {checkpoint_dir}")
    save_model(model, checkpoint_dir)


if __name__ == "__main__":
    main()
