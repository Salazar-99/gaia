from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote

from fasthtml.common import (
    Div,
    Form,
    H1,
    H2,
    Input,
    Label,
    Li,
    Link,
    Main,
    Meta,
    P,
    Script,
    Section,
    Span,
    Strong,
    Style,
    Ul,
    fast_app,
    serve,
)


BYTES_PER_GIB = 1024**3
TRAINING_MFU = 0.5
POWER_OF_TWO_SLIDERS = frozenset({"data_parallelism"})


@dataclass(frozen=True)
class Slider:
    name: str
    label: str
    default: int
    minimum: int
    maximum: int
    step: int
    help: str


@dataclass(frozen=True)
class Tpu:
    name: str
    hbm_gb: int
    bf16_peak_tflops: int
    max_chips: int
    topology: str


SLIDERS: tuple[Slider, ...] = (
    Slider("batch_size", "Batch size", 16, 1, 64, 1, "Training examples per step."),
    Slider(
        "sequence_len", "Sequence length", 2048, 256, 32768, 256, "Tokens per example."
    ),
    Slider(
        "vocab_size",
        "Vocabulary size",
        50257,
        16000,
        200000,
        1000,
        "Tokenizer vocabulary entries.",
    ),
    Slider("n_layer", "Layers", 24, 1, 96, 1, "Transformer blocks."),
    Slider("n_head", "Query heads", 12, 1, 64, 1, "Attention query heads."),
    Slider("n_kv_head", "KV heads", 12, 1, 64, 1, "Key/value heads for GQA."),
    Slider("n_embd", "Embedding width", 1536, 128, 8192, 128, "Hidden dimension."),
    Slider("param_bytes", "Parameter bytes", 2, 1, 4, 1, "2 = bf16/fp16, 4 = fp32."),
    Slider(
        "activation_bytes", "Activation bytes", 2, 1, 4, 1, "2 = bf16/fp16, 4 = fp32."
    ),
    Slider(
        "training_tokens",
        "Training tokens",
        7,
        1,
        32,
        1,
        "Total tokens in the training dataset, in billions.",
    ),
    Slider(
        "data_parallelism",
        "Data parallelism",
        1,
        1,
        32,
        1,
        "Chips used by one model replica; remaining chips form data-parallel replicas.",
    ),
)

TPUS: tuple[Tpu, ...] = (
    Tpu("TPU 8i", 288, 5050, 1152, "Boardfly (Inference)"),
    Tpu("TPU 8t", 216, 6300, 9600, "3D Torus (Training)"),
    Tpu("TPU v7 (Ironwood)", 192, 2307, 9216, "3D Torus"),
    Tpu("TPU v6e (Trillium)", 32, 918, 256, "2D Torus"),
    Tpu("TPU v5p", 95, 459, 8960, "3D Torus"),
    Tpu("TPU v5e", 16, 197, 256, "2D Torus"),
    Tpu("TPU v4", 32, 275, 4096, "3D Torus"),
    Tpu("TPU v3", 32, 123, 2048, "2D Torus"),
)
TPU_CHIP_COUNTS = (1, 2, 4, 8, 16, 32)


def _slider_defaults() -> dict[str, int]:
    return {slider.name: slider.default for slider in SLIDERS}


def _coerce_slider_value(raw: str | None, slider: Slider) -> int:
    if raw is None:
        return slider.default
    try:
        value = int(raw)
    except ValueError:
        return slider.default
    value = max(slider.minimum, value)
    if slider.name in POWER_OF_TWO_SLIDERS:
        powers = tuple(
            count
            for count in TPU_CHIP_COUNTS
            if slider.minimum <= count <= slider.maximum
        )
        return min(powers, key=lambda power: abs(power - value))
    return value


def _query_values(request) -> dict[str, int]:
    params = request.query_params
    return {
        slider.name: _coerce_slider_value(params.get(slider.name), slider)
        for slider in SLIDERS
    }


def _format_count(value: float) -> str:
    return f"{round(value):,}"


def _format_gib(value: float) -> str:
    return f"{value / BYTES_PER_GIB:,.2f} GiB"


def _format_flops(value: float) -> str:
    scientific = f"{value:.2e}"
    units = (
        ("ZFLOPs", 10**21),
        ("EFLOPs", 10**18),
        ("PFLOPs", 10**15),
        ("TFLOPs", 10**12),
    )
    for unit, scale in units:
        if value >= scale:
            return f"{value / scale:,.2f} {unit} ({scientific})"
    return f"{value:,.0f} FLOPs ({scientific})"


def _format_duration(total_seconds: float) -> str:
    total_minutes = round(total_seconds / 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:,} hr {minutes:02d} min"


def _estimate_memory(values: dict[str, int]) -> dict[str, float | list[str]]:
    batch_size = values["batch_size"]
    sequence_len = values["sequence_len"]
    vocab_size = values["vocab_size"]
    n_layer = values["n_layer"]
    n_head = values["n_head"]
    n_kv_head = values["n_kv_head"]
    n_embd = values["n_embd"]
    param_bytes = values["param_bytes"]
    activation_bytes = values["activation_bytes"]
    training_tokens = values["training_tokens"] * 1_000_000_000

    warnings = []
    if n_embd % n_head != 0:
        warnings.append("n_embd must be divisible by n_head for the model to run.")
    if n_head % n_kv_head != 0:
        warnings.append(
            "n_head must be divisible by n_kv_head for grouped-query attention."
        )

    head_dim = n_embd / n_head
    token_embedding_params = vocab_size * n_embd
    lm_head_params = vocab_size * n_embd
    qkv_params = n_embd * ((n_head + 2 * n_kv_head) * head_dim)
    attention_projection_params = n_embd * n_embd
    mlp_params = 8 * n_embd * n_embd
    block_params = qkv_params + attention_projection_params + mlp_params
    total_params = token_embedding_params + lm_head_params + n_layer * block_params

    parameter_bytes = total_params * param_bytes
    gradient_bytes = total_params * param_bytes
    adamw_state_bytes = total_params * 2 * 4

    hidden_activation_bytes = (
        batch_size * sequence_len * n_embd * activation_bytes * (n_layer + 1)
    )
    average_attention_window = (sequence_len + sequence_len / 2) / 2
    attention_workspace_bytes = (
        n_layer
        * batch_size
        * n_head
        * sequence_len
        * average_attention_window
        * activation_bytes
    )

    total_bytes = (
        parameter_bytes
        + gradient_bytes
        + adamw_state_bytes
        + hidden_activation_bytes
        + attention_workspace_bytes
    )
    tokens_per_step = batch_size * sequence_len
    total_flops = 6 * total_params * training_tokens

    return {
        "warnings": warnings,
        "tokens_per_step": tokens_per_step,
        "total_params": total_params,
        "total_flops": total_flops,
        "parameter_bytes": parameter_bytes,
        "gradient_bytes": gradient_bytes,
        "adamw_state_bytes": adamw_state_bytes,
        "hidden_activation_bytes": hidden_activation_bytes,
        "attention_workspace_bytes": attention_workspace_bytes,
        "total_bytes": total_bytes,
    }


def _favicon() -> Link:
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
        "<text y='.9em' font-size='90'>🧠</text>"
        "</svg>"
    )
    return Link(rel="icon", href=f"data:image/svg+xml,{quote(svg)}")


def _style() -> Style:
    return Style(
        """
        :root {
            color-scheme: dark;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0d0d0d;
            color: #f5f5f5;
        }
        body {
            margin: 0;
            min-height: 100vh;
            background: linear-gradient(180deg, #151515 0%, #0d0d0d 38%, #050505 100%);
        }
        main {
            max-width: 1180px;
            margin: 0 auto;
            padding: 48px 24px;
        }
        .hero {
            margin-bottom: 28px;
        }
        h1 {
            font-size: clamp(2.2rem, 6vw, 5rem);
            line-height: 0.95;
            letter-spacing: -0.08em;
            margin: 0 0 14px;
        }
        .subtitle {
            color: #a1a1aa;
            font-size: 1.08rem;
            margin: 0;
            max-width: 760px;
        }
        .dashboard {
            display: grid;
            grid-template-columns: 1fr;
            gap: 22px;
            align-items: start;
        }
        .panel {
            background: rgba(24, 24, 27, 0.86);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 18px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.42);
            padding: 24px;
            backdrop-filter: blur(18px);
            box-sizing: border-box;
        }
        .overview-panel {
            display: grid;
            grid-template-columns: 1fr;
            gap: 34px;
            align-items: start;
        }
        .results-panel {
            min-width: 0;
        }
        .overview-panel form {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        form {
            display: grid;
            gap: 18px;
        }
        .control {
            display: grid;
            gap: 8px;
        }
        .control-header {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: baseline;
        }
        label {
            font-weight: 700;
        }
        .number-input {
            width: 132px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 10px;
            background: #0f0f10;
            color: #f4f4f5;
            padding: 7px 9px;
            text-align: right;
            font-variant-numeric: tabular-nums;
            font-weight: 700;
            outline: none;
        }
        .number-with-unit {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .unit-label {
            color: #a1a1aa;
            font-size: 0.88rem;
            font-weight: 800;
        }
        .number-input:focus {
            border-color: rgba(255, 255, 255, 0.36);
            box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.08);
        }
        input[type="range"] {
            width: 100%;
            accent-color: #f5f5f5;
        }
        .help {
            color: #8b8b92;
            font-size: 0.88rem;
        }
        .total {
            display: grid;
            gap: 4px;
            margin-bottom: 20px;
        }
        .total span:first-child {
            color: #a1a1aa;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.76rem;
            font-weight: 800;
        }
        .total strong {
            font-size: clamp(2.2rem, 7vw, 4.4rem);
            letter-spacing: -0.06em;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }
        .stat {
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 14px;
            padding: 14px;
            background: rgba(9, 9, 11, 0.58);
        }
        .stat span {
            display: block;
            color: #8b8b92;
            font-size: 0.78rem;
            margin-bottom: 6px;
        }
        .stat strong {
            font-size: 1.08rem;
            font-variant-numeric: tabular-nums;
        }
        .warnings {
            margin: 0 0 18px;
            padding-left: 20px;
            color: #f5f5f5;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 12px;
            padding: 12px 12px 12px 32px;
        }
        .tpu-panel {
            grid-column: 1 / -1;
            overflow-x: auto;
        }
        .section-title {
            margin: 0 0 8px;
            font-size: 1.28rem;
            letter-spacing: -0.03em;
        }
        .section-copy {
            color: #a1a1aa;
            margin: 0 0 18px;
        }
        .tpu-grid {
            display: grid;
            grid-template-columns: minmax(180px, 1.4fr) repeat(6, minmax(94px, 1fr));
            gap: 8px;
            min-width: 820px;
        }
        .training-time-grid {
            display: grid;
            grid-template-columns: minmax(180px, 1.4fr) repeat(6, minmax(94px, 1fr));
            gap: 8px;
            min-width: 820px;
        }
        .tpu-header,
        .tpu-name,
        .tpu-cell,
        .training-time-cell {
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 12px;
            background: rgba(9, 9, 11, 0.58);
        }
        .tpu-header {
            color: #a1a1aa;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            padding: 10px 12px;
            text-align: right;
            text-transform: uppercase;
        }
        .tpu-header:first-child {
            text-align: left;
        }
        .tpu-name {
            display: grid;
            gap: 4px;
            padding: 12px;
        }
        .tpu-name strong {
            font-size: 0.98rem;
        }
        .tpu-name span {
            color: #8b8b92;
            font-size: 0.78rem;
            line-height: 1.35;
        }
        .tpu-cell {
            display: grid;
            align-content: center;
            min-height: 62px;
            padding: 10px;
            text-align: right;
            transition: background 160ms ease, border-color 160ms ease;
        }
        .tpu-cell strong {
            font-variant-numeric: tabular-nums;
        }
        .tpu-cell span {
            color: rgba(255, 255, 255, 0.70);
            font-size: 0.72rem;
            margin-top: 3px;
        }
        .training-time-cell {
            display: grid;
            align-content: center;
            min-height: 56px;
            padding: 10px 12px;
            text-align: right;
        }
        .training-time-cell strong {
            font-variant-numeric: tabular-nums;
        }
        .training-time-cell span {
            color: #8b8b92;
            font-size: 0.74rem;
            margin-top: 3px;
        }
        .training-time-cell.tpu-row-label {
            text-align: left;
        }
        .tpu-cell.fits {
            background: rgba(22, 163, 74, 0.22);
            border-color: rgba(74, 222, 128, 0.45);
        }
        .tpu-cell.short {
            background: rgba(220, 38, 38, 0.20);
            border-color: rgba(248, 113, 113, 0.42);
        }
        .training-time-cell.short {
            background: rgba(220, 38, 38, 0.20);
            border-color: rgba(248, 113, 113, 0.42);
        }
        @media (max-width: 860px) {
            .overview-panel form {
                grid-template-columns: 1fr;
            }
        }
        """
    )


def _script() -> Script:
    return Script(
        """
        document.addEventListener("input", (event) => {
            const input = event.target;
            if (!(input instanceof HTMLInputElement)) return;

            if (input.type === "range") {
                const value = input.dataset.powerOfTwo === "true" ? String(2 ** Number(input.value)) : input.value;
                syncControl(input.id, value);
                updateEstimate();
                return;
            }

            if (input.type === "number" && input.id.endsWith("-number")) {
                syncControl(input.id.replace(/-number$/, ""), input.value);
                updateEstimate();
            }
        }, { capture: true });

        document.addEventListener("DOMContentLoaded", updateEstimate);

        function syncControl(baseId, value) {
            const numberInput = document.getElementById(`${baseId}-number`);
            const rangeInput = document.getElementById(baseId);
            const normalizedValue = rangeInput?.dataset.powerOfTwo === "true" ? String(nearestPowerOfTwo(Number(value))) : value;
            if (numberInput && numberInput.value !== normalizedValue) numberInput.value = normalizedValue;
            if (rangeInput) {
                const numeric = Number(normalizedValue);
                const min = Number(rangeInput.min);
                const max = Number(rangeInput.max);
                if (Number.isFinite(numeric)) {
                    const rangeValue = rangeInput.dataset.powerOfTwo === "true" ? Math.log2(nearestPowerOfTwo(numeric)) : numeric;
                    rangeInput.value = String(Math.min(max, Math.max(min, rangeValue)));
                }
            }
        }

        function nearestPowerOfTwo(value) {
            const powers = [1, 2, 4, 8, 16, 32];
            return powers.reduce((closest, power) => (
                Math.abs(power - value) < Math.abs(closest - value) ? power : closest
            ), powers[0]);
        }

        function readInt(name, fallback) {
            const input = document.querySelector(`input[name="${name}"]`);
            if (!input) return fallback;
            const value = Number.parseInt(input.value, 10);
            const min = Number(input.min || 0);
            const max = Number(input.max || Number.POSITIVE_INFINITY);
            const normalizedValue = Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback;
            return input.dataset.powerOfTwo === "true" ? nearestPowerOfTwo(normalizedValue) : normalizedValue;
        }

        function formatCount(value) {
            return Math.round(value).toLocaleString();
        }

        function formatGiB(value) {
            return `${(value / (1024 ** 3)).toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            })} GiB`;
        }

        function formatHbmGb(value) {
            return `${Math.round(value / (1024 ** 3)).toLocaleString()} GB`;
        }

        function formatFlops(value) {
            const scientific = value.toExponential(2);
            const units = [
                ["ZFLOPs", 1e21],
                ["EFLOPs", 1e18],
                ["PFLOPs", 1e15],
                ["TFLOPs", 1e12],
            ];
            const unit = units.find(([, scale]) => value >= scale);
            if (!unit) return `${Math.round(value).toLocaleString()} FLOPs (${scientific})`;
            return `${(value / unit[1]).toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            })} ${unit[0]} (${scientific})`;
        }

        function formatDuration(totalSeconds) {
            const totalMinutes = Math.round(totalSeconds / 60);
            const hours = Math.floor(totalMinutes / 60);
            const minutes = totalMinutes % 60;
            return `${hours.toLocaleString()} hr ${String(minutes).padStart(2, "0")} min`;
        }

        function setText(id, value) {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        }

        function updateTpuGrid(totalBytes, chipsPerReplica) {
            document.querySelectorAll("[data-tpu-hbm-bytes]").forEach((cell) => {
                const chipCount = Number(cell.dataset.tpuChipCount || 1);
                const perChipHbmBytes = Number(cell.dataset.tpuHbmPerChipBytes);
                const replicaHbmBytes = perChipHbmBytes * chipsPerReplica;
                const canHostReplica = chipCount >= chipsPerReplica;
                const fits = canHostReplica && replicaHbmBytes >= totalBytes;
                const value = cell.querySelector("strong");
                const note = cell.querySelector("span");
                cell.classList.toggle("fits", fits);
                cell.classList.toggle("short", !fits);
                if (value) value.textContent = canHostReplica ? formatHbmGb(replicaHbmBytes) : `needs ${chipsPerReplica} chips`;
                if (note) note.textContent = canHostReplica ? `${chipsPerReplica}-chip replica HBM` : "not enough chips";
                cell.setAttribute("aria-label", `${cell.dataset.tpuLabel}: ${fits ? "fits" : "below estimate"}`);
            });
        }

        function updateTrainingTimes(totalFlops, totalBytes, chipsPerReplica) {
            document.querySelectorAll("[data-tpu-peak-tflops]").forEach((cell) => {
                const peakFlops = Number(cell.dataset.tpuPeakTflops) * 1e12;
                const chipCount = Number(cell.dataset.tpuChipCount || 1);
                const perChipHbmBytes = Number(cell.dataset.trainingHbmPerChipBytes);
                const replicaHbmBytes = perChipHbmBytes * chipsPerReplica;
                const canHostReplica = chipCount >= chipsPerReplica;
                const fits = canHostReplica && replicaHbmBytes >= totalBytes;
                const replicas = Math.floor(chipCount / chipsPerReplica);
                const seconds = totalFlops / (peakFlops * chipCount * 0.5);
                const value = cell.querySelector("strong");
                const note = cell.querySelector("span");
                cell.classList.toggle("short", !fits);
                if (value) {
                    value.textContent = fits
                        ? formatDuration(seconds)
                        : canHostReplica
                            ? "won't fit in HBM"
                            : `needs ${chipsPerReplica} chips`;
                }
                if (note) note.textContent = fits ? `${replicas.toLocaleString()} replica${replicas === 1 ? "" : "s"}` : "insufficient HBM";
            });
        }

        function updateEstimate() {
            const batchSize = readInt("batch_size", 16);
            const sequenceLen = readInt("sequence_len", 2048);
            const vocabSize = readInt("vocab_size", 50257);
            const nLayer = readInt("n_layer", 24);
            const nHead = readInt("n_head", 12);
            const nKvHead = readInt("n_kv_head", 12);
            const nEmbd = readInt("n_embd", 1536);
            const paramBytes = readInt("param_bytes", 2);
            const activationBytes = readInt("activation_bytes", 2);
            const trainingTokens = readInt("training_tokens", 7) * 1000000000;
            const chipsPerReplica = readInt("data_parallelism", 1);

            const warnings = [];
            if (nEmbd % nHead !== 0) warnings.push("n_embd must be divisible by n_head for the model to run.");
            if (nHead % nKvHead !== 0) warnings.push("n_head must be divisible by n_kv_head for grouped-query attention.");

            const headDim = nEmbd / nHead;
            const tokenEmbeddingParams = vocabSize * nEmbd;
            const lmHeadParams = vocabSize * nEmbd;
            const qkvParams = nEmbd * ((nHead + 2 * nKvHead) * headDim);
            const attentionProjectionParams = nEmbd * nEmbd;
            const mlpParams = 8 * nEmbd * nEmbd;
            const blockParams = qkvParams + attentionProjectionParams + mlpParams;
            const totalParams = tokenEmbeddingParams + lmHeadParams + nLayer * blockParams;

            const parameterBytes = totalParams * paramBytes;
            const gradientBytes = totalParams * paramBytes;
            const adamwStateBytes = totalParams * 2 * 4;
            const hiddenActivationBytes = batchSize * sequenceLen * nEmbd * activationBytes * (nLayer + 1);
            const averageAttentionWindow = (sequenceLen + sequenceLen / 2) / 2;
            const attentionWorkspaceBytes = nLayer * batchSize * nHead * sequenceLen * averageAttentionWindow * activationBytes;
            const totalBytes = parameterBytes + gradientBytes + adamwStateBytes + hiddenActivationBytes + attentionWorkspaceBytes;
            const totalFlops = 6 * totalParams * trainingTokens;

            const warningList = document.getElementById("warnings");
            if (warningList) {
                warningList.replaceChildren(...warnings.map((warning) => {
                    const item = document.createElement("li");
                    item.textContent = warning;
                    return item;
                }));
                warningList.hidden = warnings.length === 0;
            }

            setText("total-memory", formatGiB(totalBytes));
            setText("total-params", formatCount(totalParams));
            setText("total-flops", formatFlops(totalFlops));
            setText("tokens-per-step", formatCount(batchSize * sequenceLen));
            setText("parameter-memory", formatGiB(parameterBytes));
            setText("gradient-memory", formatGiB(gradientBytes));
            setText("adamw-state", formatGiB(adamwStateBytes));
            setText("hidden-activations", formatGiB(hiddenActivationBytes));
            setText("attention-workspace", formatGiB(attentionWorkspaceBytes));
            updateTpuGrid(totalBytes, chipsPerReplica);
            updateTrainingTimes(totalFlops, totalBytes, chipsPerReplica);
        }
        """
    )


def _control(slider: Slider, value: int):
    input_id = f"scaling-{slider.name}"
    is_power_of_two = slider.name in POWER_OF_TWO_SLIDERS
    uses_billions = slider.name == "training_tokens"
    range_value = (
        str(TPU_CHIP_COUNTS.index(value))
        if is_power_of_two
        else str(min(slider.maximum, max(slider.minimum, value)))
    )
    return Div(
        Div(
            Label(slider.label, fr=input_id),
            Div(
                Input(
                    type="number",
                    id=f"{input_id}-number",
                    name=slider.name,
                    value=str(value),
                    min=str(slider.minimum),
                    max=str(slider.maximum),
                    step=str(slider.step),
                    aria_label=f"{slider.label} value",
                    cls="number-input",
                    data_power_of_two="true" if is_power_of_two else None,
                ),
                Span("B", cls="unit-label") if uses_billions else "",
                cls="number-with-unit" if uses_billions else "",
            ),
            cls="control-header",
        ),
        Input(
            type="range",
            id=input_id,
            name=False,
            value=range_value,
            min="0" if is_power_of_two else str(slider.minimum),
            max=str(len(TPU_CHIP_COUNTS) - 1)
            if is_power_of_two
            else str(slider.maximum),
            step=str(slider.step),
            data_power_of_two="true" if is_power_of_two else None,
        ),
        Div(slider.help, cls="help"),
        cls="control",
    )


def _stat(label: str, value: str, value_id: str):
    return Div(Span(label), Strong(value, id=value_id), cls="stat")


def _tpu_cell(tpu: Tpu, chip_count: int, total_bytes: float, chips_per_replica: int):
    replica_hbm_gb = tpu.hbm_gb * chips_per_replica
    replica_hbm_bytes = replica_hbm_gb * BYTES_PER_GIB
    can_host_replica = chip_count >= chips_per_replica
    fits = can_host_replica and replica_hbm_bytes >= total_bytes
    label = f"{tpu.name}, {chip_count} chip{'s' if chip_count != 1 else ''}"
    return Div(
        Strong(
            f"{replica_hbm_gb:,} GB"
            if can_host_replica
            else f"needs {chips_per_replica} chips"
        ),
        Span(
            f"{chips_per_replica}-chip replica HBM"
            if can_host_replica
            else "not enough chips"
        ),
        cls=f"tpu-cell {'fits' if fits else 'short'}",
        data_tpu_hbm_bytes=str(tpu.hbm_gb * chip_count * BYTES_PER_GIB),
        data_tpu_hbm_per_chip_bytes=str(tpu.hbm_gb * BYTES_PER_GIB),
        data_tpu_chip_count=str(chip_count),
        data_tpu_label=label,
        aria_label=f"{label}: {'fits' if fits else 'below estimate'}",
    )


def _training_time_cell(
    tpu: Tpu,
    total_flops: float,
    total_bytes: float,
    chip_count: int,
    chips_per_replica: int,
):
    replica_hbm_bytes = tpu.hbm_gb * chips_per_replica * BYTES_PER_GIB
    can_host_replica = chip_count >= chips_per_replica
    fits = can_host_replica and replica_hbm_bytes >= total_bytes
    replica_count = chip_count // chips_per_replica
    seconds = total_flops / (tpu.bf16_peak_tflops * 10**12 * chip_count * TRAINING_MFU)
    return Div(
        Strong(
            _format_duration(seconds)
            if fits
            else "won't fit in HBM"
            if can_host_replica
            else f"needs {chips_per_replica} chips"
        ),
        Span(
            f"{replica_count:,} replica{'s' if replica_count != 1 else ''}"
            if fits
            else "insufficient HBM"
        ),
        cls=f"training-time-cell {'short' if not fits else ''}",
        data_tpu_peak_tflops=str(tpu.bf16_peak_tflops),
        data_tpu_chip_count=str(chip_count),
        data_training_hbm_per_chip_bytes=str(tpu.hbm_gb * BYTES_PER_GIB),
    )


def _tpu_grid(total_bytes: float, chips_per_replica: int):
    return Section(
        H2("TPU HBM Fit Grid", cls="section-title"),
        P(
            "Uses the Data parallelism slider as chips per model replica. "
            "Each column shows whether that TPU type has enough HBM across one replica "
            "at the selected chip count; red cells either need more chips for a replica "
            "or have insufficient per-replica HBM.",
            cls="section-copy",
        ),
        Div(
            Div("TPU type", cls="tpu-header"),
            *[
                Div(f"{count} chip{'s' if count != 1 else ''}", cls="tpu-header")
                for count in TPU_CHIP_COUNTS
            ],
            *[
                item
                for tpu in TPUS
                for item in (
                    Div(
                        Strong(tpu.name),
                        Span(f"{tpu.hbm_gb} GB per chip"),
                        Span(f"Max {tpu.max_chips:,} chips. {tpu.topology}"),
                        cls="tpu-name",
                    ),
                    *[
                        _tpu_cell(tpu, count, total_bytes, chips_per_replica)
                        for count in TPU_CHIP_COUNTS
                    ],
                )
            ],
            cls="tpu-grid",
        ),
        cls="panel tpu-panel",
    )


def _training_time_grid(total_flops: float, total_bytes: float, chips_per_replica: int):
    return Section(
        H2("TPU Training Time Estimate", cls="section-title"),
        P(
            "Uses 6 x parameters x training tokens, divided by BF16 peak FLOPs. "
            "Assumes 50% MFU (mean FLOPs utilization).",
            cls="section-copy",
        ),
        Div(
            Div("TPU type", cls="tpu-header"),
            *[
                Div(f"{count} chip{'s' if count != 1 else ''}", cls="tpu-header")
                for count in TPU_CHIP_COUNTS
            ],
            *[
                item
                for tpu in TPUS
                for item in (
                    Div(
                        Strong(tpu.name),
                        Span(f"{tpu.bf16_peak_tflops:,} BF16 peak TFLOPS per chip"),
                        cls="training-time-cell tpu-row-label",
                    ),
                    *[
                        _training_time_cell(
                            tpu,
                            total_flops,
                            total_bytes,
                            count,
                            chips_per_replica,
                        )
                        for count in TPU_CHIP_COUNTS
                    ],
                )
            ],
            cls="training-time-grid",
        ),
        cls="panel tpu-panel",
    )


def _warnings(items: Iterable[str]):
    items = tuple(items)
    if not items:
        return (Ul(id="warnings", cls="warnings", hidden=True),)
    return Ul(*[Li(item) for item in items], id="warnings", cls="warnings")


def _results(values: dict[str, int], cls: str = "panel"):
    estimate = _estimate_memory(values)
    return Div(
        *_warnings(estimate["warnings"]),
        Div(
            Span("Estimated total training memory"),
            Strong(_format_gib(float(estimate["total_bytes"])), id="total-memory"),
            cls="total",
        ),
        Div(
            _stat(
                "Parameters",
                _format_count(float(estimate["total_params"])),
                "total-params",
            ),
            _stat(
                "Total FLOPs",
                _format_flops(float(estimate["total_flops"])),
                "total-flops",
            ),
            _stat(
                "Tokens per step",
                _format_count(float(estimate["tokens_per_step"])),
                "tokens-per-step",
            ),
            _stat(
                "Parameter memory",
                _format_gib(float(estimate["parameter_bytes"])),
                "parameter-memory",
            ),
            _stat(
                "Gradient memory",
                _format_gib(float(estimate["gradient_bytes"])),
                "gradient-memory",
            ),
            _stat(
                "AdamW state",
                _format_gib(float(estimate["adamw_state_bytes"])),
                "adamw-state",
            ),
            _stat(
                "Hidden activations",
                _format_gib(float(estimate["hidden_activation_bytes"])),
                "hidden-activations",
            ),
            _stat(
                "Attention workspace",
                _format_gib(float(estimate["attention_workspace_bytes"])),
                "attention-workspace",
            ),
            cls="stats",
        ),
        id="results",
        cls=cls,
    )


def _dashboard(values: dict[str, int]):
    estimate = _estimate_memory(values)
    chips_per_replica = values["data_parallelism"]
    return Div(
        Div(
            _results(values, cls="results-panel"),
            Form(
                *[_control(slider, values[slider.name]) for slider in SLIDERS],
            ),
            cls="panel overview-panel",
        ),
        _tpu_grid(float(estimate["total_bytes"]), chips_per_replica),
        _training_time_grid(
            float(estimate["total_flops"]),
            float(estimate["total_bytes"]),
            chips_per_replica,
        ),
        cls="dashboard",
    )


app, rt = fast_app(
    title="gchat Scaling",
    pico=False,
    secret_key="gchat-scaling-dashboard",
    hdrs=(
        Meta(name="viewport", content="width=device-width, initial-scale=1"),
        _favicon(),
        _style(),
        _script(),
    ),
)


@rt("/")
def index():
    values = _slider_defaults()
    return Main(
        Section(
            H1("gchat Scaling"),
            P(
                "Explore rough memory requirements for gchat model and training "
                "configuration changes. Type exact values or use the sliders to update "
                "the estimate immediately.",
                cls="subtitle",
            ),
            cls="hero",
        ),
        _dashboard(values),
    )


@rt("/estimate")
def estimate(request):
    return _results(_query_values(request))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the gchat scaling dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--reload", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    serve(
        appname="gchat.scaling.dashboard",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
