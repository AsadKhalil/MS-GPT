#!/usr/bin/env python3
"""Generate thesis figures from saved trainer/evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "thesis" / "figures"


# Figure 5.5: only the two LLM runs that completed a full one-epoch pass.
# Partial / early-stopped runs (Phi-3.5-mini, Qwen2.5-3B, DeepSeek-R1-7B) are
# documented in Table 5.3 but omitted here so the trace stays apples-to-apples.
RUNS = [
    (
        "Mistral-7B-Instruct-v0.3",
        ROOT
        / "MS-GPT/models/fine_tuned_llms/mistral_7b_v0.3/checkpoint-31619/trainer_state.json",
    ),
    (
        "Qwen2.5-7B-Instruct",
        ROOT / "MS-GPT/models/fine_tuned_llms/qwen2.5_7b/checkpoint-31619/trainer_state.json",
    ),
]

RUN_COLORS = ["#1f77b4", "#d62728"]

# Models in the 1k generation slice whose adapters come from partial or
# early-stopped checkpoints — their bars are hatched in Figure 5.6.
PARTIAL_RUN_NAMES = {"phi3.5_mini", "deepseek_r1_distill_7b"}

DISPLAY_NAMES = {
    "phi3.5_mini": "Phi-3.5-mini",
    "mistral_7b_v0.3": "Mistral-7B",
    "qwen2.5_7b": "Qwen2.5-7B",
    "deepseek_r1_distill_7b": "DeepSeek-R1-7B",
}


def rolling(values: list[float], window: int = 25) -> list[float]:
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed.append(mean(values[start : idx + 1]))
    return smoothed


def read_history(path: Path) -> tuple[list[int], list[float], list[float]]:
    state = json.loads(path.read_text())
    steps: list[int] = []
    losses: list[float] = []
    lrs: list[float] = []
    for row in state.get("log_history", []):
        if {"step", "loss", "learning_rate"}.issubset(row):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            lrs.append(float(row["learning_rate"]))
    return steps, losses, lrs


def training_trace() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    for (label, path), color in zip(RUNS, RUN_COLORS):
        steps, losses, lrs = read_history(path)
        smoothed = rolling(losses)
        axes[0].plot(steps, losses, alpha=0.18, linewidth=0.5, color=color)
        axes[0].plot(steps, smoothed, label=label, linewidth=2.2, color=color)
        axes[1].plot(steps, lrs, label=label, linewidth=2.2, color=color)
        # Annotate the final smoothed loss at the right edge.
        axes[0].annotate(
            f"{smoothed[-1]:.3f}",
            xy=(steps[-1], smoothed[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color=color,
            va="center",
        )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(bottom=1.4)
    axes[0].legend(
        frameon=True,
        fontsize=10,
        loc="upper right",
        facecolor="white",
        framealpha=0.95,
        edgecolor="lightgray",
    )

    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Learning rate")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].legend(
        frameon=True,
        fontsize=10,
        loc="upper right",
        facecolor="white",
        framealpha=0.95,
        edgecolor="lightgray",
    )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "qlora_training_trace.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generation_quality() -> None:
    summary_path = (
        ROOT / "MS-GPT/paper_results/evaluation/llms_eval_1k/summary_n1000.json"
    )
    rows = json.loads(summary_path.read_text())
    labels = [DISPLAY_NAMES.get(row["name"], row["name"]) for row in rows]
    metrics = [
        ("ROUGE-L", "rougeL"),
        ("BERTScore F1", "bertscore_f1"),
        ("Faithfulness", "faithfulness_score"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = list(range(len(rows)))
    width = 0.26

    for offset, (metric_name, key) in enumerate(metrics):
        values = [float(row[key]) for row in rows]
        positions = [item + (offset - 1) * width for item in x]
        bars = ax.bar(positions, values, width=width, label=metric_name)
        # Hatch bars whose underlying adapter is partial / early-stopped.
        for bar, row in zip(bars, rows):
            if row["name"] in PARTIAL_RUN_NAMES:
                bar.set_hatch("///")
                bar.set_edgecolor("white")
        # Numeric value on top of each bar.
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)

    metric_handles, metric_labels = ax.get_legend_handles_labels()
    partial_patch = Patch(
        facecolor="lightgray",
        hatch="///",
        edgecolor="white",
        label="Partial / early-stopped run",
    )
    ax.legend(
        metric_handles + [partial_patch],
        metric_labels + ["Partial / early-stopped run"],
        frameon=True,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        facecolor="white",
        framealpha=0.95,
        edgecolor="lightgray",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "llm_generation_quality.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    training_trace()
    generation_quality()
    print(FIGURE_DIR / "qlora_training_trace.png")
    print(FIGURE_DIR / "llm_generation_quality.png")


if __name__ == "__main__":
    main()
