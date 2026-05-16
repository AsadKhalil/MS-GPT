#!/usr/bin/env python3
"""Generate thesis figures from saved trainer/evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "thesis" / "figures"


RUNS = [
    (
        "Phi-3.5-mini (15k)",
        ROOT / "MS-GPT/models/fine_tuned_llms/phi3.5_mini/final_adapter/trainer_state.json",
    ),
    (
        "Qwen2.5-3B (47k)",
        ROOT / "MS-GPT/models/fine_tuned_llms/qwen2.5_3b/final_adapter/trainer_state.json",
    ),
    (
        "Mistral-7B (31.6k)",
        ROOT
        / "MS-GPT/models/fine_tuned_llms/mistral_7b_v0.3/checkpoint-31619/trainer_state.json",
    ),
    (
        "Qwen2.5-7B (31.6k)",
        ROOT / "MS-GPT/models/fine_tuned_llms/qwen2.5_7b/checkpoint-31619/trainer_state.json",
    ),
    (
        "DeepSeek-R1-7B (5k)",
        ROOT
        / "MS-GPT/models/fine_tuned_llms/deepseek_r1_distill_7b/checkpoint-5000/trainer_state.json",
    ),
]


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

    for label, path in RUNS:
        steps, losses, lrs = read_history(path)
        axes[0].plot(steps, losses, alpha=0.12, linewidth=0.5)
        axes[0].plot(steps, rolling(losses), label=label, linewidth=1.8)
        axes[1].plot(steps, lrs, label=label, linewidth=1.8)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(bottom=1.35)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Learning rate")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "qlora_training_trace.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generation_quality() -> None:
    summary_path = ROOT / "paper_results/model_results/llms/summary_n1000.json"
    rows = json.loads(summary_path.read_text())
    names = [row["name"].replace("_", "\n") for row in rows]
    metrics = [
        ("ROUGE-L", "rougeL"),
        ("BERTScore F1", "bertscore_f1"),
        ("Faithfulness", "faithfulness_score"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = list(range(len(rows)))
    width = 0.24

    for offset, (metric_name, key) in enumerate(metrics):
        values = [float(row[key]) for row in rows]
        positions = [item + (offset - 1) * width for item in x]
        ax.bar(positions, values, width=width, label=metric_name)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Generation Quality on 1,000 Test Examples")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))

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
