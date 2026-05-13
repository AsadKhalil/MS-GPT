"""Generate retrieval-results figures for the MS-GPT thesis.

Produces three PNGs in ``thesis/figures/``:

* ``retrieval_recall_bar.png`` -- grouped Recall@k bar chart across BM25
  and six fine-tuned dense encoders.
* ``retrieval_radar.png`` -- radar plot of the five retrieval metrics
  across the six fine-tuned dense encoders.
* ``retrieval_improvement_heatmap.png`` -- absolute gain
  (Fine-tuned -- Base) heat-map on R@10, MRR@10, NDCG@10 for the five
  dense encoders for which base numbers are recorded in the paper.

All numbers are read from the repository's result JSONs where possible.
Base-model scores are sourced from each model's
``training_summary.json`` (``metrics_history[0].score``); fine-tuned
scores come from ``eval_results_test.json``; the MiniLM entries use
``training_summary.json`` at the repo root; BM25 comes from
``paper_results/evaluation/bm25_baseline_results.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("thesis_figures")

REPO = Path(__file__).resolve().parents[1]
# paper_results, thesis, and enriched data live one level up — the
# top-level workspace. Resolve all of those off WORKSPACE, not REPO.
WORKSPACE = REPO.parent
FIG_DIR = WORKSPACE / "thesis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Models reported in the MSQA-Bench paper (E5 / BGE / Nomic).
EMB_MODELS: list[tuple[str, Path]] = [
    ("E5-base",    REPO / "models" / "fine_tuned_embeddings_e5_base_v2"),
    ("E5-large",   REPO / "models" / "fine_tuned_embeddings_e5_large_v2"),
    ("BGE-base",   REPO / "models" / "fine_tuned_embeddings_bge_base_en_v1.5"),
    ("BGE-large",  REPO / "models" / "fine_tuned_embeddings_bge_large_en_v1.5"),
    ("Nomic-v1.5", REPO / "models" / "fine_tuned_embeddings_nomic_embed_v1.5"),
]

METRICS = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
COSINE_KEYS = {
    "recall@1":  "qa_retrieval_cosine_recall@1",
    "recall@5":  "qa_retrieval_cosine_recall@5",
    "recall@10": "qa_retrieval_cosine_recall@10",
    "mrr@10":    "qa_retrieval_cosine_mrr@10",
    "ndcg@10":   "qa_retrieval_cosine_ndcg@10",
}


def _load_json(p: Path) -> dict | None:
    try:
        with p.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s (%s)", p, exc)
        return None


def collect() -> dict:
    """Return {'bm25': {m: v}, 'ft': {model: {m: v}}, 'base': {model: {m: v}}}.

    Both ``base`` and ``ft`` are read from the same post-hoc evaluator
    (``paper_results/model_results/embeddings/embedding_comparison.json``)
    so the two are directly comparable. We deliberately do not read
    base from ``training_summary.json/metrics_history[0]`` (a step-0
    in-training evaluation) because that path uses a slightly different
    filter chain and systematically underestimates by ~0.5 pp.
    """
    bm25 = _load_json(WORKSPACE / "paper_results" / "evaluation" / "bm25_baseline_results.json") or {}

    cmp_json = _load_json(
        WORKSPACE / "paper_results" / "model_results" / "embeddings" / "embedding_comparison.json"
    ) or []
    # Map embedding_comparison.json model_name -> friendly figure label.
    name_map = {
        "e5_base_v2":         "E5-base",
        "e5_large_v2":        "E5-large",
        "bge_base_en_v1.5":   "BGE-base",
        "bge_large_en_v1.5":  "BGE-large",
        "nomic_embed_v1.5":   "Nomic-v1.5",
    }

    # Canonical figure/table order (mirrors ch5_experiments.tex tab:retrieval_full).
    canonical = ["E5-base", "E5-large", "BGE-base", "BGE-large", "Nomic-v1.5"]
    by_label = {name_map.get(e.get("model_name", "")): e for e in cmp_json
                if name_map.get(e.get("model_name", ""))}

    ft: dict[str, dict[str, float]] = {}
    base: dict[str, dict[str, float]] = {}
    for label in canonical:
        entry = by_label.get(label)
        if not entry:
            continue
        base[label] = {m: float(entry["base_results"][m]) for m in METRICS
                       if m in entry.get("base_results", {})}
        ft[label]   = {m: float(entry["finetuned_results"][m]) for m in METRICS
                       if m in entry.get("finetuned_results", {})}

    return {"bm25": bm25, "ft": ft, "base": base}


def fig_recall_bar(data: dict) -> Path:
    """Recall@10 across BM25 / Base dense / Fine-tuned dense for each backbone.

    The previous version of this figure showed only BM25 + the fine-tuned
    models, which made the dense-vs-BM25 jump visible but hid the
    incremental gain from fine-tuning. The defense panel asked for the
    full BM25/Base/+FT triple, so each dense backbone now gets two bars
    (base, +FT) plus a BM25 reference group on the left.
    """
    backbones = list(data["ft"].keys())
    groups = ["BM25"] + backbones
    xs = np.arange(len(groups))
    width = 0.38

    base_vals: list[float] = []
    ft_vals: list[float] = []
    for m in groups:
        if m == "BM25":
            v = data["bm25"].get("recall@10", np.nan)
            base_vals.append(v)
            ft_vals.append(np.nan)
        else:
            base_vals.append(data.get("base", {}).get(m, {}).get("recall@10", np.nan))
            ft_vals.append(data["ft"][m].get("recall@10", np.nan))

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.bar(xs - width / 2, base_vals, width,
           label="Base / BM25", color="#4c78a8")
    ax.bar(xs + width / 2, ft_vals, width,
           label="Fine-tuned (+FT)", color="#2b8a3e")
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall@10")
    ax.set_title("Retrieval Recall@10 on MSQA-Bench (5{,}000 test queries) "
                 "-- BM25 vs Base dense vs Fine-tuned dense")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    # Annotate the absolute value above each bar so the comparison is legible.
    for i, v in enumerate(base_vals):
        if not np.isnan(v):
            ax.text(i - width / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(ft_vals):
        if not np.isnan(v):
            ax.text(i + width / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "retrieval_recall_bar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_qtype_recall_bar() -> Path | None:
    """Grouped bar chart: per-question-type Recall@10 for BM25/Base/+FT.

    Reads paper_results/diagnostics/retrieval_by_qtype.json. Buckets are
    ordered by absolute gain (BM25 -> +FT) descending so the
    ``worst-baseline-biggest-gain'' pattern reads top-down.
    """
    src = WORKSPACE / "paper_results" / "diagnostics" / "retrieval_by_qtype.json"
    blob = _load_json(src)
    if not blob:
        log.warning("skipping qtype figure: %s not found", src)
        return None

    types = [t for t in blob["bm25"] if t != "__overall__"]

    def r10(slot: str, t: str) -> float:
        return blob.get(slot, {}).get(t, {}).get("recall@10", np.nan)

    types.sort(key=lambda t: r10("ft_e5", t) - r10("bm25", t), reverse=True)

    bm = [r10("bm25", t) for t in types]
    bs = [r10("base_e5", t) for t in types]
    ft = [r10("ft_e5", t) for t in types]

    xs = np.arange(len(types))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.bar(xs - width, bm, width, label="BM25",       color="#4c78a8")
    ax.bar(xs,         bs, width, label="Base dense", color="#c8102e")
    ax.bar(xs + width, ft, width, label="+FT dense",  color="#2b8a3e")
    ax.set_xticks(xs)
    ax.set_xticklabels([t.capitalize() for t in types], rotation=0)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Recall@10")
    ax.set_title("Per-question-type Recall@10 on MSQA-Bench "
                 "(5{,}000 queries; ordered by BM25$\\to$+FT gain)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "retrieval_recall_by_qtype.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_radar(data: dict) -> Path:
    models = list(data["ft"].keys())
    labels = ["R@1", "R@5", "R@10", "MRR@10", "NDCG@10"]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756",
              "#72b7b2", "#b279a2"]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for i, m in enumerate(models):
        vals = [data["ft"][m].get(mk, 0.0) for mk in METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, label=m, color=colors[i % len(colors)], linewidth=1.8)
        ax.fill(angles, vals, color=colors[i % len(colors)], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.7, 1.0)
    ax.set_rlabel_position(10)
    ax.grid(alpha=0.4)
    ax.set_title("Fine-tuned Embedding Metrics (MSQA-Bench test)", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "retrieval_radar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_improvement_heatmap(data: dict) -> Path:
    # Only models that have both base and FT, aligned on recall/mrr/ndcg @10.
    metrics = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    mlabels = ["R@1", "R@5", "R@10", "MRR@10", "NDCG@10"]
    models = [m for m in data["ft"] if m in data["base"]]
    mat = np.full((len(models), len(metrics)), np.nan)
    for i, m in enumerate(models):
        for j, mk in enumerate(metrics):
            if mk in data["ft"][m] and mk in data["base"][m]:
                mat[i, j] = data["ft"][m][mk] - data["base"][m][mk]

    fig, ax = plt.subplots(figsize=(10.0, 0.8 + 0.7 * len(models)))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=max(0.25, np.nanmax(mat)),
                   aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(mlabels, rotation=0, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.tick_params(axis="x", pad=6)
    for i in range(len(models)):
        for j in range(len(metrics)):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", color="grey", fontsize=9)
            else:
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        color="black" if v < 0.18 else "white", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"Fine-tuned $-$ Base")
    ax.set_title("Absolute Gain from Domain Fine-Tuning")
    fig.tight_layout()
    out = FIG_DIR / "retrieval_improvement_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_publication_year_histogram() -> Path | None:
    """Per-document publication year histogram from enriched.jsonl."""
    src = WORKSPACE / "paper_results" / "dataset" / "enriched.jsonl"
    if not src.exists():
        log.warning("skipping year histogram: %s not found", src)
        return None
    doc_year: dict[str, int] = {}
    with src.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            did, yr = r.get("doc_id"), r.get("year")
            if did and isinstance(yr, int) and 1990 <= yr <= 2030 and did not in doc_year:
                doc_year[did] = yr
    if not doc_year:
        log.warning("no valid year metadata in enriched.jsonl")
        return None
    years = np.array(list(doc_year.values()))
    lo, hi = int(years.min()), int(years.max())
    bins = np.arange(lo, hi + 2) - 0.5

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(years, bins=bins, color="#4c78a8", edgecolor="white")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Documents")
    ax.set_title(f"MSQA-Bench source corpus: {len(doc_year):,} documents, "
                 f"{lo}\u2013{hi}")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(lo - 0.5, hi + 0.5)
    fig.tight_layout()
    out = FIG_DIR / "corpus_year_histogram.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    data = collect()
    log.info("FT models found:   %s", sorted(data["ft"]))
    log.info("Base models found: %s", sorted(data["base"]))
    for fn in (fig_recall_bar, fig_radar, fig_improvement_heatmap):
        out = fn(data)
        log.info("wrote %s", out.relative_to(WORKSPACE))
    hist = fig_publication_year_histogram()
    if hist is not None:
        log.info("wrote %s", hist.relative_to(WORKSPACE))
    qtype = fig_qtype_recall_bar()
    if qtype is not None:
        log.info("wrote %s", qtype.relative_to(WORKSPACE))


if __name__ == "__main__":
    main()
