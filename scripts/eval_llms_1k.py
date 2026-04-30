#!/usr/bin/env python3
"""Re-run the fine-tuned LLM eval on a 1000-query test subsample.

Addresses the reviewer concern that the released eval used only 200 of
62,139 test queries. Reuses the existing
``src.llm_trainers.evaluators.evaluate_model`` and the trained adapters
in ``models/fine_tuned_llms/<name>/final_adapter``; the only difference
vs. the original eval is sample_size=1000.

Run on a CUDA box. Idempotent: skips a model whose
``eval_results_test_n1000.json`` already exists.

Usage (on GPU box):
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_llms_1k.py \\
        --config config/llm_finetuner.json \\
        --sample-size 1000 \\
        --models phi3.5_mini mistral_7b_v0.3 qwen2.5_7b deepseek_r1_distill_7b

Estimated wall time on a single 3090/4090: ~30–45 minutes per 7B model,
~15 min for Phi-3.5 (3.8B). Total ~2.5–3 h for all four.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_1k")


DEFAULT_MODELS = [
    "phi3.5_mini",
    "mistral_7b_v0.3",
    "qwen2.5_7b",
    "deepseek_r1_distill_7b",
]


def find_adapter_dir(model_name: str, models_root: Path):
    """Locate a LoRA adapter for `model_name` under `models_root`.

    Tries, in order:
      1. <root>/<name>/final_adapter
      2. <root>/<name>                (if it contains adapter_config.json)
      3. <root>/<name>/checkpoint-*   (highest-numbered)
      4. any descendant of <root>/<name> with adapter_config.json
    """
    base = models_root / model_name
    final = base / "final_adapter"
    if final.exists():
        return final
    if (base / "adapter_config.json").exists():
        return base
    if base.exists():
        candidates = sorted(
            base.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        if candidates:
            return candidates[-1]
        for sub in base.rglob("adapter_config.json"):
            return sub.parent
    return None


def parse_adapter_overrides(specs):
    out = {}
    for spec in specs or []:
        if "=" in spec:
            name, path = spec.split("=", 1)
        elif ":" in spec:
            name, path = spec.split(":", 1)
        else:
            raise ValueError(f"Invalid --adapter spec {spec!r} (expected NAME=PATH)")
        out[name.strip()] = Path(path.strip()).expanduser()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config/llm_finetuner.json"))
    p.add_argument("--test-jsonl", type=Path,
                   default=Path("paper_results/dataset/splits/test.jsonl"))
    p.add_argument("--sample-size", type=int, default=1000)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--models-root", type=Path,
                   default=Path("models/fine_tuned_llms"))
    p.add_argument("--output-root", type=Path,
                   default=Path("paper_results/model_results/llms"))
    p.add_argument("--models", nargs="*", default=None,
                   help="Model names to score (default: enabled models from config).")
    p.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH",
                   help="Per-model adapter path override. Repeatable, e.g. "
                        "--adapter qwen2.5_7b=/data/lora/qwen --adapter phi3.5_mini=/data/lora/phi.")
    p.add_argument("--no-bertscore", action="store_true")
    p.add_argument("--no-faithfulness", action="store_true")
    p.add_argument("--no-perplexity", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if eval_results_test_n<size>.json exists.")
    args = p.parse_args(argv)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        from src.llm_trainers.evaluators import evaluate_model as run_eval
        from src.llm_trainers.data_utils import LLMDataConfig
    except Exception as e:
        logger.error("Import failed (%s). Requires torch + transformers + peft + the project venv.", e)
        return 1

    # Compat shim: newer transformers renamed/removed several DynamicCache members
    # that some model archs (DeepSeek-R1-Distill) and older peft paths still read.
    #   seen_tokens          -> get_seq_length()
    #   get_max_length()     -> get_max_cache_shape()
    try:
        from transformers.cache_utils import DynamicCache
        patched = []
        if not hasattr(DynamicCache, "seen_tokens"):
            def _seen_tokens(self):
                if hasattr(self, "get_seq_length"):
                    try:
                        return self.get_seq_length()
                    except Exception:
                        return 0
                return 0
            DynamicCache.seen_tokens = property(_seen_tokens)
            patched.append("seen_tokens")
        if not hasattr(DynamicCache, "get_max_length"):
            def _get_max_length(self):
                fn = getattr(self, "get_max_cache_shape", None)
                if fn is not None:
                    try:
                        return fn()
                    except Exception:
                        return None
                return None
            DynamicCache.get_max_length = _get_max_length
            patched.append("get_max_length")
        if patched:
            logger.info("Patched DynamicCache compat: %s", ", ".join(patched))
    except Exception as e:
        logger.warning("DynamicCache compat shim skipped: %s", e)

    if not args.test_jsonl.exists():
        logger.error("Test JSONL not found: %s", args.test_jsonl)
        return 1

    try:
        adapter_overrides = parse_adapter_overrides(args.adapter)
    except ValueError as e:
        logger.error("%s", e)
        return 1

    config = json.loads(args.config.read_text())
    if args.models:
        wanted = set(args.models)
        models = [m for m in config.get("models_to_train", []) if m["name"] in wanted]
    else:
        models = [m for m in config.get("models_to_train", []) if m.get("enabled")]
    if not models:
        logger.error("No models to score. Pass --models explicitly or enable some in the config.")
        return 1

    logger.info("Sample size: %d  Models: %s", args.sample_size, [m["name"] for m in models])

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    data_cfg = LLMDataConfig()

    summary = []
    for entry in models:
        name = entry["name"]
        out_dir = args.output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"eval_results_test_n{args.sample_size}.json"
        if out_file.exists() and not args.force:
            logger.info("[%s] already done (%s) — skipping. Use --force to re-run.", name, out_file)
            with out_file.open() as fh:
                d = json.load(fh)
            summary.append({"name": name, "n": d.get("num_samples"), "rougeL": d.get("rougeL"),
                            "bertscore_f1": d.get("bertscore_f1"),
                            "faithfulness_score": d.get("faithfulness_score")})
            continue

        if name in adapter_overrides:
            adapter_dir = adapter_overrides[name]
            if not adapter_dir.exists():
                logger.warning("[%s] override path does not exist: %s — skipping.", name, adapter_dir)
                continue
        else:
            adapter_dir = find_adapter_dir(name, args.models_root)
        if adapter_dir is None:
            logger.warning("[%s] no adapter found under %s (override with --adapter %s=PATH) — skipping.",
                           name, args.models_root, name)
            continue

        base_model_id = entry["model_path"]
        logger.info("[%s] base=%s adapter=%s", name, base_model_id, adapter_dir)

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(base_model_id,
                                            trust_remote_code=entry.get("trust_remote_code", False))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=entry.get("trust_remote_code", False),
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model.eval()
        logger.info("[%s] load took %.1fs", name, time.time() - t0)

        result = run_eval(
            model=model,
            tokenizer=tok,
            jsonl_path=str(args.test_jsonl),
            model_name=name,
            split="test",
            sample_size=args.sample_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            config=data_cfg,
            output_dir=str(out_dir),
            compute_perplexity_flag=not args.no_perplexity,
            compute_faithfulness_flag=not args.no_faithfulness,
            compute_bertscore_flag=not args.no_bertscore,
        )
        # The inner function writes eval_results_test.json — rename to *_n<size>.
        default_out = out_dir / "eval_results_test.json"
        if default_out.exists():
            default_out.replace(out_file)
        default_preds = out_dir / "predictions_test.json"
        if default_preds.exists():
            (out_dir / f"predictions_test_n{args.sample_size}.json").write_text(
                default_preds.read_text())
            default_preds.unlink()

        run_dt = time.time() - t0
        logger.info("[%s] DONE in %.1fs  ROUGE-L=%.3f BERT-F1=%.3f Faith=%.3f",
                    name, run_dt, result.rougeL, result.bertscore_f1, result.faithfulness_score)
        summary.append({
            "name": name, "n": result.num_samples,
            "rougeL": result.rougeL, "bertscore_f1": result.bertscore_f1,
            "faithfulness_score": result.faithfulness_score,
            "perplexity": result.perplexity, "wall_time_s": run_dt,
        })
        del model, base, tok
        torch.cuda.empty_cache()

    summary_path = args.output_root / f"summary_n{args.sample_size}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", summary_path)
    print("\n=== Summary ===")
    for s in summary:
        print(f"  {s.get('name'):28s} n={s.get('n'):>5}  rougeL={s.get('rougeL', 0):.3f}  "
              f"bert={s.get('bertscore_f1', 0):.3f}  faith={s.get('faithfulness_score', 0):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
