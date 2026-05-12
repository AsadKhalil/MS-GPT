# Repository Guidelines

## Project Structure & Module Organization

MS-GPT is a Python pipeline for turning mass spectrometry PDFs into extracted text, Q&A datasets, and fine-tuned models. Core code lives in `src/`: `pdf_processors/` handles PDF conversion, `vision_extractors/` handles OCR/vision extraction, `qa_generators/` builds Q&A data, and `embedding_trainers/` plus `llm_trainers/` run training. Entry points are in `scripts/`, JSON settings in `config/`, examples in `examples/`, and docs in `docs/`. Data and generated artifacts are under `data/`, `models/`, `figures_legacy/`, and `paper_results/` (including `paper_results/evaluation/llms_eval_1k/`) inside this repo, and under `paper_results/`, `submitted_version/` (the submitted NeurIPS 2026 paper), `archive/paper-v2-prep-draft/`, `archive/paper-v1-original/`, `thesis/`, `proposals/`, and `github_release/` in the parent workspace; avoid committing large regenerated outputs unless required. The public benchmark archive is published on Zenodo (concept DOI `10.5281/zenodo.19831805`); regenerate it with `scripts/prepare_neurips_release.py`.

## Build, Test, and Development Commands

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python -m pytest src/vision_extractors/test_vision_extractor.py
python scripts/validate_embedding_setup.py
python src/qa_generators/qa_generator.py --config config/qa_generator.json
python scripts/train_all_models.py --config config/embedding_finetuner.json -y
```

Use the first two commands to enter the local environment and install dependencies. Run `pytest` before submitting changes; use the narrower command while iterating on vision extraction. Q&A generation and training may require local services, CUDA, and substantial disk/GPU resources.

## Coding Style & Naming Conventions

Use Python 3.10+ style with 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and descriptive JSON config keys. Keep modules focused on existing pipeline stages. Development tools in `requirements.txt` include `black`, `flake8`, and `mypy`; run them on changed Python files when practical, for example `black src scripts` and `flake8 src scripts`.

## Testing Guidelines

Tests use `pytest`. Existing tests are colocated with modules, such as `src/vision_extractors/test_vision_extractor.py`; name new tests `test_*.py` and keep fixtures small. Prefer tests for parsing, file handling, config loading, and failure paths without external LLM services. For GPU or service-backed workflows, add a smoke test or document the manual command used.

## Commit & Pull Request Guidelines

Recent history uses short, lower-case messages such as `fix` and `benchmarking dataset`; improve on that with concise imperative subjects, for example `fix qa consolidation ordering` or `add embedding setup validation`. PRs should describe the pipeline stage affected, list commands run, note required services or GPUs, and call out any generated data/model artifacts intentionally included.

## Security & Configuration Tips

Do not hard-code API keys, private paths, or credentials in source files or configs. Prefer environment variables or local-only config overrides. Keep `data/input/`, logs, and model outputs free of sensitive or licensed material unless they are explicitly approved for repository use.

## Paper, Thesis, and Proposal Builds

LaTeX sources live in `submitted_version/` (the actual NeurIPS 2026 submitted paper), `archive/paper-v2-prep-draft/latex/` (preparation draft, superseded), `thesis/latex/`, and `proposals/`. Compile the Innopolis PhD proposal with `cd proposals && xelatex -interaction=nonstopmode 2026-05-05-innopolis-phd-proposal.tex` (run twice for cross-references). XeLaTeX is required because the proposal contains a Russian abstract (via `polyglossia`); the main font defaults to Times New Roman, which has Cyrillic coverage on macOS. Submission diagnostics for the NeurIPS package are run with `python scripts/submission_diagnostics.py`. Post-experiment notes and follow-ups are tracked in `EXPERIMENT_NOTES.md`.
