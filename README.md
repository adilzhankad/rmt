# Assignment 3 — do four OCR benchmarks rank multimodal LLMs the same way?

Group SE-2429 · Miras Dzhanin, Adilzhan Kadyrov · Research Methods

RQ: *To what extent do four public OCR benchmarks rank multimodal LLMs in the same order?*
Benchmarks: OCRBench v2, CC-OCR, OmniDocBench, olmOCR-Bench.

The study has two stages.

* **Stage 1 (published leaderboards).** The published tables share no model across all
  four benchmarks and no pair shares 5 or more models (`data/audit.txt`), so agreement
  cannot be estimated from them.
* **Stage 2 (re-run).** We run one fixed list of 8 models on a stratified subset of every
  benchmark with each benchmark's own evaluator (VLMEvalKit, pinned commit), then compare
  the rankings with Kendall's τ-b.

Every score in `processed/` comes from a model run logged in `logs/run_log.csv`; the raw
model outputs are in `raw/rerun/`.

## Folders

| Path | What it is |
| --- | --- |
| `code/` | Stage 1: `extract.py`, `audit.py`, `figures.py`, and `harness.py` (fixed rules: seed, N, thresholds, Kendall τ-b, bootstrap — used by Stage 2 unchanged) |
| `data/` | Stage 1 published-score dataset, data dictionary, audit output |
| `raw/` | Stage 1 leaderboard sources (downloaded 2026-10-13) |
| `raw/rerun/<split>/<model>/<benchmark>/` | Stage 2 raw outputs: prediction file, `predictions.jsonl` (exact model answers), evaluator outputs, `metrics.json`, `run_meta.json` (versions, GPU, token usage) |
| `items/` | the item lists: `<benchmark>_main.csv` (400) and `<benchmark>_pilot.csv` (20) with strata and the TSV md5 |
| `logs/run_log.csv` | one row per model × benchmark run, written by `run.py`, never edited by hand |
| `logs/usage/usage.jsonl` | API calls and tokens per run (cost estimate) |
| `processed/` | `scores_rerun.csv`, `item_scores.csv`, τ tables, `agreement_<split>.json` |
| `charts/` | `fig1`, `fig2` (Stage 1), `fig3`–`fig8` (Stage 2) |
| `experiment/` | Stage 2 code (below) and `kaggle_notebook.ipynb` for the open models |

## Fixed settings

| Rule | Value | Where |
| --- | --- | --- |
| seed | 2026 | `code/harness.py` |
| items per benchmark | 400 main, 20 pilot (pilot disjoint from main) | `harness.py`, `experiment/config.yaml` |
| bootstrap replicates | 1000 | `harness.py` |
| minimum shared models for τ | 5 | `harness.py` |
| success criterion | mean τ-b ≥ 0.70 and min τ-b ≥ 0.40 | `harness.verdict` |
| temperature / max_tokens | 0 / 4096 for every model | `experiment/config.yaml` |
| VLMEvalKit | `f57c3d90154010ba176fbafc39bb79ee521fbf70` | `experiment/config.yaml` |

Primary metric per benchmark: OCRBench v2 *English Overall Score* (higher better), CC-OCR
*total* = mean of the 4 track scores (higher better), OmniDocBench *overall_EN* = mean
normalised edit distance on English pages (**lower better**, flipped before ranking),
olmOCR-Bench *overall* = mean per-category pass rate (higher better).

## Reproduce Stage 2

```bash
# 1. environment (Python 3.11)
git clone https://github.com/open-compass/VLMEvalKit.git third_party/VLMEvalKit
git -C third_party/VLMEvalKit checkout f57c3d90154010ba176fbafc39bb79ee521fbf70
python -m venv .venv && source .venv/bin/activate
pip install -e third_party/VLMEvalKit -r experiment/requirements-extra.txt
python -m playwright install chromium      # olmOCR-Bench renders KaTeX for its math tests
#   (macOS arm64: decord has no wheel - install the requirements without it plus eva-decord,
#    then `pip install --no-deps -e third_party/VLMEvalKit`)

# 2. data: 4 TSVs (~6 GB) into ./LMUData, md5-checked, plus the nltk corpora the scorers use
export LMUData=$PWD/LMUData
python experiment/download_data.py

# 3. item lists (already committed; re-running gives identical files)
python experiment/sample_items.py

# 4. tests: dummy model on synthetic data shaped like each benchmark
python -m pytest experiment/tests -q

# 5. runs (API keys in .env, never committed: OPENAI_API_KEY, GOOGLE_API_KEY,
#    ANTHROPIC_API_KEY, ANTHROPIC_BACKEND=official)
python experiment/run.py --model GPT4o_MINI --bench OCRBench_v2 CCOCR OmniDocBench olmOCRBench \
       --split pilot --operator Adilzhan
python experiment/estimate_cost.py          # cost of the main run from pilot token usage
#    open models: a second env, because InternVL3's remote code breaks on transformers 5 and
#    flash-attn (Qwen2.5-VL) has prebuilt wheels only up to torch 2.8:
#    experiment/setup_gpu_open_env.sh  (torch 2.8.0+cu128, transformers 4.57.6, flash-attn 2.8.3.post2)
#    then run.py with that env active; outputs from another machine are merged with
python experiment/merge_logs.py <copied output dir>

# 6. analysis
python experiment/collect.py
python experiment/item_scores.py
python experiment/analyze.py --split main
```

## Reproduce Stage 1

```bash
python -I code/extract.py raw data/scores_long.csv
python -I code/audit.py data/scores_long.csv data
python -I code/harness.py
python -I code/figures.py data/overlap.csv charts
```
