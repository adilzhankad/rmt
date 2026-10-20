# Context for Claude Code: A3 experiment (OCR benchmark agreement)

Put this file in the root of the project folder as `CLAUDE.md` (or paste it as the first message).
Also unzip `A3_data_and_code.zip` into the same folder: it contains `code/harness.py` (fixed rules + Kendall tau-b + bootstrap, already tested), `code/extract.py`, `code/audit.py`, `code/figures.py`, `data/`.

## 1. Who and what

- Course: Research Methods, Assignment 3 (A3). Group SE-2429, students Miras Dzhanin and Adilzhan Kadyrov.
- Talk to the students in simple Russian. Code, comments and report text in English.
- RQ: **To what extent do four public OCR benchmarks rank multimodal LLMs in the same order?**
- Benchmarks: OCRBench v2, CC-OCR, OmniDocBench, olmOCR-Bench.
- Design: comparative benchmark study. Unit of analysis = one model–benchmark score. Independent variable = benchmark. Dependent variable = model rank on each benchmark and pairwise Kendall tau-b between benchmarks.
- Why a re-run is needed: the published leaderboards share 0 models across all four benchmarks and no pair shares 5+ models (see `data/audit.txt`). So we run the same 8 models on all four benchmarks ourselves.

## 2. HARD RULES (grading)

1. **Never invent, estimate or "fill in" any number.** Every score must come from a real model run logged in `logs/run_log.csv`. Fabricated data = −100 points. If a run fails, log the failure; do not replace it.
2. Results must be reproducible from submitted files (code + item lists + raw predictions + logs).
3. Every link in the report must open publicly (−10 per dead link).
4. Never put API keys into files that are committed. Use environment variables / `.env` in `.gitignore`.
5. Do not send the user's email address anywhere.

## 3. Fixed study rules (already in `code/harness.py`, do not change)

| Rule | Value |
|---|---|
| SEED | 2026 |
| ITEMS_PER_BENCHMARK (main) | 400 |
| Pilot items per benchmark | 20 |
| BOOTSTRAP replicates | 1000 |
| MIN_SHARED_MODELS | 5 |
| HIGH_AGREEMENT (tau) | 0.70 |
| LOW_AGREEMENT (tau) | 0.40 |
| temperature | 0 |
| max_tokens | same value for every model (suggest 4096; OmniDocBench/olmOCR pages need long outputs) |

`harness.py` functions: `kendall_tau_b(x, y)`, `oriented(value, direction)`, `pairwise(table)`, `verdict(mean_tau, min_tau)`, `bootstrap_tau(items, bench_a, bench_b, models, reps, seed)`. Running `python code/harness.py` executes self-tests (reproduces A2 pilot taus [0.40, 0.60, 0.20, 0.80, 0.40, 0.60], matches scipy on 200 random cases). Reuse it, do not rewrite it.

## 4. Models (one fixed list, 8 models; VLMEvalKit keys)

Closed (API): `GPT4o_20240806`, `GPT4o_MINI`, `GeminiPro2-5`, `GeminiFlash2-0`, `Claude4_Sonnet`
Open (GPU): `Qwen2.5-VL-7B-Instruct`, `InternVL3-8B`, `MiniCPM-V-4_5`

- Env vars: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY` + `ANTHROPIC_BACKEND=official`.
- Open 7–8B models need a GPU with ≥16 GB (Kaggle free GPU or a local card). Qwen2.5-VL-72B is NOT used (too big) — the report must be changed to MiniCPM-V-4.5 everywhere.
- `GeminiPro2-5` default max_tokens is 65536 in VLMEvalKit config: override to the common value so all models are equal.

## 5. Tool: VLMEvalKit

- Repo: https://github.com/open-compass/VLMEvalKit, pin commit `f57c3d90154010ba176fbafc39bb79ee521fbf70`. Install: `pip install -e .` from the clone.
- Data root: env `LMUData` (default `~/LMUData`). Dataset TSVs download automatically from HuggingFace on first `build_dataset(name)`; `prepare_tsv` re-downloads on md5 mismatch. Record the md5 of every TSV used.
- Main entry for our own runner: `vlmeval.inference.infer_data_job(model, work_dir, model_name, dataset, api_nproc=4, ...)`.
  - It iterates over `dataset.data`, so **a subset = filter `dataset.data` in memory** to the saved item `index` values before calling it.
  - It writes predictions in `dataset.data` order to `get_pred_file_path(work_dir, model_name, alias, use_env_format=True)` and drops the `image` column.
  - It resumes from checkpoints (`*_checkpoint.pkl`, `*_PREV.pkl`); failed API answers contain `'Failed to obtain answer via API.'` and are retried on re-run. Count them and log them.
- Dataset names: `OCRBench_v2` (there is also `OCRBench_v2_MINI`), `CCOCR` (combined), `OmniDocBench`, `olmOCRBench`. Verify exact names in `vlmeval/dataset/__init__.py`.

### Per-benchmark details (read from the code at the pinned commit; re-verify)

**OCRBench v2** (`vlmeval/dataset/image_vqa.py`, class OCRBench_v2 ~line 4013; scorer `vlmeval/dataset/utils/ocrbrnch_v2_eval.py`)
- `evaluate` is a classmethod and reads only `eval_file` → filtering works directly.
- `process_predictions(list)` gives a per-item `score` → use for bootstrap.
- `ocrbench_v2_aggregate_accuracy` groups items into 8 English and 5 Chinese capability groups; it raises "Unknown task type!" if a type is missing → stratify the sample by category and keep ≥1 item in every category.
- Primary score: **English Overall Score** (higher = better). Chinese Overall is kept for the language expectation (ER3).

**CC-OCR** (`vlmeval/dataset/image_ccocr.py`, evaluators in `vlmeval/dataset/utils/ccocr_evaluator/`)
- Needs columns: answer, prediction, category, image_name, l2-category, split.
- Subset summary: `macro_f1_score` for multi_lan_ocr and multi_scene_ocr, `acc` for kie, `score` for doc_parsing. Track = mean of its subsets; Total = mean of the 4 tracks (higher = better).
- All four subset metrics are means of per-item values (per-item macro F1, per-file acc, per-item parsing score) → per-item scores can be pulled out for bootstrap, but you must write that extraction yourself; check it reproduces the official Total exactly.
- Sampling: 100 items per track, proportional by subset inside each track.
- Writes `_acc.csv` and `_comprehensive_eval.json`.

**OmniDocBench** (`vlmeval/dataset/OmniDocBench/omnidocbench.py`)
- `evaluate` reads ground truth from `self.data_path` (the FULL TSV) and matches predictions **by position**. For a subset you must write a subset TSV in the same order as the predictions and set `dataset.data_path` to it before evaluate. Test that this gives the same result as the full run on the same items.
- Outputs `overall_EN` / `overall_CH` = mean edit distance → **lower is better**. Use `oriented(value, 'lower')` before ranking. The report currently says "v1.5 Overall (higher better)" — this is wrong for VLMEvalKit; record which TSV version is actually downloaded (md5) and fix the report.
- Per-page/per-element scores: check whether the evaluator exposes them; if not, use subsample re-evaluation for uncertainty and say so.

**olmOCR-Bench** (`vlmeval/dataset/olmOCRBench/evaluator.py`)
- `evaluator(tsv_path, eval_file)` reads `eval_file` with `pd.read_excel` → **prediction file must be .xlsx** (set the prediction file format env var or convert).
- Overall = mean of per-category pass rates (higher = better). Writes `olmOCRBench_eval_summary.csv` next to eval_file, returns None → read the CSV.
- Per-test pass/fail exists → usable for bootstrap.
- Extra deps: fuzzysearch, rapidfuzz, beautifulsoup4.

## 6. What to build (folder `experiment/`)

```
experiment/
  config.yaml          models, datasets, seed, N, max_tokens, temperature, primary metric + direction per benchmark
  sample_items.py      stratified sample, seed 2026: items/<bench>_main.csv (400) and items/<bench>_pilot.csv (20); store index, strata, TSV md5
  run.py               --model --bench --split pilot|main ; builds model+dataset, filters items, OmniDocBench subset TSV, olmOCR xlsx,
                       infer_data_job, evaluate, saves raw predictions to raw/, appends one row to logs/run_log.csv
  collect.py           -> processed/scores_rerun.csv (long: model, benchmark, split, metric, value, direction, n_items, n_failed)
  item_scores.py       per-item scores where possible -> processed/item_scores.csv
  analyze.py           pairwise tau-b matrix, mean/min tau, verdict(), bootstrap CI, EN vs ZH comparison (ER3), figures to charts/
  tests/               dummy-model end-to-end test on tiny synthetic data shaped like each benchmark
  kaggle_notebook.ipynb  runs the 3 open models on Kaggle GPU, saves raw/ + logs/ as output
```

`logs/run_log.csv` columns: timestamp_utc, operator (Miras/Adilzhan), model, benchmark, split, n_items, n_failed, duration_s, vlmevalkit_commit, tsv_md5, max_tokens, temperature, primary_score, notes. One row per model × benchmark run, written automatically, never edited by hand.

## 7. Order of work

1. Install VLMEvalKit at the pinned commit; download the 4 TSVs; record md5.
2. Write `sample_items.py`, generate item lists, commit them.
3. Write `run.py` + dummy-model tests; prove each benchmark's subset evaluation equals the official evaluator on the same items.
4. **Pilot**: 20 items × 4 benchmarks × at least 2 models (one cheap API model, e.g. GPT4o_MINI, and one open model). The A3 report must check at least one expected result against this real pilot (Segment 4).
5. Main run: 400 items × 4 × 8 = 32 runs. Estimate API cost from the pilot first and tell the students before running.
6. `collect.py` → `analyze.py` → figures (A3 needs ≥5 figures of their own study).
7. Public GitHub repo with folders raw / processed / logs / code / charts / items + README with exact reproduce commands.

## 8. A3 report items that depend on this (Block D of the official spec)

- D.1 dataset card (4 benchmarks: source, version, md5, licence, size, our subset size)
- D.2 splits: pilot 20 / main 400 per benchmark, seed 2026, stratification rule
- D.3 system config: VLMEvalKit commit, Python/torch versions, GPU type, model keys, max_tokens, temperature
- D.4 metric formulas (tau-b; each benchmark's primary metric and direction)
- D.5 evidence: notebook link, logs of ALL runs, full-run screenshot, leakage statement (benchmarks may be in model training data — state it, we only compare rankings)
- D.6 measurements per run (from run_log.csv)

Also fix in the student doc: one consistent model list (MiniCPM-V-4.5 instead of Qwen2.5-VL-72B), ER1 range must be one value (doc says both 0.50–0.70 and 0.40–0.70), OmniDocBench metric direction.
