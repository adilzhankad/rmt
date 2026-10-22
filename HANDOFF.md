# HANDOFF: A3 Stage 2 re-run (for a teammate's Claude Code)

Written 2026-10-22 ~11:20 UTC by Adilzhan's Claude Code session. It is meant for **Miras's Claude
Code**: read this file first, then `CLAUDE_CODE_CONTEXT.md` (the original brief) and `README.md`
(reproduce commands). This file says what has happened since the brief, what is running right now,
which accesses you need and how to get them, and what is left.

**No secrets are in this file on purpose.** The repository is public, and the hard rules forbid
keys in committed files. Section 4 lists every secret, where it lives and how to get it privately.

---

## 0. Talk / work rules (unchanged, from the brief and from the students)

- Talk to the students in **simple Russian**; code, comments, commit messages and report text in **English**.
- **Never invent, estimate or "fill in" any number.** Every score comes from a real run logged in
  `logs/run_log.csv`, which is written only by `experiment/run.py` / `experiment/queue_runs.py`
  (via `common.append_run_log`). Never edit that file by hand to add or change a score.
  Fabricated data = −100 points. A failed run is logged as a failure, not replaced.
- Results must be reproducible from the repo (code + item lists + raw predictions + logs).
- Every link in the report must open publicly.
- No API keys in committed files (`.env` is in `.gitignore`).
- Do not send the students' e-mail addresses anywhere.
- `code/harness.py` must not be changed.
- Spending money (GPU machines, API calls) or anything irreversible needs the student's explicit
  "yes" first. Telegram messages to the bot are questions, not approvals.
- No "Co-Authored-By: Claude" / "Generated with Claude" lines in commits (the students' rule).

---

## 1. The study in one paragraph

RQ: do four OCR benchmarks (OCRBench v2, CC-OCR, OmniDocBench, olmOCR-Bench) rank multimodal LLMs in
the same order? Published leaderboards share too few models, so we re-run one fixed model list on
all four benchmarks ourselves: VLMEvalKit pinned at `f57c3d90154010ba176fbafc39bb79ee521fbf70`,
400 stratified items per benchmark (seed 2026, `items/*_main.csv`), temperature 0, max_tokens 4096,
every model gets the benchmark's own prompt (`prompt: benchmark` in `experiment/config.yaml`).
Then Kendall tau-b between benchmark rankings (`code/harness.py`), bootstrap CIs, ≥5 figures, and a
comparison of our ranking with the published leaderboards (`experiment/compare_published.py`).

---

## 2. Model set (final, 6 models; decided by the students on 2026-10-21 for the budget)

| Key (VLMEvalKit) | Kind | Env | Engine |
|---|---|---|---|
| `GPT4o_20240806` | API (OpenAI) | `.venv` | API |
| `GPT4o_MINI` | API (OpenAI) | `.venv` | API |
| `Qwen2.5-VL-7B-Instruct` | open | `.venv-vllm` | **VLMEvalKit's vLLM backend** (`use_vllm: true`) |
| `InternVL3-8B` | open | `.venv-open` | transformers 4.57 |
| `Ovis2-8B` | open | `.venv-legacy` | transformers 4.46.2 |
| `llava-onevision-qwen2-7b-ov-hf` | open | `.venv-open` | transformers 4.57, `generate_overrides` force max_new_tokens 4096 + greedy |

Dropped, with the reasons written in `config.yaml` comments and the attempts in `logs/run_log.csv`:
- Gemini 2.5 Pro / 2.0 Flash / Claude 4 Sonnet: retired models; their successors were piloted but
  need paid tiers we cannot fund (Gemini free tier is 20 requests/day; no Anthropic credits).
- Idefics3-8B and Phi-3.5-Vision: ~5 min per OmniDocBench page, too slow.
- MiniCPM-V-4.5 (piloted fine, but slowest, ~21 GPU-h) and DeepSeek-VL2-small (16B, cannot share a GPU): dropped for the GPU budget.
- LLaVA-OneVision went straight to the main run without the 20-item pilot (the students' call).

All 6 are on the published OCRBench v2 board (`data/scores_long.csv`).

`MIN_SHARED_MODELS` in harness is 5, so the verdict needs at least 5 models with all 4 benchmarks.

---

## 3. State right now (2026-10-22 ~11:20 UTC)

### 3.1 Main run: 10 of 24 jobs had a score before today
Scores from `logs/run_log.csv` (main split, 400 items each, 0 failed answers):

| Model | OCRBench v2 (EN, ↑) | CC-OCR (↑) | olmOCR (↑) | OmniDocBench (EN edit dist, ↓) |
|---|---|---|---|---|
| GPT-4o | 0.460 | 0.735 | 72.4 | 0.209 |
| GPT-4o-mini | 0.413 | 0.652 | 60.7 | 0.375 |
| InternVL3-8B | 0.465 | ABORTED 338/400 (resumes) | ABORTED 135/400 (resumes) | — |
| Qwen2.5-VL-7B | 0.445 on transformers, **superseded** (see 3.3) | — | — | — |
| Ovis2-8B | — | — | — | — |
| LLaVA-OV-7B | — | — | — | — |

So 15 GPU jobs are left (Qwen ×4 incl. the OCRBench redo, InternVL ×3, Ovis2 ×4, LLaVA ×4). API runs
are complete: the OpenAI key is no longer needed.

### 3.2 A GPU machine is RUNNING and billing
- Brev instance **`a3-main`**, type `massedcompute_A6000_plusx2` (2× RTX A6000 48 GB, 30 vCPU
  Xeon Gold 6238R 2.2 GHz, 125 GB RAM, 503 GB disk), **$1.37/h**, created 2026-10-22 ~11:00 UTC.
  It cannot be stopped, only deleted.
- Started with `experiment/brev_bootstrap.sh a3-main 1.37 29`: **cap $29** for this machine. The
  budget guard stops the queue at **2026-10-23 07:53 UTC** at the latest, uploads everything to
  Drive and deletes the instance itself.
- Expected: ~13–15 h, ~$18–20, if vLLM Qwen works. Setup finished at 11:20:45Z (all 4 envs,
  data, weights). The queue started at 11:20:47Z with 15 GPU jobs on 2 GPUs: Qwen OCRBench v2 on
  GPU 1 (vLLM, model loaded and compiling at 11:22Z), Ovis2 and LLaVA OCRBench v2 on GPU 0 (both
  answering). Qwen-on-vLLM works: OCRBench v2 scored at 11:55Z, then Qwen CC-OCR started on GPU 1.
- Scheduling (`experiment/queue_runs.py`): every job is pinned to one GPU
  (`CUDA_VISIBLE_DEVICES`). Since 12:15Z: GPU 1 runs the vLLM (Qwen) job with half the card
  (`gpu_utils: 0.5`) plus one transformers job, and keeps room for the next Qwen job. GPU 0 runs
  two transformers jobs (~18 GB each). When no Qwen job is left, GPU 1 takes two transformers jobs.
  A new job never starts on a GPU within 3 min of the previous start there. (Until 12:15Z vLLM
  took the whole GPU 1; Qwen CC-OCR was stopped then and resumed, logged as ABORTED.) Order is benchmark-major: OCRBench v2 → CC-OCR → olmOCR → OmniDocBench.
- Telegram bot (@clde_alert_bot) posts ▶️ start / ✅ score / ❌ failure / ☁️ upload / 🏁 end
  messages to Adilzhan's chat.

### 3.3 Why Qwen's OCRBench v2 is run again
Qwen on transformers was ~1.4× slower than InternVL and the GPU budget did not cover it, so Qwen
moved to VLMEvalKit's own vLLM path (same messages and image resizing; greedy, since the
transformers path uses top_k=1). OCRBench v2 is redone on vLLM so all four benchmarks of the model
use one engine. The transformers run (0.445) stays in `logs/run_log.csv` and its outputs are in
`raw/superseded/main/Qwen2.5-VL-7B-Instruct/OCRBench_v2/` (see `raw/superseded/README.md`).
`collect.py` takes the last scored row, so the vLLM row wins once it exists.
Result (2026-10-22 11:55Z): vLLM English Overall **0.457** (0 failed, 35 min) vs transformers 0.445;
256 of 400 answers are identical as text, 144 differ (different kernels/numerics in bf16). **For the report:**
state this, and give both OCRBench numbers as an engine sanity check. `run.py` adds
`engine vLLM <version>` to the run's notes and the vLLM version to `run_meta.json`.

### 3.4 Money spent so far
- Brev: ≈ $7.2 before today (pilot on L40S ≈ $4.0; first A6000 ≈ $0.4, lost by a guard race; second
  A6000 2026-10-21 13:13–17:20 UTC ≈ $2.8), plus today's machine (≤ $29). Brev balance was ≈ $43
  before today, from $50.
- OpenAI: main runs finished, ≈ $21 (estimate from usage; check the OpenAI billing page for the real number).

### 3.5 Pilot (20 items, done 2026-10-20 on an L40S)
All pilot rows are in `logs/run_log.csv` (split `pilot`), outputs in `raw/rerun/pilot/`, consoles
in `pilot_console_logs/`. Six models completed all 4 benchmarks: GPT-4o, GPT-4o-mini,
InternVL3-8B, Qwen2.5-VL-7B, MiniCPM-V-4.5, Ovis2-8B. The pilot analysis is run by `analyze.py --split pilot`.

---

## 4. Accesses you need and how to get them

| What | Needed for | How to get it (privately, never through git) |
|---|---|---|
| **GitHub** `adilzhankad/rmt` (public) | push commits | Adilzhan adds Miras as a collaborator (repo → Settings → Collaborators). Reading needs nothing. |
| **NVIDIA Brev** org `akomen1000-2502b9-w1hx` | watch / stop / delete `a3-main`, start a new machine | Adilzhan invites Miras into the org at brev.nvidia.com (Org → Members), then Miras runs `brev login`, `brev ls`, `brev refresh` (writes the `a3-main` ssh entry). **Do not copy Adilzhan's `~/.brev/credentials.json` to another laptop.** A copy is on the running machine only so the guard can delete it. |
| **Google Drive** teacher's folder | the mirror the teacher sees | Folder https://drive.google.com/drive/folders/1kLvWzxFfnU7-mleWUjN8shY53PlOu3gG. Ask Adilzhan/the teacher to share it with Miras's Google account, then `rclone config` → new remote **`a3drive`**, type drive, scope `drive`, `root_folder_id = 1kLvWzxFfnU7-mleWUjN8shY53PlOu3gG`, own Google login. The `token` in Adilzhan's rclone.conf is his Google login, do not share it. The rclone shared client_id is being retired in 2026; if uploads start failing, make your own client_id (rclone.org/drive/#making-your-own-client-id). |
| **`.env`** in the repo root (gitignored) | Telegram messages, LMUData path | Keys: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_BACKEND`, `LMUData`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Get the values from Adilzhan privately. The OpenAI key is **not needed any more** (API runs are done) and should be rotated: it was pasted in a chat once. Leave `OPENAI_API_KEY` empty in your copy. `TELEGRAM_CHAT_ID` is Adilzhan's chat: for messages to Miras, create your own bot with @BotFather and use your own chat id. |
| **SSH to the GPU machine** | logs, restarts | Through Brev only (`ssh a3-main` after `brev refresh`). User `shadeform`, repo at `~/a3`. |

When the work is done: Adilzhan revokes the rclone Google access (Google account → Security →
third-party access) and rotates the OpenAI key.

---

## 5. Repository map (what matters)

```
CLAUDE_CODE_CONTEXT.md   original brief (rules, benchmarks, per-benchmark evaluator notes)
HANDOFF.md               this file
README.md                reproduce commands
code/harness.py          fixed rules + kendall_tau_b + bootstrap  (DO NOT CHANGE)
data/                    Stage 1: published leaderboards (scores_long.csv), audit, overlap
items/                   item samples: <bench>_main.csv (400), <bench>_pilot.csv (20), sampling_summary.json
experiment/
  config.yaml            models (kind, env, overrides), benchmarks (md5, metric, direction), seed, N
  common.py              paths, run_log writer (flock), .env loader, key redaction, nltk length-guard lift
  run.py                 one model x benchmark x split -> raw/rerun/<split>/<model>/<bench>/ + one run_log row
  queue_runs.py          GPU/API job queue (skips done jobs, adopts running ones, retries once, multi-GPU)
  budget_guard.sh        USD cap, Drive upload every 30 min, post_run analysis, self-delete
  brev_bootstrap.sh      whole Brev run on a fresh instance (guard + 4 envs + data + weights + queue)
  setup_gpu_main_env.sh / setup_gpu_open_env.sh / setup_gpu_legacy_env.sh / setup_gpu_vllm_env.sh
  post_run.sh            collect -> item_scores -> analyze main+pilot -> compare_published
  collect.py             run_log + metrics.json -> processed/scores_rerun.csv, missing_runs.csv
  item_scores.py         per-item scores (checked to rebuild the official aggregate) -> processed/item_scores.csv
  analyze.py             tau matrix, verdict, bootstrap CIs, language comparison, figures -> charts/
  compare_published.py   our ranks vs leaderboard ranks -> processed/published_vs_ours.*
  figures_rerun.py, estimate_cost.py, merge_logs.py, download_data.py, sample_items.py
  tests/                 20 pytest tests on synthetic data (python -m pytest experiment/tests -q)
logs/run_log.csv         THE source of every number (one row per run attempt, failures included)
logs/usage/              API token usage per run
raw/rerun/<split>/...    predictions, evaluator outputs, metrics.json, run_meta.json per run
raw/superseded/          complete runs replaced by a later one (kept, see its README)
console_logs/            stdout of every main-run job;  pilot_console_logs/ for the pilot
processed/, charts/      analysis outputs (currently PARTIAL drafts from 10/24 jobs; regenerate)
third_party/VLMEvalKit   clone at the pinned commit (gitignored)
```

Envs (Linux GPU machine; on a Mac only `.venv` exists and no datasets are local):
- `.venv`: main, transformers 5.x, API models, all evaluators, analysis.
- `.venv-open`: torch 2.8.0+cu128, transformers 4.57, flash-attn 2.8.3.post2, `datasets>=3`.
- `.venv-legacy`: transformers 4.46.2 (Ovis2), xformers 0.0.32.post2.
- `.venv-vllm`: vLLM 0.10.2 on torch 2.8.0, transformers ≥4.55.2,<5, qwen-vl-utils.

---

## 6. Watching the live run

```bash
brev ls                                   # a3-main RUNNING?  "No instances" = it was deleted
ssh a3-main 'cd ~/a3; cat bootstrap.log; tail -20 queue.log; tail -5 guard.log'
ssh a3-main 'cd ~/a3; nvidia-smi; pgrep -af "[r]un.py" | grep -o "model [^ ]* --bench [^ ]*"'
ssh a3-main 'cd ~/a3; tail -c 2000 console_logs/main_<model>_<bench>.txt'   # one job's progress
ssh a3-main 'cd ~/a3; .venv/bin/python experiment/queue_runs.py --dry-run'   # jobs still to do
ssh a3-main 'cd ~/a3; tail -5 logs/run_log.csv'
```
Setup logs on the machine: `setup_main.log`, `setup_open.log`, `setup_legacy.log`,
`setup_vllm.log`, `weights.log`. VLMEvalKit saves partial answers every 10 items as
`raw/rerun/main/<model>/<bench>/01_<bench>.pkl` and resumes from it.

Control files in `~/a3` on the machine:
- `HOLD`: stops the guard's 60-min "idle" auto-end while you restart things by hand.
- `STOP_REASON`: text written into the ABORTED rows when the queue is stopped.
- `QUEUE_DONE`: the queue finished everything → the guard runs `post_run.sh`, uploads, deletes the machine.
- `QUEUE_STOPPED`: the queue was stopped (no delete by itself, except when the cap was reached).
- `QUEUE_ORDER`: optional, one model per line → model-by-model order. Set at 12:05Z on 2026-10-22
  to InternVL3 → Qwen → Ovis2 → LLaVA: LLaVA-OV falls into repetition loops up to the 4096-token
  limit on full-page OCR (~6 min per such item at ~11 tok/s), so its long benchmarks go last and
  the other models finish inside the cap first. Its running OCRBench v2 (168/400 after 56 min) was
  paused at 12:19Z (ABORTED row, partial answers kept); it resumes after the other models.
- `ALLOW_API`: API jobs run only if it exists. Not needed now.
- `BOOTSTRAP_RUNNING`: setup in progress.
- `experiment/queue_watchdog.sh` (started by hand at 12:45Z on 2026-10-22, log `watchdog.log`):
  every 5 min, restarts the queue if it is not running and neither `QUEUE_DONE` nor
  `QUEUE_STOPPED` nor `HOLD` exists. **Touch `HOLD` before stopping the queue by hand**, or it comes back.
- Decision 2026-10-22 (Adilzhan): LLaVA keeps running after the other models until the $29 cap.

### If Qwen on vLLM fails
The queue retries a failed job once, then moves on (each failure = an ERROR row, which is correct
and stays). Read `console_logs/main_Qwen2.5-VL-7B-Instruct_OCRBench_v2.txt`. Likely suspects: vLLM
startup memory check (it wants 90% of GPU 1 free), `VLLM_WORKER_MULTIPROC_METHOD=spawn` (set by the
queue), a dependency conflict in `.venv-vllm`. If it cannot be fixed quickly, the fallback the
students already accepted is Qwen on transformers in `.venv-open`. Then set Qwen back to `env: open`
without `use_vllm` in `config.yaml`, move `raw/superseded/main/Qwen2.5-VL-7B-Instruct/OCRBench_v2`
back to `raw/rerun/main/...` (its transformers score then counts again) and restart the queue. Ask
the students first if the estimate goes over the cap.

### Stopping or restarting the queue by hand (safe way)
```bash
ssh a3-main 'cd ~/a3; touch HOLD; echo "operator restarted the queue: <why>" > STOP_REASON'
ssh a3-main 'cd ~/a3; pkill -TERM -f "[q]ueue_runs.py"'        # running jobs -> ABORTED rows, partial answers kept
ssh a3-main 'cd ~/a3; rm -f QUEUE_STOPPED STOP_REASON; setsid nohup .venv/bin/python experiment/queue_runs.py >> queue.log 2>&1 < /dev/null & disown'
ssh a3-main 'cd ~/a3; rm -f HOLD'
```
- **Gotcha:** never put a `pkill -f` pattern's plain text into the same ssh command as other work.
  `pkill -f` matches the remote shell's own command line and kills it. Use the `[q]ueue` bracket
  trick and separate ssh calls.
- **Gotcha:** never edit a running bash script (`budget_guard.sh`) in place. Bash reads it
  lazily. Copy a new file next to it, `mv` it over, then restart the guard.
- The queue adopts `run.py` processes that are still running, so a queue restart does not kill jobs
  unless you send TERM to the queue itself.
- `queue_runs.py` writes `--operator Adilzhan` into every row. If Miras runs it, change that
  string (allowed values: `Miras`, `Adilzhan`), because the operator column must be true.

### Changing the cap
The cap is an argument of the guard process. To change it: copy the guard (see the gotcha above),
`pkill -f "[b]udget_guard.sh a3"` in one ssh call, and in another start
`setsid nohup experiment/budget_guard.sh a3-main 1.37 <new cap> >> guard.log 2>&1 < /dev/null & disown`.
The cap counts from the machine's boot (`/proc/uptime`). **Only with the students' "yes".**

### Deleting the machine by hand
`brev delete a3-main`. Before that, make sure the outputs are on Drive (the guard uploads every
30 min; the `☁️` Telegram message) or rsync them first:
```bash
rsync -az --exclude "*_subset_gt.tsv" --exclude "*.md5.json" a3-main:a3/logs a3-main:a3/console_logs a3-main:a3/processed a3-main:a3/charts ./
rsync -az --exclude "*_subset_gt.tsv" a3-main:a3/raw/rerun/main/ raw/rerun/main/
```
Partial answers (`*.pkl`) go to Drive under `resume_cache/raw/rerun/...`. They are gitignored.

---

## 7. Resuming on a NEW machine (if this one dies or hits the cap)

1. `brev create a3-main --type massedcompute_A6000_plusx2` (or `massedcompute_A6000_plus`, 1 GPU,
   $0.68/h). Wait for READY, then `brev refresh`.
2. From the repo root on the laptop, with the partial answers back in place (from Drive
   `resume_cache/raw/rerun/main/<model>/<bench>/01_<bench>.pkl` → `raw/rerun/main/<model>/<bench>/`):
   `rsync -az --exclude .venv/ --exclude third_party/ --exclude LMUData/ --exclude work/ --exclude hf/ --exclude __pycache__/ --exclude .DS_Store --exclude '*.zip' ./ a3-main:a3/`
3. On the machine: install Linux `rclone` (downloads.rclone.org) and the `brev` CLI
   (github.com/brevdev/brev-cli releases) into `~/.local/bin`. Copy `rclone.conf` and the Brev
   credentials there (the guard needs them to upload and to delete the machine).
4. `chmod +x experiment/*.sh; setsid nohup experiment/brev_bootstrap.sh a3-main <usd/h> <cap> > bootstrap.log 2>&1 < /dev/null & disown`
5. The queue skips every job that has a score and its `metrics.json`, and resumes partial ones.
6. First merge `logs/run_log.csv` from Drive if the machine died before you rsynced. Drive has
   the last upload. `experiment/merge_logs.py` exists for merging output directories.

---

## 8. After the queue finishes (the guard runs this on the machine; redo it on the laptop)

1. Pull results: rsync from the machine or `rclone copy a3drive:logs logs` etc. Then
   `git add logs raw/rerun raw/superseded console_logs processed charts && git commit && git push`.
2. On the laptop: `source .venv/bin/activate; export LMUData=...`; `python experiment/collect.py;
   python experiment/item_scores.py; python experiment/analyze.py --split main; python
   experiment/analyze.py --split pilot; python experiment/compare_published.py --split main`.
   `item_scores.py` needs the TSVs (LMUData). They are not on the Mac: either run it on the GPU
   machine (post_run.sh does) or `python experiment/download_data.py` first (~10 GB).
3. **Redraw the figures on the Mac:** the Linux machine lacks the Times font, so the charts made
   there are drafts.
4. Check `processed/missing_runs.csv`: every model × benchmark without a score must be explained
   in the report, not filled.
5. Mirror the final repo state to Drive (`rclone copy <dir> a3drive:<dir>` for logs, raw/rerun,
   raw/superseded, console_logs, processed, charts, experiment, items, README.md, HANDOFF.md).
   Exclude `*.pkl` and `*_subset_gt.tsv`.

---

## 9. Report (Block D) — notes collected so far

- **D.1 dataset card:** 4 benchmarks, md5 of each TSV in `config.yaml`; OmniDocBench in VLMEvalKit
  is the **v1.0** TSV and its metric is edit distance, **lower is better** (the student doc said
  "v1.5, higher better": wrong, fix it). olmOCR predictions must be .xlsx.
- **D.2 splits:** pilot 20 / main 400 per benchmark, seed 2026, stratified (`items/sampling_summary.json`).
- **D.3 system config:** VLMEvalKit commit; per-model env and transformers version (see the table
  in section 2; exact versions are in each `run_meta.json`); vLLM 0.10.2 for Qwen; GPUs: pilot on
  L40S, main on RTX A6000 (one machine with 1 GPU, then one with 2). max_tokens 4096,
  temperature 0, benchmark prompt for every model.
- **D.4 metrics:** tau-b via harness; primary metric + direction per benchmark in `config.yaml`.
- **D.5 evidence:** all runs incl. failures in `logs/run_log.csv`; console logs; leakage statement.
  Explain the log rows that are not plain successes:
  - nltk ≥3.10.3 refuses `edit_distance` on strings > 2000 chars, which crashed the OCRBench v2
    evaluator → 3 ERROR rows. The guard is lifted in `common.py` (via `importlib`, because nltk
    rebinds `nltk.metrics`).
  - ABORTED rows: operator restarts of the queue, the loss of the first A6000 instance (12:53:59Z),
    and the budget pause on 2026-10-21 ~17:15Z.
  - Pilot ERROR/ABORTED rows: Idefics3/Phi-3.5 too slow; transformers-version incompatibilities
    before the separate envs existed; Gemini quota.
  - Model drops and reasons (section 2); LLaVA skipped the pilot; Qwen engine switch (section 3.3).
- **D.6 measurements:** from `run_log.csv` (duration, n_items, n_failed, score per run).
- **Published vs ours:** compare **ranks only**: the leaderboard's OCRBench v2 uses the private set
  and VLMEvalKit the public one; the units differ. On the pilot, tau was 0.6 (EN, 5 models) and
  0.4 (ZH). Published OCRBench v2 EN (private): GPT-4o 47.6, Ovis2-8B 46.1, InternVL3-8B 45.3,
  GPT-4o-mini 44.1, Qwen2.5-VL-7B 41.8, LLaVA-OV-7B 33.7 (`data/scores_long.csv`).
- One link in the Stage 1 materials pointed to a site with an expired TLS certificate. Check every
  link opens before submitting.

---

## 10. Gotchas learned the hard way

- The Mac's network blocks api.openai.com and Gemini, so API models were run on the Brev machine.
- The A6000 host CPU is slow (Xeon 2.2 GHz): transformers batch-1 decoding is CPU-bound at
  ~11 tokens/s per process, so two processes per GPU are run. InternVL3 needs ~18 slot-hours for
  the whole main run; Qwen on transformers needed ~1.4× that.
- `LLaVA_OneVision_HF` hard-codes `generate(max_new_tokens=2048)` → `generate_overrides` in
  config.yaml force 4096 + greedy (`run.build_model`).
- InternVL/MiniCPM have their own per-benchmark prompts in VLMEvalKit; we force the benchmark's
  prompt for every model (`use_custom_prompt` off).
- OmniDocBench's evaluator matches ground truth by position → `run.py` writes a subset GT TSV in
  prediction order (tested equal to a full run).
- Empty olmOCR answers read back from xlsx as NaN → shimmed in `run.py`.
- Brev instances of these types cannot be stopped, only deleted, and billing runs from boot.
- One guard race deleted a machine during a manual restart. Fixed: the queue writes
  `QUEUE_STOPPED` when stopped, `QUEUE_DONE` only when finished.
- Single `ssh`/`brev ls` failures happen; don't treat one failed check as "machine gone".
- `uv pip install` needs `--index-strategy unsafe-best-match` with the PyTorch index, or it swaps torch cu128 for another build.

---

## 11. What is left (in order)

1. Watch the run until the queue is done (or the cap is reached): first confirm Qwen on vLLM
   gives a score; check `n_failed`; react to ❌ messages.
2. Pull the results, commit, push, mirror to Drive (section 8).
3. Final analysis on all 6 models + figures redrawn on the Mac; check `missing_runs.csv`.
4. Write report sections D.1–D.6 with the notes in section 9.
5. Fix the student doc: one model list, the ER1 range as one value, OmniDocBench direction/version.
6. Final link check; final Drive sync.
7. Remind Adilzhan: rotate the OpenAI key, revoke rclone's Google access, confirm `brev ls` is empty.
