"""Run the queue of model x benchmark jobs on one GPU machine (used on NVIDIA Brev).

    python experiment/queue_runs.py                 # pilot of new models, then the main run
    python experiment/queue_runs.py --dry-run       # print the queue only

Each job is one `run.py` process in the model's environment (config.yaml `env`). Two jobs share
the GPU when their memory fits (two processes keep it busier than one batch-1 generate loop).
Jobs that already have a score in logs/run_log.csv are skipped, so the queue can be restarted;
run.py processes left by a previous queue are adopted, not restarted. A file QUEUE_ORDER (one
model per line) switches the GPU lane to model-by-model order; it is read on every step.
API models run in their own lane and only when the file ALLOW_API exists (they cost money).
On SIGTERM (the budget guard) every running job is stopped and logged as ABORTED; a stopped job
can be resumed later because VLMEvalKit keeps its partial answers in the work directory.
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, ROOT, RUN_LOG, append_run_log, load_config, load_env, read_items  # noqa: E402

PYTHON = {"main": ROOT / ".venv/bin/python", "open": ROOT / ".venv-open/bin/python",
          "legacy": ROOT / ".venv-legacy/bin/python"}
GPU_GB = 44                                   # usable memory of a 48 GB card, with headroom
MODEL_GB = {"deepseek_vl2_small": 40}         # 16B MoE in bf16; every other open model is ~8B
DEFAULT_GB = 21
# cheapest benchmarks first: if the budget runs out, every model still has the same benchmarks
BENCH_ORDER = ["OCRBench_v2", "CCOCR", "olmOCRBench", "OmniDocBench"]
# The two replacement models go straight to the main run (no separate 20-item pilot) to save GPU
# budget; a broken setup still shows up on the first, fastest benchmark.
PILOT_MODELS = []
CONSOLE = ROOT / "console_logs"
STOP = False
STOP_REASON = ""


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def say(text):
    print(f"{now()} {text}", flush=True)
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=20)
        except Exception:
            pass


def logged():
    """(model, bench, split) -> last run_log row."""
    if not RUN_LOG.exists():
        return {}
    with open(RUN_LOG, newline="") as f:
        return {(r["model"], r["benchmark"], r["split"]): r for r in csv.DictReader(f)}


def scored(rows, key):
    return bool((rows.get(key) or {}).get("primary_score", "").strip())


def build_queue(cfg):
    local = [m for m, v in cfg["models"].items() if v["kind"] == "local"]
    api = [m for m, v in cfg["models"].items() if v["kind"] == "api"]
    gpu = [(m, b, "pilot") for m in PILOT_MODELS for b in BENCH_ORDER]
    gpu += [(m, b, "main") for b in BENCH_ORDER for m in local]
    return gpu, [(m, b, "main") for b in BENCH_ORDER for m in api]


class Adopted:
    """A run.py process started by an earlier queue (looks like a Popen to the loop)."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        return None if Path(f"/proc/{self.pid}").exists() else "ended"

    def wait(self, timeout=None):
        end = time.time() + (timeout or 1e9)
        while self.poll() is None:
            if time.time() > end:
                raise subprocess.TimeoutExpired("run.py", timeout)
            time.sleep(1)


def running_runs():
    """(model, bench, split) -> pid for run.py processes already on this machine."""
    found = {}
    for d in Path("/proc").glob("[0-9]*"):
        try:
            argv = (d / "cmdline").read_bytes().decode().split("\0")
        except OSError:
            continue
        if any(a.endswith("experiment/run.py") for a in argv) and "--model" in argv:
            get = lambda flag: argv[argv.index(flag) + 1]
            found[(get("--model"), get("--bench"), get("--split"))] = int(d.name)
    return found


def order_queue(gpu_q):
    f = ROOT / "QUEUE_ORDER"
    if not f.exists():
        return
    models = [m.strip() for m in f.read_text().split() if m.strip()]
    gpu_q.sort(key=lambda j: (j.split != "pilot", models.index(j.model) if j.model in models else len(models),
                              BENCH_ORDER.index(j.bench)))


class Job:
    def __init__(self, cfg, model, bench, split):
        self.model, self.bench, self.split = model, bench, split
        m = cfg["models"][model]
        self.env = m.get("env", "main")
        self.gb = 0 if m["kind"] == "api" else MODEL_GB.get(model, DEFAULT_GB)
        self.proc = self.start = None
        self.retry = False

    @property
    def key(self):
        return (self.model, self.bench, self.split)

    def launch(self):
        CONSOLE.mkdir(exist_ok=True)
        out = open(CONSOLE / f"{self.split}_{self.model}_{self.bench}.txt", "a")
        cmd = [str(PYTHON[self.env]), str(ROOT / "experiment/run.py"), "--model", self.model,
               "--bench", self.bench, "--split", self.split, "--operator", "Adilzhan"]
        if self.env == "main":
            cmd += ["--api-nproc", "8"]
        self.start = time.time()
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        say(f"▶️ {self.model} · {self.bench} · {self.split}")


def report(job, rows):
    r = rows.get(job.key) or {}
    mins = (time.time() - job.start) / 60
    if job.proc.returncode in (0, None) and scored(rows, job.key):   # None: adopted process
        say(f"✅ {job.model} · {job.bench} · {job.split}\nscore {float(r['primary_score']):.3f} · "
            f"failed answers {r['n_failed']} of {r['n_items']} · {mins:.0f} min")
    else:
        say(f"❌ {job.model} · {job.bench} · {job.split} (exit {job.proc.returncode})\n{r.get('notes', '')[:300]}")


def abort(job, cfg):
    os.killpg(job.proc.pid, signal.SIGTERM)
    try:
        job.proc.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(job.proc.pid, signal.SIGKILL)
        job.proc.wait()
    b = cfg["benchmarks"][job.bench]
    append_run_log({"timestamp_utc": now(), "operator": "Adilzhan", "model": job.model, "benchmark": job.bench,
                    "split": job.split, "n_items": len(read_items(job.bench, job.split)),
                    "duration_s": round(time.time() - job.start, 1), "vlmevalkit_commit": cfg["vlmevalkit"]["commit"],
                    "tsv_md5": b["md5"], "max_tokens": cfg["max_tokens"], "temperature": cfg["temperature"],
                    "notes": f"ABORTED during inference ({STOP_REASON}); no score "
                             "(partial answers kept for a resume)"})
    say(f"⏹ {job.model} · {job.bench} · {job.split}: stopped ({STOP_REASON})")


def main():
    global STOP
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_env()
    cfg = load_config()
    gpu_q, api_q = build_queue(cfg)
    rows = logged()
    gpu_q = [Job(cfg, *k) for k in gpu_q if not scored(rows, k)]
    api_q = [Job(cfg, *k) for k in api_q if not scored(rows, k)]
    if args.dry_run:
        for j in gpu_q + api_q:
            print(j.env.ljust(7), str(j.gb).rjust(3), *j.key)
        return

    def on_term(*_):
        global STOP, STOP_REASON
        STOP = True
        reason = ROOT / "STOP_REASON"   # written by whoever stops the queue
        STOP_REASON = reason.read_text().strip() if reason.exists() else "stopped by SIGTERM"
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    running, failed_pilot = [], set()
    for key, pid in running_runs().items():
        job = next((j for j in gpu_q + api_q if j.key == key), None)
        if job:
            (gpu_q if job in gpu_q else api_q).remove(job)
            job.proc, job.start = Adopted(pid), time.time()
            running.append(job)
            say(f"🔗 adopted the running {job.model} · {job.bench} · {job.split}")
    say(f"🚀 queue started: {len(gpu_q)} GPU jobs, {len(api_q)} API jobs")
    while (gpu_q or running or (api_q and (ROOT / "ALLOW_API").exists())) and not STOP:
        for job in [j for j in running if j.proc.poll() is not None]:
            running.remove(job)
            rows = logged()
            report(job, rows)
            if job.split == "pilot" and not scored(rows, job.key):
                failed_pilot.add(job.model)
            elif not scored(rows, job.key) and not job.retry:
                # one retry, first in its lane; it resumes from the answers already saved
                again = Job(cfg, *job.key)
                again.retry = True
                (api_q if job.env == "main" else gpu_q).insert(0, again)
                say(f"🔁 {job.model} · {job.bench} · {job.split}: queued once more")
        # a new model goes to the main run only if its whole pilot worked
        for job in [j for j in gpu_q if j.split == "main" and j.model in failed_pilot]:
            gpu_q.remove(job)
            say(f"⏭ {job.model} · {job.bench} · main skipped: its pilot failed")
        order_queue(gpu_q)
        used = sum(j.gb for j in running)
        pilot_left = {j.model for j in gpu_q + running if j.split == "pilot"}
        ready = [j for j in gpu_q if not (j.split == "main" and j.model in pilot_left)]
        if ready and ready[0].gb + used <= GPU_GB:   # no skipping past a big job, so it is not starved
            job = ready[0]
            gpu_q.remove(job)
            job.launch()
            running.append(job)
            continue
        if api_q and (ROOT / "ALLOW_API").exists() and not any(j.env == "main" for j in running):
            job = api_q.pop(0)
            job.launch()
            running.append(job)
            continue
        time.sleep(20)
    if STOP:
        for job in running:
            if job.proc.poll() is None:
                abort(job, cfg)
            else:
                report(job, logged())
        say(f"🛑 queue stopped ({STOP_REASON})")
    else:
        say("🏁 queue finished" + (f"; API jobs waiting for ALLOW_API: {len(api_q)}" if api_q else ""))
    # the budget guard ends the run (and deletes the machine) on QUEUE_DONE only, so a stopped
    # queue, e.g. one restarted by hand, must never write it
    (ROOT / ("QUEUE_STOPPED" if STOP else "QUEUE_DONE")).write_text(now() + "\n")


if __name__ == "__main__":
    main()
