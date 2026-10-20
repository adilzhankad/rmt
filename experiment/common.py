"""Shared paths, config loading and small helpers for the Stage 2 re-run."""
import csv
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiment"
ITEMS = ROOT / "items"
RAW = ROOT / "raw" / "rerun"
LOGS = ROOT / "logs"
PROCESSED = ROOT / "processed"
CHARTS = ROOT / "charts"
RUN_LOG = LOGS / "run_log.csv"
USAGE_DIR = LOGS / "usage"

RUN_LOG_COLUMNS = [
    "timestamp_utc", "operator", "model", "benchmark", "split", "n_items", "n_failed",
    "duration_s", "vlmevalkit_commit", "tsv_md5", "max_tokens", "temperature",
    "primary_score", "notes",
]
FAIL_MSG = "Failed to obtain answer via API."
SPLITS = ("pilot", "main")

sys.path.insert(0, str(ROOT / "code"))
import harness  # noqa: E402  (fixed study rules, Kendall tau-b, bootstrap)


def load_env(path=ROOT / ".env"):
    """Read KEY=VALUE lines from .env into os.environ (existing values win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def load_config(path=EXP / "config.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    # The rules live in harness.py; the config only mirrors them.
    assert cfg["seed"] == harness.SEED, "config seed differs from harness.SEED"
    assert cfg["items"]["main"] == harness.ITEMS_PER_BENCHMARK, "config main N differs from harness"
    assert cfg["bootstrap"] == harness.BOOTSTRAP, "config bootstrap differs from harness"
    return cfg


def lmu_root():
    return Path(os.environ.get("LMUData", ROOT / "LMUData"))


def md5_file(path, chunk=1 << 22):
    """md5 of a file, cached next to it by (size, mtime) because the TSVs are GBs."""
    path = Path(path)
    st = path.stat()
    cache = path.with_suffix(path.suffix + ".md5.json")
    key = [st.st_size, int(st.st_mtime)]
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            return c["md5"]
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    digest = h.hexdigest()
    try:
        cache.write_text(json.dumps({"key": key, "md5": digest}))
    except OSError:
        pass
    return digest


def items_path(bench, split):
    return ITEMS / f"{bench}_{split}.csv"


def read_items(bench, split):
    with open(items_path(bench, split), newline="") as f:
        return list(csv.DictReader(f))


def append_run_log(row, path=RUN_LOG):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RUN_LOG_COLUMNS})


class RedactSecrets(logging.Filter):
    """VLMEvalKit logs the API key when a GPT wrapper is built; never let it reach output."""

    NAMES = ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN")

    def filter(self, record):
        msg = record.getMessage()
        for name in self.NAMES:
            v = os.environ.get(name)
            if v and v in msg:
                msg = msg.replace(v, "***")
                record.msg, record.args = msg, ()
        return True


def install_redaction():
    f = RedactSecrets()
    for name, lg in list(logging.root.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger):
            lg.addFilter(f)
            for h in lg.handlers:
                h.addFilter(f)
    logging.getLogger().addFilter(f)
    for h in logging.getLogger().handlers:
        h.addFilter(f)
