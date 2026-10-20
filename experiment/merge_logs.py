"""Merge a run log produced elsewhere (Kaggle) into logs/run_log.csv, plus its raw/ files.

    python experiment/merge_logs.py <unzipped kaggle output dir>

Rows are appended unchanged and in time order; a row already present (same timestamp,
model, benchmark, split) is skipped. raw/rerun/... files are copied without overwriting
existing ones that differ (that would be an error and stops the merge).
"""
import csv
import filecmp
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, RUN_LOG, RUN_LOG_COLUMNS, USAGE_DIR, append_run_log  # noqa: E402


def key(r):
    return (r["timestamp_utc"], r["model"], r["benchmark"], r["split"])


def main(src):
    src = Path(src)
    have = set()
    if RUN_LOG.exists():
        have = {key(r) for r in csv.DictReader(open(RUN_LOG, newline=""))}
    rows = list(csv.DictReader(open(src / "logs" / "run_log.csv", newline="")))
    assert rows and list(rows[0]) == RUN_LOG_COLUMNS, "unexpected run_log columns"
    for f in sorted((src / "raw" / "rerun").rglob("*")):
        if f.is_dir() or f.suffix == ".pkl":
            continue
        dest = RAW / f.relative_to(src / "raw" / "rerun")
        if dest.exists() and not filecmp.cmp(f, dest, shallow=False):
            raise SystemExit(f"{dest} exists and differs; not overwriting")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
    added = 0
    for r in sorted(rows, key=lambda r: r["timestamp_utc"]):
        if key(r) not in have:
            append_run_log(r)
            added += 1
    u = src / "logs" / "usage" / "usage.jsonl"
    if u.exists():
        USAGE_DIR.mkdir(parents=True, exist_ok=True)
        existing = (USAGE_DIR / "usage.jsonl").read_text() if (USAGE_DIR / "usage.jsonl").exists() else ""
        with open(USAGE_DIR / "usage.jsonl", "a") as f:
            for line in u.read_text().splitlines():
                if line and line not in existing:
                    f.write(line + "\n")
    print(f"{added} new run rows merged from {src}")


if __name__ == "__main__":
    main(sys.argv[1])
