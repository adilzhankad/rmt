"""Output-level failure counts of every scored main run -> processed/output_checks_main.csv.

    python experiment/output_checks.py

Read from the exact model answers (raw/rerun/main/<model>/<benchmark>/predictions.jsonl) and,
for the API models, the finish reasons in run_meta.json. Rules (fixed here, applied to every run):

  empty      answer is empty after stripping white space
  long       answer longer than 12,000 characters (the A3 warning sign for loops and cut answers)
  loop       the last 2,000 characters contain one block of 20-200 characters (with at least 3
             visible characters) repeated in a row over at least 1,000 characters (a repetition
             loop that runs into the output limit; blank padding is not counted)
  cut (API)  the API reported finish_reason = "length", i.e. the answer stopped at 4,096 tokens

The longest looping answer of each run is written to processed/failure_examples_main.csv
(item, length and the last 300 characters) for the individual failure inspection.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, RAW, load_config  # noqa: E402

LONG = 12_000
LOOP = re.compile(r"(.{20,200}?)\1{4,}", re.S)


def is_loop(text):
    for m in LOOP.finditer(text[-2000:]):
        if len(m.group(0)) >= 1000 and len(re.sub(r"\s", "", m.group(1))) >= 3:
            return True
    return False


def main():
    cfg = load_config()
    rows, examples = [], []
    for meta_file in sorted((RAW / "main").glob("*/*/run_meta.json")):
        meta = json.loads(meta_file.read_text())
        if meta.get("primary_score") in (None, "") or meta["model"] not in cfg["models"]:
            continue
        preds = [json.loads(line) for line in (meta_file.parent / "predictions.jsonl").read_text().splitlines() if line]
        texts = [(str(p["index"]), "" if p.get("prediction") is None else str(p["prediction"])) for p in preds]
        loops = [(i, t) for i, t in texts if is_loop(t)]
        finish = (meta.get("usage") or {}).get("finish_reasons", {})
        rows.append({"model": meta["model"], "benchmark": meta["benchmark"], "answers": len(texts),
                     "empty": sum(not t.strip() for _, t in texts),
                     "long": sum(len(t) > LONG for _, t in texts),
                     "loop": len(loops),
                     "cut_api": finish.get("length", "") if finish else "",
                     "median_chars": int(pd.Series([len(t) for _, t in texts]).median()),
                     "max_chars": max(len(t) for _, t in texts)})
        for i, t in sorted(loops, key=lambda x: -len(x[1]))[:1]:
            examples.append({"model": meta["model"], "benchmark": meta["benchmark"], "item": i,
                             "chars": len(t), "tail_300": t[-300:]})
    PROCESSED.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED / "output_checks_main.csv", index=False)
    pd.DataFrame(examples).to_csv(PROCESSED / "failure_examples_main.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
