"""Estimate the API cost of the main run from the tokens actually used in the pilot.

    python experiment/estimate_cost.py

Reads logs/usage/usage.jsonl (written by run.py, counts every API call incl. retries),
takes tokens per item for each piloted model x benchmark and scales to the main sample.
Prices (USD per 1M tokens) are in experiment/prices.yaml - CHECK THEM on the providers'
pricing pages on the day of the run; thinking tokens are billed as output.
Models without a pilot are listed as "no pilot - no estimate".
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, USAGE_DIR, load_config  # noqa: E402


def main():
    cfg = load_config()
    prices = yaml.safe_load((EXP / "prices.yaml").read_text())["usd_per_million_tokens"]
    last = {}
    path = USAGE_DIR / "usage.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        r = json.loads(line)
        if r["split"] == "pilot" and r["usage"]["calls"]:
            last[(r["model"], r["benchmark"])] = r
    n_main = cfg["items"]["main"]
    total = 0.0
    print(f"{'model':24s} {'benchmark':13s} {'calls/item':>10s} {'in/item':>9s} {'out/item':>9s} {'main USD':>9s}")
    for m, mc in cfg["models"].items():
        if mc["kind"] != "api":
            continue
        p = prices[m]
        for bench in cfg["benchmarks"]:
            r = last.get((m, bench))
            if not r:
                print(f"{m:24s} {bench:13s} {'no pilot - no estimate':>40s}")
                continue
            n, u = r["n_items"], r["usage"]
            t_in = u["tokens"]["input"] / n
            t_out = (u["tokens"]["output"] + u["tokens"]["thinking"]) / n
            if p["input"] is None or p["output"] is None:
                print(f"{m:24s} {bench:13s} {u['calls'] / n:10.2f} {t_in:9.0f} {t_out:9.0f} {'no price':>9s}")
                continue
            usd = n_main * (t_in * p["input"] + t_out * p["output"]) / 1e6
            total += usd
            print(f"{m:24s} {bench:13s} {u['calls'] / n:10.2f} {t_in:9.0f} {t_out:9.0f} {usd:9.2f}")
    print(f"\nEstimated main-run cost of the piloted model x benchmark cells: {total:.2f} USD")


if __name__ == "__main__":
    main()
