"""Derived numbers quoted in the A4 report -> processed/a4_numbers.json.

    python experiment/a4_numbers.py

Only arithmetic on files that analyze.py and collect.py already wrote, so every number in the
report that is not in those files is computed here, from them:
  * ER2: mean tau-b of the two same-type pairs, of the four cross-type pairs, and their gap,
    for the point estimate and in each of the 1,000 bootstrap replicates (bootstrap_main.csv);
  * the verdict computed with the pre-registered plain-mean statistic (harness.bootstrap_tau);
  * ER3: the first-ranked model on each benchmark;
  * the engine check: Qwen2.5-VL OCRBench v2 on transformers vs vLLM, and whether the rank changes.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, ROOT, harness  # noqa: E402

SAME = [("OCRBench v2", "CC-OCR"), ("OmniDocBench", "olmOCR-Bench")]


def main():
    pairs = pd.read_csv(PROCESSED / "tau_pairs_main.csv")
    key = {(r.a, r.b): r for r in pairs.itertuples()}
    same = [key[p].tau_b for p in SAME]
    cross = [r.tau_b for (a, b), r in key.items() if (a, b) not in SAME]
    out = {"same_type_taus": same, "cross_type_taus": cross,
           "same_type_mean": sum(same) / 2, "cross_type_mean": sum(cross) / 4}
    out["gap_same_minus_cross"] = out["same_type_mean"] - out["cross_type_mean"]

    boot = pd.read_csv(PROCESSED / "bootstrap_main.csv")
    s_cols = [f"{a} x {b}" for a, b in SAME]
    c_cols = [c for c in boot.columns if " x " in c and c not in s_cols]
    gap = boot[s_cols].mean(axis=1) - boot[c_cols].mean(axis=1)
    g = sorted(gap)
    out["gap_bootstrap"] = {"replicates": len(g), "ci95": [g[int(0.025 * len(g))], g[int(0.975 * len(g)) - 1]],
                            "share_same_above_cross": float((gap > 0).mean()),
                            "share_gap_below_0.15": float(((gap > 0) & (gap < 0.15)).mean())}

    plain = pairs["tau_b_plain_mean"].tolist()
    out["plain_mean_statistic"] = {"mean_tau": sum(plain) / len(plain), "min_tau": min(plain),
                                   "verdict": harness.verdict(sum(plain) / len(plain), min(plain))}

    ranks = pd.read_csv(PROCESSED / "rank_table_main.csv").dropna()
    out["first_ranked"] = {c.replace(" rank", ""): ranks.loc[ranks[c].idxmin(), "model"]
                           for c in ranks.columns if c.endswith(" rank")}
    top = pd.Series(out["first_ranked"]).value_counts()
    out["leader_max_benchmarks"] = int(top.max())

    log = pd.read_csv(ROOT / "logs" / "run_log.csv")
    q = log[(log.model == "Qwen2.5-VL-7B-Instruct") & (log.benchmark == "OCRBench_v2") & (log.split == "main")
            & log.primary_score.notna()]
    tr = float(q[~q.notes.astype(str).str.contains("vLLM")].primary_score.iloc[0])
    vl = float(q[q.notes.astype(str).str.contains("vLLM")].primary_score.iloc[-1])
    ocr = dict(zip(ranks.model, ranks["OCRBench v2 score"]))
    rank_with = lambda v: 1 + sum(x > v for m, x in ocr.items() if m != "Qwen2.5-VL-7B-Instruct")
    out["engine_check"] = {"transformers": tr, "vllm": vl, "rank_among_six_transformers": rank_with(tr),
                           "rank_among_six_vllm": rank_with(vl)}
    (PROCESSED / "a4_numbers.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
