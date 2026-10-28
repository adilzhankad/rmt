"""Derived numbers quoted in the A4 report -> processed/a4_numbers.json.

    python experiment/a4_numbers.py

Only arithmetic on files that analyze.py and collect.py already wrote, so every number in the
report that is not in those files is computed here, from them:
  * ER2: mean tau-b of the two same-type pairs, of the four cross-type pairs, and their gap,
    for the point estimate and in each of the 1,000 bootstrap replicates (bootstrap_main.csv);
  * the verdict computed with the pre-registered plain-mean statistic (harness.bootstrap_tau);
  * ER3: the first-ranked model on each benchmark;
  * the engine check: Qwen2.5-VL OCRBench v2 on transformers vs vLLM, and whether the rank changes;
  * leave-one-model-out: mean and min tau-b when each of the six complete models is left out;
  * answer-level failure totals over the 25 scored main runs (output_checks_main.csv);
  * sample facts (strata per benchmark, English pages in the OmniDocBench sample), the number of
    identical answers in the two Qwen2.5-VL OCRBench v2 runs, and the run time of the scored main runs.
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
    six = ranks.set_index("model")
    labels = [c.replace(" score", "") for c in six.columns if c.endswith(" score")]
    oriented = {b: {m: harness.oriented(v, "lower" if b == "OmniDocBench" else "higher")
                    for m, v in six[f"{b} score"].items()} for b in labels}
    loo = {}
    for left in six.index:
        tab = {b: {m: v for m, v in d.items() if m != left} for b, d in oriented.items()}
        taus = [t for *_, t, _ in harness.pairwise(tab)]
        loo[left] = {"mean_tau": sum(taus) / len(taus), "min_tau": min(taus),
                     "verdict": harness.verdict(sum(taus) / len(taus), min(taus))}
    out["leave_one_model_out"] = loo

    oc = pd.read_csv(PROCESSED / "output_checks_main.csv")
    out["answer_level"] = {"runs": len(oc), "answers": int(oc.answers.sum()), "empty": int(oc["empty"].sum()),
                           "long": int(oc.long.sum()), "loop": int(oc.loop.sum()),
                           "cut_api": int(pd.to_numeric(oc.cut_api, errors="coerce").fillna(0).sum()),
                           "loop_by_benchmark": oc.groupby("benchmark").loop.sum().astype(int).to_dict()}
    items = {b: pd.read_csv(ROOT / "items" / f"{b}_main.csv") for b in ("OCRBench_v2", "CCOCR", "OmniDocBench", "olmOCRBench")}
    out["sample"] = {"strata": {b: int(d.stratum.nunique()) for b, d in items.items()},
                     "omnidoc_english_pages": int(items["OmniDocBench"].stratum.str.startswith("english").sum())}

    def answers(path):
        return {str(j["index"]): str(j["prediction"]) for j in map(json.loads, path.read_text().splitlines()) if j}
    a = answers(ROOT / "raw/superseded/main/Qwen2.5-VL-7B-Instruct/OCRBench_v2/predictions.jsonl")
    b = answers(ROOT / "raw/rerun/main/Qwen2.5-VL-7B-Instruct/OCRBench_v2/predictions.jsonl")
    out["engine_check"]["identical_answers"] = sum(a[k] == b.get(k) for k in a)
    out["engine_check"]["answers"] = len(a)

    scored = log[(log.split == "main") & log.primary_score.notna()]
    out["run_hours"] = {"scored_main_runs": len(scored), "total": round(scored.duration_s.sum() / 3600, 1),
                        "by_model": scored.groupby("model").duration_s.sum().div(3600).round(1).to_dict(),
                        "by_benchmark": scored.groupby("benchmark").duration_s.sum().div(3600).round(1).to_dict()}
    (PROCESSED / "a4_numbers.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
