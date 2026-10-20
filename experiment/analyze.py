"""Agreement analysis of the re-run -> processed/ and charts/.

    python experiment/analyze.py --split main      # or --split pilot

1. Score table: primary metric of every model on every benchmark, oriented so that
   higher = better (harness.oriented; OmniDocBench edit distance is flipped).
2. Pairwise Kendall tau-b (harness.pairwise, >= MIN_SHARED_MODELS shared models),
   mean / min tau and harness.verdict().
3. Uncertainty, two ways:
   a. stratified bootstrap (main result): items are resampled inside the sampling
      strata, the SAME draw is used for every model, and each benchmark's official
      aggregate is rebuilt from the item scores (item_scores.py), so a replicate is
      the same statistic as the point estimate;
   b. harness.bootstrap_tau as fixed before data collection (plain item means,
      unstratified) - reported as a sensitivity check.
   Both use harness.BOOTSTRAP replicates and harness.SEED.
4. Language (ER3): English vs Chinese rankings inside OCRBench v2 and OmniDocBench,
   and cross-benchmark agreement in each language.
5. Figures fig3-fig8 in charts/.
"""
import argparse
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHARTS, PROCESSED, harness, load_config, read_items  # noqa: E402
from item_scores import AGG  # noqa: E402

# (benchmark, metric) pairs used for the language analysis
LANG_MEASURES = {"OCRBench v2 EN": ("OCRBench_v2", "English Overall Score", "higher"),
                 "OCRBench v2 ZH": ("OCRBench_v2", "Chinese Overall Score", "higher"),
                 "OmniDocBench EN": ("OmniDocBench", "overall_EN", "lower"),
                 "OmniDocBench ZH": ("OmniDocBench", "overall_CH", "lower")}


def percentile_ci(values, reps):
    v = sorted(x for x in values if not math.isnan(x))
    if len(v) < reps * 0.9:   # too many undefined replicates (ties everywhere): no CI
        return float("nan"), float("nan")
    k = len(v)
    return v[int(0.025 * k)], v[int(0.975 * k) - 1]


# ---------------------------------------------------------------- data
def load_scores(split, cfg, processed=PROCESSED):
    df = pd.read_csv(Path(processed) / "scores_rerun.csv")
    df = df[df["split"] == split]
    raw = {}
    for r in df.itertuples():
        raw.setdefault((r.benchmark, r.metric), {})[r.model] = (float(r.value), r.direction)
    table = {}
    for bench, b in cfg["benchmarks"].items():
        vals = raw.get((bench, b["primary_metric"]), {})
        table[b["label"]] = {m: harness.oriented(v, d) for m, (v, d) in vals.items()}
    return df, raw, table


def load_items(split, cfg, processed=PROCESSED):
    """{bench: {model: {item: [rows]}}} and {bench: {item: stratum}}."""
    path = Path(processed) / "item_scores.csv"
    if not path.exists():
        return {}, {}
    df = pd.read_csv(path, dtype={"item": str, "unit": str})
    df = df[df["split"] == split]
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in df.to_dict("records"):
        if "ignored" in r and isinstance(r["ignored"], str):
            r["ignored"] = r["ignored"] == "True"
        rows[r["benchmark"]][r["model"]][r["item"]].append(r)
    strata = {bench: {str(i["index"]): i["stratum"] for i in read_items(bench, split)} for bench in rows}
    return rows, strata


# ---------------------------------------------------------------- bootstrap
def stratified_bootstrap(rows, strata, models_by_bench, cfg, reps=harness.BOOTSTRAP, seed=harness.SEED):
    """Yields one dict per replicate: {(bench, metric): {model: value}} (raw, not oriented)."""
    rng = random.Random(seed)
    benches = sorted(rows)
    groups = {b: defaultdict(list) for b in benches}
    for b in benches:
        for item, s in sorted(strata[b].items()):
            groups[b][s].append(item)
    for _ in range(reps):
        rep = {}
        for b in benches:
            draw = [rng.choice(pool) for s in sorted(groups[b]) for pool in [groups[b][s]] for _ in pool]
            for m in models_by_bench[b]:
                agg = AGG[b]([r for it in draw for r in rows[b][m].get(it, [])])
                for metric, v in agg.items():
                    rep.setdefault((b, metric), {})[m] = v
        yield rep


def oriented_table(rep, cfg, measures=None):
    if measures is None:
        measures = {b["label"]: (k, b["primary_metric"], b["direction"]) for k, b in cfg["benchmarks"].items()}
    return {name: {m: harness.oriented(v, d) for m, v in rep.get((bench, metric), {}).items()}
            for name, (bench, metric, d) in measures.items()}


def plain_items(rows, cfg):
    """Per-item score for harness.bootstrap_tau (pre-registered plain-mean version), oriented."""
    out = {}
    for bench, by_model in rows.items():
        b = cfg["benchmarks"][bench]
        label = b["label"]
        out[label] = defaultdict(dict)
        for m, items in by_model.items():
            for it, rs in items.items():
                if bench == "OCRBench_v2":
                    rs = [r for r in rs if r["lang"] == "en" and not r["ignored"]]
                elif bench == "OmniDocBench":
                    rs = [r for r in rs if r["lang"] == "EN"]
                if rs:
                    out[label][it][m] = harness.oriented(sum(r["score"] for r in rs) / len(rs), b["direction"])
        # keep items scored for every model
        ms = set(by_model)
        out[label] = {it: v for it, v in out[label].items() if set(v) == ms}
    return out


# ---------------------------------------------------------------- analysis
def analyze(split, cfg, processed=PROCESSED, charts=CHARTS, reps=harness.BOOTSTRAP, figures=True):
    processed, charts = Path(processed), Path(charts)
    df, raw, table = load_scores(split, cfg, processed)
    labels = [b["label"] for b in cfg["benchmarks"].values()]
    models = sorted({m for t in table.values() for m in t})
    out = {"split": split, "models": models, "n_models": len(models)}

    # 1. score / rank table
    rank_rows = []
    for m in models:
        row = {"model": m}
        for lab in labels:
            t = table[lab]
            if m in t:
                row[f"{lab} score"] = t[m] if t[m] >= 0 else -t[m]
                row[f"{lab} rank"] = 1 + sum(v > t[m] for v in t.values())
        rank_rows.append(row)
    pd.DataFrame(rank_rows).to_csv(processed / f"rank_table_{split}.csv", index=False)

    # 2. pairwise tau-b
    pairs = harness.pairwise(table)
    taus = [p[3] for p in pairs if p[3] is not None and not math.isnan(p[3])]
    out["pairs"] = [{"a": a, "b": b, "n_shared": n, "tau_b": tau} for a, b, n, tau, _ in pairs]
    if taus:
        out["mean_tau"], out["min_tau"] = sum(taus) / len(taus), min(taus)
        out["verdict"] = harness.verdict(out["mean_tau"], out["min_tau"])
    else:
        out["mean_tau"] = out["min_tau"] = None
        out["verdict"] = f"not estimable (< {harness.MIN_SHARED_MODELS} shared models)"
    # pilot with few models: share of model pairs ordered the same way on each benchmark pair
    out["pair_concordance"] = []
    for a, b in itertools.combinations(labels, 2):
        shared = sorted(set(table[a]) & set(table[b]))
        conc = disc = 0
        for x, y in itertools.combinations(shared, 2):
            s = (table[a][x] - table[a][y]) * (table[b][x] - table[b][y])
            conc += s > 0
            disc += s < 0
        out["pair_concordance"].append({"a": a, "b": b, "n_shared": len(shared), "concordant": conc,
                                        "discordant": disc})

    # 4. language (point estimates)
    lang_tab = {name: {m: harness.oriented(v, d) for m, (v, _) in raw.get((bench, metric), {}).items()}
                for name, (bench, metric, d) in LANG_MEASURES.items()}
    lang_pairs = [("OCRBench v2 EN", "OCRBench v2 ZH"), ("OmniDocBench EN", "OmniDocBench ZH"),
                  ("OCRBench v2 EN", "OmniDocBench EN"), ("OCRBench v2 ZH", "OmniDocBench ZH")]

    def lang_tau(tab, a, b):
        shared = sorted(set(tab[a]) & set(tab[b]))
        if len(shared) < harness.MIN_SHARED_MODELS:
            return float("nan"), len(shared)
        return harness.kendall_tau_b([tab[a][m] for m in shared], [tab[b][m] for m in shared]), len(shared)
    out["language"] = [{"a": a, "b": b, "tau_b": lang_tau(lang_tab, a, b)[0], "n_shared": lang_tau(lang_tab, a, b)[1]}
                       for a, b in lang_pairs]

    # 3a. stratified bootstrap
    rows, strata = load_items(split, cfg, processed)
    boot = None
    if rows and len(models) >= harness.MIN_SHARED_MODELS:
        models_by_bench = {b: sorted(rows[b]) for b in rows}
        pair_draws = defaultdict(list)
        lang_draws = defaultdict(list)
        mean_draws, min_draws, verdicts = [], [], defaultdict(int)
        for rep in stratified_bootstrap(rows, strata, models_by_bench, cfg, reps):
            tab = oriented_table(rep, cfg)
            ps = harness.pairwise(tab)
            ts = []
            for a, b, n, tau, _ in ps:
                pair_draws[(a, b)].append(float("nan") if tau is None else tau)
                if tau is not None and not math.isnan(tau):
                    ts.append(tau)
            if ts:
                mn, mi = sum(ts) / len(ts), min(ts)
                mean_draws.append(mn)
                min_draws.append(mi)
                verdicts[harness.verdict(mn, mi)] += 1
            lt = oriented_table(rep, cfg, LANG_MEASURES)
            for a, b in lang_pairs:
                lang_draws[(a, b)].append(lang_tau(lt, a, b)[0])
        for p in out["pairs"]:
            p["ci_stratified"] = percentile_ci(pair_draws[(p["a"], p["b"])], reps)
        for p in out["language"]:
            p["ci_stratified"] = percentile_ci(lang_draws[(p["a"], p["b"])], reps)
        out["mean_tau_ci"] = percentile_ci(mean_draws, reps)
        out["min_tau_ci"] = percentile_ci(min_draws, reps)
        out["verdict_share"] = {k: v / reps for k, v in sorted(verdicts.items())}
        boot = {"mean": mean_draws}

        # 3b. pre-registered harness bootstrap (plain item means)
        plain = plain_items(rows, cfg)
        for p in out["pairs"]:
            if p["tau_b"] is None:
                continue
            ms = sorted(set(table[p["a"]]) & set(table[p["b"]]))
            p["ci_harness_plain_mean"] = harness.bootstrap_tau(plain, p["a"], p["b"], ms, reps, harness.SEED)
            # point estimate of the plain-mean statistic, for comparison with the official one
            xa = [sum(plain[p["a"]][i][m] for i in plain[p["a"]]) / len(plain[p["a"]]) for m in ms]
            xb = [sum(plain[p["b"]][i][m] for i in plain[p["b"]]) / len(plain[p["b"]]) for m in ms]
            p["tau_b_plain_mean"] = harness.kendall_tau_b(xa, xb)

    pd.DataFrame(out["pairs"]).to_csv(processed / f"tau_pairs_{split}.csv", index=False)
    pd.DataFrame(out["language"]).to_csv(processed / f"tau_language_{split}.csv", index=False)
    (processed / f"agreement_{split}.json").write_text(json.dumps(out, indent=2, default=list) + "\n")
    if figures:
        import figures_rerun
        figures_rerun.draw_all(split, cfg, table, out, raw, lang_tab, boot, charts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="main", choices=["pilot", "main"])
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    out = analyze(a.split, load_config(), figures=not a.no_figures)
    keys = ("n_models", "mean_tau", "min_tau", "verdict", "mean_tau_ci", "verdict_share")
    print(json.dumps({k: out.get(k) for k in keys}, indent=2, default=list))


if __name__ == "__main__":
    main()
