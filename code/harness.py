"""Evaluation harness: rules fixed before data collection + the agreement metric.

Kendall's tau-b between two rankings of the same n models:

    tau_b = (C - D) / sqrt((n0 - T1) * (n0 - T2)),   n0 = n(n-1)/2

C = concordant pairs, D = discordant pairs, T1/T2 = pairs tied in ranking 1/2.
Range -1..1, HIGHER = more agreement. Without ties tau_b = (C - D) / n0.
"""
import itertools, math, random

# ---------------- rules fixed in advance (do not change after data collection) --
SEED = 2026                 # item subsampling and bootstrap
ITEMS_PER_BENCHMARK = 400   # stratified subsample per benchmark (Stage 2)
BOOTSTRAP = 1000            # resamples of items for 95% confidence intervals
MIN_SHARED_MODELS = 5       # a benchmark pair with fewer shared models is not estimated
HIGH_AGREEMENT = 0.70       # success criterion (Qian et al., 2026)
LOW_AGREEMENT = 0.40
TASK_TYPE = {"OCRBench v2": "text reading", "CC-OCR": "text reading",
             "OmniDocBench": "document parsing", "olmOCR-Bench": "document parsing"}
DIRECTION = {"higher": 1, "lower": -1}   # edit distance: lower is better


def kendall_tau_b(x, y):
    """x, y: scores of the same models in the same order (higher = better)."""
    n = len(x)
    if n < 2:
        return float("nan")
    c = d = t1 = t2 = 0
    for i, j in itertools.combinations(range(n), 2):
        a, b = x[i] - x[j], y[i] - y[j]
        if a == 0 and b == 0:
            t1 += 1; t2 += 1
        elif a == 0:
            t1 += 1
        elif b == 0:
            t2 += 1
        elif (a > 0) == (b > 0):
            c += 1
        else:
            d += 1
    n0 = n * (n - 1) / 2
    den = math.sqrt((n0 - t1) * (n0 - t2))
    return (c - d) / den if den else float("nan")


def oriented(value, direction):
    """Flip lower-is-better metrics so that higher always means better."""
    return value * DIRECTION[direction]


def pairwise(table):
    """table: {benchmark: {model: oriented score}} -> list of (A, B, n, tau, shared)."""
    out = []
    for a, b in itertools.combinations(table, 2):
        shared = sorted(set(table[a]) & set(table[b]))
        if len(shared) < MIN_SHARED_MODELS:
            out.append((a, b, len(shared), None, shared)); continue
        tau = kendall_tau_b([table[a][m] for m in shared], [table[b][m] for m in shared])
        out.append((a, b, len(shared), tau, shared))
    return out


def verdict(mean_tau, min_tau):
    """Success criterion stated before data collection (Task 3)."""
    if mean_tau >= HIGH_AGREEMENT and min_tau >= LOW_AGREEMENT:
        return "rankings agree"
    if mean_tau < LOW_AGREEMENT:
        return "rankings do not agree"
    return "partial agreement"


def bootstrap_tau(items, bench_a, bench_b, models, reps=BOOTSTRAP, seed=SEED):
    """Stage 2: items = {benchmark: {item_id: {model: score}}}.
    Resample items within each benchmark, recompute model means and tau.
    Returns the 2.5th and 97.5th percentiles."""
    rng = random.Random(seed)
    def means(bench, ids):
        return [sum(items[bench][i][m] for i in ids) / len(ids) for m in models]
    ia, ib = list(items[bench_a]), list(items[bench_b])
    taus = sorted(kendall_tau_b(means(bench_a, [rng.choice(ia) for _ in ia]),
                                means(bench_b, [rng.choice(ib) for _ in ib])) for _ in range(reps))
    return taus[int(0.025 * reps)], taus[int(0.975 * reps) - 1]


if __name__ == "__main__":
    # Smoke test 1: reproduce Pilot A of Assignment 2 (CC-OCR tracks, 5 models).
    cc = {"Multi-scene": [83.25, 77.95, 76.92, 76.40, 72.87],
          "Multilingual": [78.97, 71.14, 46.57, 73.44, 65.68],
          "Doc parsing": [62.37, 53.78, 35.33, 53.30, 47.79],
          "KIE": [67.28, 71.76, 61.60, 63.45, 64.58]}
    got = {f"{a} vs {b}": round(kendall_tau_b(cc[a], cc[b]), 2) for a, b in itertools.combinations(cc, 2)}
    expected = [0.40, 0.60, 0.20, 0.80, 0.40, 0.60]
    assert list(got.values()) == expected, got
    print("Pilot A reproduced:", got)
    # Smoke test 2: agreement with scipy on random data with ties.
    try:
        from scipy.stats import kendalltau
        r = random.Random(1)
        for _ in range(200):
            n = r.randint(3, 12)
            x = [r.randint(0, 5) for _ in range(n)]; y = [r.randint(0, 5) for _ in range(n)]
            s = kendalltau(x, y).statistic
            m = kendall_tau_b(x, y)
            assert (math.isnan(s) and math.isnan(m)) or abs(s - m) < 1e-12
        print("Matches scipy.stats.kendalltau (tau-b) on 200 random cases")
    except ImportError:
        print("scipy not installed: skipped cross-check")
    # Smoke test 3: bootstrap runs on synthetic item-level data.
    r = random.Random(3); models = ["m1", "m2", "m3", "m4", "m5", "m6"]
    syn = {b: {i: {m: r.random() + k * 0.05 for k, m in enumerate(models)} for i in range(60)} for b in ("A", "B")}
    print("Bootstrap 95% CI on synthetic data:", bootstrap_tau(syn, "A", "B", models, reps=200))
