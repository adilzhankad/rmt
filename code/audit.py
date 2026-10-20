"""Audit of the published-score dataset (Stage 1).

Usage: python -I audit.py <scores_long.csv> <out_dir>
Writes audit.txt and overlap.csv; prints the same summary.
"""
import collections, csv, itertools, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from harness import kendall_tau_b, oriented, MIN_SHARED_MODELS

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf8")))
out = Path(sys.argv[2]); out.mkdir(exist_ok=True)
lines = []
say = lambda s="": (lines.append(s), print(s))

# Primary split per benchmark, fixed in advance (one number per model per benchmark).
PRIMARY = {"OCRBench v2": ("private set, 2025-09 release", "English (private)"),
           "CC-OCR": ("v1 (Nov 2024 runs)", "Total"),
           "OmniDocBench": ("v1.0 (CVPR 2025)", "English pages"),
           "olmOCR-Bench": ("README leaderboard", "Overall")}

say(f"Rows: {len(rows)}   Unique model labels: {len({r['model_id'] for r in rows})}")
say("\nRows per benchmark split:")
for k, v in collections.Counter((r["benchmark"], r["benchmark_version"], r["split"]) for r in rows).items():
    say(f"  {v:>3}  {k[0]} | {k[1]} | {k[2]}")
say("\nStatus: " + str(dict(collections.Counter(r["status"] for r in rows))))
bad = [r for r in rows if r["status"] != "ok"]
for r in bad:
    say(f"  not usable: {r['benchmark']} {r['split']} {r['model_raw']} = {r['value']} ({r['status']})")
dup = [k for k, v in collections.Counter((r["benchmark"], r["benchmark_version"], r["split"], r["model_id"]) for r in rows).items() if v > 1]
say(f"Duplicate model rows inside one split: {len(dup)} {dup}")
stated = collections.Counter((r["benchmark"], r["model_version_stated"] != "not stated") for r in rows)
say("\nModel version stated (rows): " + ", ".join(f"{b}: {stated[(b, True)]}/{stated[(b, True)] + stated[(b, False)]}" for b in PRIMARY))

table = {}
for b, (ver, split) in PRIMARY.items():
    table[b] = {r["model_id"]: oriented(float(r["value"]), r["direction"]) for r in rows
                if r["benchmark"] == b and r["benchmark_version"] == ver and r["split"] == split and r["status"] == "ok"}
say("\nModels per benchmark (primary split): " + ", ".join(f"{b}: {len(m)}" for b, m in table.items()))
allfour = set.intersection(*(set(m) for m in table.values()))
say(f"Models on all four benchmarks: {len(allfour)} {sorted(allfour)}")

with open(out / "overlap.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["benchmark_a", "benchmark_b", "shared_models", "estimable", "models"])
    say(f"\nShared models per benchmark pair (estimable if >= {MIN_SHARED_MODELS}):")
    for a, b in itertools.combinations(table, 2):
        s = sorted(set(table[a]) & set(table[b]))
        w.writerow([a, b, len(s), len(s) >= MIN_SHARED_MODELS, "; ".join(s)])
        say(f"  {a} x {b}: {len(s)}  {s}")

say("\nHarness check on large within-benchmark splits (language factor, not the RQ):")
for b, ver, s1, s2 in [("OCRBench v2", "private set, 2025-09 release", "English (private)", "Chinese (private)"),
                       ("OmniDocBench", "v1.0 (CVPR 2025)", "English pages", "Chinese pages")]:
    get = lambda s: {r["model_id"]: oriented(float(r["value"]), r["direction"]) for r in rows
                     if r["benchmark"] == b and r["benchmark_version"] == ver and r["split"] == s and r["status"] == "ok"}
    x, y = get(s1), get(s2); m = sorted(set(x) & set(y))
    say(f"  {b}: {s1} vs {s2}, n = {len(m)}, Kendall tau-b = {kendall_tau_b([x[k] for k in m], [y[k] for k in m]):.2f}")

(out / "audit.txt").write_text("\n".join(lines), encoding="utf8")
