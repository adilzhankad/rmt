"""Our re-run scores vs the published leaderboards (data/scores_long.csv, collected in Stage 1).

    python experiment/compare_published.py [--split main]

For every benchmark metric that a leaderboard also reports, lists both values per model and,
when at least 3 of our models are on the leaderboard, Kendall tau-b between the two rankings.
Only ranks are compared: the published OCRBench v2 board uses the private set, VLMEvalKit the
public one, and the units differ (0-100 vs 0-1). Writes processed/published_vs_ours.csv/json.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, ROOT, harness, load_config  # noqa: E402

# our model key -> model_id in data/scores_long.csv
PUBLISHED_ID = {"GPT4o_20240806": "gpt-4o", "GPT4o_MINI": "gpt-4o-mini", "Qwen2.5-VL-7B-Instruct": "qwen2.5-vl-7b",
                "InternVL3-8B": "internvl3-8b", "Ovis2-8B": "ovis2-8b", "llava-onevision-qwen2-7b-ov-hf": "llava-ov-7b",
                "MiniCPM-V-4_5": "minicpm-v-4.5-8b"}
# (our benchmark, our metric) -> (leaderboard benchmark, split, metric prefix)
PAIRS = {("OCRBench_v2", "English Overall Score"): ("OCRBench v2", "English (private)", "Average score"),
         ("OCRBench_v2", "Chinese Overall Score"): ("OCRBench v2", "Chinese (private)", "Average score"),
         ("CCOCR", "total"): ("CC-OCR", "Total", "Track score"),
         ("OmniDocBench", "overall_EN"): ("OmniDocBench", "English pages", "Overall edit dist"),
         ("OmniDocBench", "overall_CH"): ("OmniDocBench", "Chinese pages", "Overall edit dist")}


def compare(split="main", scores=PROCESSED / "scores_rerun.csv", published=ROOT / "data/scores_long.csv",
            out_dir=PROCESSED, cfg=None):
    cfg = cfg or load_config()
    ours = pd.read_csv(scores)
    ours = ours[(ours["split"] == split) & ours["model"].isin(cfg["models"])]
    pub = pd.read_csv(published)
    pub = pub[pub["status"] == "ok"]
    rows, summary = [], []
    for (bench, metric), (pbench, psplit, pmetric) in PAIRS.items():
        o = ours[(ours["benchmark"] == bench) & (ours["metric"] == metric)].set_index("model")["value"]
        p = pub[(pub["benchmark"] == pbench) & (pub["split"] == psplit) & pub["metric"].str.startswith(pmetric)]
        p = p.groupby("model_id")["value"].first()
        shared = [m for m in o.index if PUBLISHED_ID.get(m) in p.index]
        for m in shared:
            rows.append({"benchmark": bench, "metric": metric, "model": m, "published_model_id": PUBLISHED_ID[m],
                         "ours": o[m], "published": p[PUBLISHED_ID[m]], "leaderboard_split": psplit})
        tau = (harness.kendall_tau_b([o[m] for m in shared], [p[PUBLISHED_ID[m]] for m in shared])
               if len(shared) >= 3 else None)
        summary.append({"benchmark": bench, "metric": metric, "leaderboard_split": psplit, "n_shared": len(shared),
                        "kendall_tau_b": None if tau is None else round(tau, 3)})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "published_vs_ours.csv", index=False)
    (out_dir / "published_vs_ours.json").write_text(json.dumps(summary, indent=2) + "\n")
    for s in summary:
        print(f"{s['benchmark']:13} {s['metric']:22} shared {s['n_shared']}  tau-b {s['kendall_tau_b']}")
    return rows, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="main", choices=["pilot", "main"])
    compare(ap.parse_args().split)
