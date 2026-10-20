"""logs/run_log.csv + raw/rerun/**/metrics.json -> processed/scores_rerun.csv (long format).

    python experiment/collect.py

For every model x benchmark x split the LAST logged run with a score is used (a re-run
retries failed API answers, so it supersedes earlier rows). Runs that crashed or whose
answers all failed have no score and are listed in processed/missing_runs.csv.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, RAW, RUN_LOG, load_config  # noqa: E402

SECONDARY_DIRECTION = {"Chinese Overall Score": "higher", "overall_CH": "lower",
                       "multi_scene_ocr": "higher", "multi_lan_ocr": "higher", "kie": "higher",
                       "doc_parsing": "higher"}


def main(run_log=RUN_LOG, raw=RAW, out_dir=PROCESSED, cfg=None):
    cfg = cfg or load_config()
    log = pd.read_csv(run_log, dtype={"primary_score": str})
    log = log[log["model"].isin(cfg["models"])].copy()   # piloted-then-dropped models stay in the log only
    log["has_score"] = log["primary_score"].fillna("").str.strip() != ""
    rows, missing = [], []
    for (model, bench, split), g in log.groupby(["model", "benchmark", "split"], sort=True):
        ok = g[g["has_score"]]
        if ok.empty:
            last = g.iloc[-1]
            missing.append({"model": model, "benchmark": bench, "split": split,
                            "last_timestamp_utc": last["timestamp_utc"], "notes": last["notes"]})
            continue
        last = ok.iloc[-1]
        b = cfg["benchmarks"][bench]
        metrics = json.loads((Path(raw) / split / model / bench / "metrics.json").read_text())
        assert abs(metrics[b["primary_metric"]] - float(last["primary_score"])) < 1e-9, (model, bench, split)
        for metric in [b["primary_metric"], *b["secondary_metrics"]]:
            if metric not in metrics:
                continue
            rows.append({"model": model, "benchmark": bench, "split": split, "metric": metric,
                         "is_primary": metric == b["primary_metric"], "value": metrics[metric],
                         "direction": b["direction"] if metric == b["primary_metric"] else SECONDARY_DIRECTION[metric],
                         "n_items": int(last["n_items"]), "n_failed": int(last["n_failed"]),
                         "timestamp_utc": last["timestamp_utc"], "operator": last["operator"]})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "scores_rerun.csv", index=False)
    pd.DataFrame(missing, columns=["model", "benchmark", "split", "last_timestamp_utc", "notes"]).to_csv(
        out_dir / "missing_runs.csv", index=False)
    print(f"{len(rows)} score rows, {len(missing)} model x benchmark x split without a score")
    return rows, missing


if __name__ == "__main__":
    main()
