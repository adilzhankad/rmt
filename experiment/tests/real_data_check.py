"""Integration check on the REAL benchmark TSVs with an oracle dummy model (no API, no GPU).

    python experiment/tests/real_data_check.py <scratch dir>

Writes only into <scratch dir> (never logs/ or raw/). For every benchmark it runs the pilot
items through run.run() with a model that answers from the ground truth (and one that
answers badly), checks that item_scores rebuilds the official aggregate, and for
OmniDocBench checks that subset evaluation gives the same page scores as an evaluation of
ALL pages.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402
import item_scores  # noqa: E402
import run  # noqa: E402

BENCHES = ["OCRBench_v2", "CCOCR", "OmniDocBench", "olmOCRBench"]


def oracle_text(bench, row):
    if bench == "OCRBench_v2":
        import ast
        return str(ast.literal_eval(row["answer"])[0])
    if bench == "CCOCR":
        return str(row["answer"])
    if bench == "OmniDocBench":
        page = json.loads(row["answer"])
        parts = []
        for d in sorted(page["layout_dets"], key=lambda d: d.get("order") or 0):
            if d["category_type"] == "table" and d.get("html"):
                parts.append(d["html"])
            elif d["category_type"] == "equation_isolated" and d.get("latex"):
                parts.append("\\[" + d["latex"].strip("$\n ") + "\\]")
            elif d.get("text"):
                parts.append(d["text"])
        return "\n\n".join(parts)
    tests = json.loads(row["answer"])
    return "\n\n".join(t["text"] for t in tests if t["type"] == "present")


class Oracle:
    is_api = False

    def __init__(self, bench, good=True):
        self.bench, self.good, self.by_image = bench, good, None
        self.ds = None

    def set_dump_image(self, f):
        pass

    def generate(self, message, dataset=None):
        if self.by_image is None:
            self.by_image = {}
            for _, r in self.ds.data.iterrows():
                self.by_image[tuple(self.ds.dump_image(r))] = r
        imgs = tuple(x["value"] for x in message if x["type"] == "image")
        text = oracle_text(self.bench, self.by_image[imgs])
        return text if self.good else text[: max(1, len(text) // 3)][::-1]


def run_with(bench, split, model_key, oracle, out):
    orig = run.build_benchmark

    def spy(b, cfg, items):
        ds, md5 = orig(b, cfg, items)
        oracle.ds = ds
        return ds, md5
    run.build_benchmark = spy
    try:
        return run.run(model_key, bench, split, "Adilzhan", model=oracle, out_root=out / "raw",
                       log_path=out / "run_log.csv", usage_dir=None)
    finally:
        run.build_benchmark = orig


def main(out, benches=BENCHES):
    out = Path(out)
    cfg = common.load_config()
    for bench in benches:
        for key, good in (("GPT4o_MINI", True), ("GPT4o_20240806", False)):
            row = run_with(bench, "pilot", key, Oracle(bench, good), out)
            b = cfg["benchmarks"][bench]
            work = out / "raw" / "pilot" / key / bench
            metrics = json.loads((work / "metrics.json").read_text())
            rows, checks = item_scores.score_run(bench, str(work / f"{key}_{b['dataset']}.{b['pred_format']}"),
                                                 metrics, common.read_items(bench, "pilot"))
            print(f"[{bench}] {'oracle' if good else 'bad   '} primary={row['primary_score']:.4f} "
                  f"failed={row['n_failed']} item-rebuild={'OK' if all(c['ok'] for c in checks) else checks}")

    # OmniDocBench: all 981 pages vs the pilot subset, same oracle answers
    import pandas as pd
    full_items = out / "items_full"
    full_items.mkdir(exist_ok=True)
    from sample_items import frame_omnidoc
    fr = frame_omnidoc(common.lmu_root() / "OmniDocBench.tsv")
    fr.assign(stratum=fr["language"]).to_csv(full_items / "OmniDocBench_full.csv", index=False)
    common.ITEMS = full_items
    run_with("OmniDocBench", "full", "GPT4o_MINI", Oracle("OmniDocBench", True), out)
    full_rows, _ = item_scores.items_omnidoc(str(out / "raw/full/GPT4o_MINI/OmniDocBench/GPT4o_MINI_OmniDocBench.tsv"),
                                             common.read_items("OmniDocBench", "full"))
    common.ITEMS = common.ROOT / "items"
    sub_rows, _ = item_scores.items_omnidoc(str(out / "raw/pilot/GPT4o_MINI/OmniDocBench/GPT4o_MINI_OmniDocBench.tsv"),
                                            common.read_items("OmniDocBench", "pilot"))
    full = {r["unit"]: r["score"] for r in full_rows}
    diffs = [abs(r["score"] - full[r["unit"]]) for r in sub_rows]
    print(f"[OmniDocBench] subset vs full: {len(sub_rows)} page x element scores, max |diff| = {max(diffs):.2e}")
    assert max(diffs) < 1e-12
    print(pd.read_csv(out / "run_log.csv")[["model", "benchmark", "split", "n_items", "n_failed", "primary_score",
                                             "duration_s", "notes"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or BENCHES)
