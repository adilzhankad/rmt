"""Per-item scores for every finished run -> processed/item_scores.csv.

    python experiment/item_scores.py [--bench B ...] [--jobs N] [--out-dir DIR]
    python experiment/item_scores.py --merge DIR [DIR ...]     # combine parts into processed/

Each benchmark's headline number is an average of averages, so per-item scores are
kept together with the groups the official aggregate averages over. For every run the
official aggregate is rebuilt from the item rows and compared with the evaluator's own
output (processed/item_scores_check.csv); a mismatch stops the script.

  OCRBench_v2   unit = question; official item scores via process_predictions
  CCOCR         unit = image; the official sub-dataset evaluator run on one image at a time
  OmniDocBench  unit = page x element type; page edit distance from the evaluator's
                saved match files, aggregated the way get_page_split does
  olmOCRBench   unit = test (grouped by page/pdf); the official test objects re-run on
                the saved prediction, with the same baseline tests the evaluator adds
"""
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, RAW, load_config, load_env, read_items  # noqa: E402
from sample_items import OCRB2_GROUP  # noqa: E402

load_env()
OMNI_ELEMENTS = ("text_block", "display_formula", "table", "reading_order")
LANG_KEY = {"english": "EN", "simplified_chinese": "CH"}


# ---------------------------------------------------------------- OCRBench v2
def items_ocrbench(eval_file):
    """Run the official evaluate() on a copy and capture its per-item list."""
    import vlmeval.dataset.utils.ocrbrnch_v2_eval as ev
    from vlmeval.dataset import OCRBench_v2
    captured = {}
    orig = ev.process_predictions

    def spy(pred):
        out = orig(pred)
        captured["items"] = out
        return out
    tmp = Path(tempfile.mkdtemp())
    try:
        f = tmp / Path(eval_file).name
        shutil.copy(eval_file, f)
        ev.process_predictions = spy
        official = OCRBench_v2.evaluate(str(f))
    finally:
        ev.process_predictions = orig
        shutil.rmtree(tmp)
    from vlmeval.smp import load
    idx = [str(i) for i in load(eval_file)["index"]]
    assert len(captured["items"]) == len(idx)
    rows = []
    for i, it in zip(idx, captured["items"]):
        rows.append({"unit": i, "item": i, "group": OCRB2_GROUP[it["type"]], "category": it["type"],
                     "lang": OCRB2_GROUP[it["type"]][:2], "score": float(it["score"]),
                     "ignored": "ignore" in it})
    return rows, {"English Overall Score": official["English Overall Score"],
                  "Chinese Overall Score": official["Chinese Overall Score"]}


def agg_ocrbench(rows):
    by = defaultdict(list)
    for r in rows:
        if not r["ignored"]:
            by[r["group"]].append(r["score"])
    g = {k: sum(v) / len(v) for k, v in by.items()}
    en = [v for k, v in g.items() if k.startswith("en_")]
    cn = [v for k, v in g.items() if k.startswith("cn_")]
    return {"English Overall Score": sum(en) / len(en) if en else float("nan"),
            "Chinese Overall Score": sum(cn) / len(cn) if cn else float("nan")}


# ---------------------------------------------------------------- CC-OCR
CC_METRIC = {"multi_lan_ocr": "macro_f1_score", "multi_scene_ocr": "macro_f1_score",
             "kie": "acc", "doc_parsing": "score"}


def items_ccocr(eval_file, metrics):
    from vlmeval.dataset.image_ccocr import CCOCRDataset
    from vlmeval.smp import load
    df = load(eval_file)
    df["prediction"] = [str(x) for x in df["prediction"]]
    df["sub_dataset_id"] = df["category"].astype(str) + "_" + df["l2-category"].astype(str) + "_" + df["split"].astype(str)
    rows = []
    for _, r in df.iterrows():
        one = pd.DataFrame([r])
        _, summary = CCOCRDataset._evaluate_single_dataset(None, one, r["split"])
        rows.append({"unit": str(r["index"]), "item": str(r["index"]), "track": r["category"],
                     "subset": r["sub_dataset_id"], "score": float(summary[CC_METRIC[r["category"]]])})
    return rows, {k: metrics[k] for k in ("total", "multi_lan_ocr", "multi_scene_ocr", "kie", "doc_parsing")
                  if k in metrics}


def agg_ccocr(rows):
    sub = defaultdict(list)
    for r in rows:
        sub[(r["track"], r["subset"])].append(r["score"])
    track = defaultdict(list)
    for (t, _), v in sub.items():
        track[t].append(sum(v) / len(v))
    out = {t: sum(v) / len(v) for t, v in track.items()}
    out["total"] = sum(out.values()) / len(out)
    return out


# ---------------------------------------------------------------- OmniDocBench
def items_omnidoc(eval_file, items):
    """Page-level edit distance per element type, as get_page_split computes it."""
    lang = {r["image_name"]: r["language"] for r in items}
    index = {r["image_name"]: r["index"] for r in items}
    stem = str(eval_file).rsplit(".", 1)[0]
    rows = []
    for el in OMNI_ELEMENTS:
        samples = json.loads(Path(f"{stem}_end2end_quick_match_{el}_result.json").read_text())
        acc = defaultdict(lambda: [0.0, 0])
        for s in samples:
            if not s.get("metric") or "Edit_dist" not in s["metric"]:
                continue
            img = s["img_id"] if s["img_id"].endswith(".jpg") else "_".join(s["img_id"].split("_")[:-1])
            gt = s["norm_gt"] if s.get("norm_gt") else s["gt"]
            pred = s["norm_pred"] if s.get("norm_pred") else s["pred"]
            up = max(len(gt), len(pred))
            acc[img][0] += s["metric"]["Edit_dist"] * up
            acc[img][1] += up
        for img, (num, den) in acc.items():
            rows.append({"unit": f"{index[img]}:{el}", "item": index[img], "element": el,
                         "lang": LANG_KEY.get(lang[img], lang[img]), "score": num / den})
    e2e = json.loads(Path(f"{stem}_End2End_Evaluation.json").read_text())
    official = {}
    for L, key in (("EN", "language: english"), ("CH", "language: simplified_chinese")):
        vals = [e2e[el]["page"]["Edit_dist"].get(key, float("nan")) for el in OMNI_ELEMENTS]
        official[f"overall_{L}"] = sum(vals) / len(vals)
    return rows, official


def agg_omnidoc(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["lang"], r["element"])].append(r["score"])
    out = {}
    for L in ("EN", "CH"):
        vals = [sum(by[(L, el)]) / len(by[(L, el)]) if by[(L, el)] else float("nan") for el in OMNI_ELEMENTS]
        out[f"overall_{L}"] = sum(vals) / len(vals)
    return out


# ---------------------------------------------------------------- olmOCR-Bench
def items_olmocr(eval_file, metrics):
    from vlmeval.dataset.olmOCRBench.tests import BaselineTest, load_single_test
    df = pd.read_excel(eval_file)
    df["prediction"] = df["prediction"].fillna("").astype(str)   # same as run.evaluate
    rows = []
    for _, r in df.iterrows():
        answer = json.loads(r["answer"]) if isinstance(r["answer"], str) else r["answer"]
        tests = [load_single_test(t) for t in answer]
        page_written = tests[0].page            # the evaluator writes one .md per pdf, page of the first test
        if not any(t.type == "baseline" for t in tests):
            tests.append(BaselineTest(id=f"{tests[0].pdf}_baseline", pdf=tests[0].pdf, page=1, type="baseline"))
            auto_baseline = tests[-1].id
        else:
            auto_baseline = None
        md = r["prediction"]
        for t in tests:
            if t.page != page_written:
                passed = False
            else:
                try:
                    passed = bool(t.run(md)[0])
                except Exception:
                    passed = False
            cat = "baseline" if t.id == auto_baseline else t.pdf.split("/")[0]
            rows.append({"unit": t.id, "item": str(r["index"]), "category": cat, "score": float(passed)})
    return rows, {k: metrics[k] for k in metrics}


def agg_olmocr(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r["score"])
    out = {k: 100 * sum(v) / len(v) for k, v in by.items()}
    out["overall"] = sum(out.values()) / len(out)
    return out


AGG = {"OCRBench_v2": agg_ocrbench, "CCOCR": agg_ccocr, "OmniDocBench": agg_omnidoc, "olmOCRBench": agg_olmocr}
# the evaluators round what they print: olmOCR CSV to 0.1, the rest is exact
TOL = {"OCRBench_v2": 1e-9, "CCOCR": 1e-6, "OmniDocBench": 1e-9, "olmOCRBench": 0.05 + 1e-9}


def score_run(bench, eval_file, metrics, items):
    if bench == "OCRBench_v2":
        rows, official = items_ocrbench(eval_file)
    elif bench == "CCOCR":
        rows, official = items_ccocr(eval_file, metrics)
    elif bench == "OmniDocBench":
        rows, official = items_omnidoc(eval_file, items)
    else:
        rows, official = items_olmocr(eval_file, metrics)
    rebuilt = AGG[bench](rows)
    checks = []
    for k, v in official.items():
        r = rebuilt.get(k, float("nan"))
        ok = (pd.isna(v) and pd.isna(r)) or abs(r - v) <= TOL[bench]
        checks.append({"metric": k, "official": v, "rebuilt": r, "ok": bool(ok)})
    return rows, checks


def find_runs(root=RAW):
    for meta in sorted(Path(root).glob("*/*/*/run_meta.json")):
        m = json.loads(meta.read_text())
        if m.get("primary_score") in (None, ""):
            continue
        yield meta.parent, m


def score_one(job):
    """Rows and checks of one run (a separate process when --jobs > 1)."""
    work_dir, m = job
    cfg = load_config()
    bench, split, model = m["benchmark"], m["split"], m["model"]
    b = cfg["benchmarks"][bench]
    eval_file = work_dir / f"{model}_{b['dataset']}.{b['pred_format']}"
    metrics = json.loads((work_dir / "metrics.json").read_text())
    rows, checks = score_run(bench, str(eval_file), metrics, read_items(bench, split))
    bad = [c for c in checks if not c["ok"]]
    print(f"{split:5s} {model:24s} {bench:13s} {len(rows):6d} rows  check {'OK' if not bad else bad}", flush=True)
    return ([{"split": split, "model": model, "benchmark": bench, **r} for r in rows],
            [{"split": split, "model": model, "benchmark": bench, **c} for c in checks])


def write(all_rows, all_checks, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_dir / "item_scores.csv", index=False)
    pd.DataFrame(all_checks).to_csv(out_dir / "item_scores_check.csv", index=False)
    bad = [c for c in all_checks if not c["ok"]]
    if bad:
        raise SystemExit(f"{len(bad)} aggregates not reproduced from item scores: {bad[:3]}")


def main(root=RAW, out_dir=PROCESSED, cfg=None, benches=None, jobs=1):
    runs = [(d, m) for d, m in find_runs(root) if benches is None or m["benchmark"] in benches]
    if jobs > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(jobs) as pool:
            parts = pool.map(score_one, runs, chunksize=1)
    else:
        parts = [score_one(r) for r in runs]
    all_rows = [r for rows, _ in parts for r in rows]
    all_checks = [c for _, checks in parts for c in checks]
    write(all_rows, all_checks, out_dir)
    return all_rows, all_checks


def merge(parts, out_dir=PROCESSED):
    """Combine item_scores.csv / item_scores_check.csv written by separate runs of this script
    (e.g. olmOCR on Linux, where its maths tests render as in the official evaluation)."""
    parts = [p for p in parts if (Path(p) / "item_scores.csv").stat().st_size > 1]   # a part with no scored run is empty
    rows = pd.concat([pd.read_csv(Path(p) / "item_scores.csv", dtype={"item": str, "unit": str}) for p in parts])
    checks = pd.concat([pd.read_csv(Path(p) / "item_scores_check.csv") for p in parts])
    runs = checks[["split", "model", "benchmark"]].drop_duplicates()
    dup = runs.duplicated().sum() + checks.duplicated(["split", "model", "benchmark", "metric"]).sum()
    if dup:
        raise SystemExit(f"{dup} runs appear in more than one part")
    print(f"merged {len(runs)} runs, {len(rows)} item rows from {len(parts)} parts")
    write(rows.to_dict("records"), checks.to_dict("records"), out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", help="only these benchmarks (default: all)")
    ap.add_argument("--jobs", type=int, default=1, help="runs scored in parallel")
    ap.add_argument("--out-dir", default=str(PROCESSED))
    ap.add_argument("--merge", nargs="+", metavar="DIR", help="combine the outputs of earlier runs instead")
    args = ap.parse_args()
    if args.merge:
        merge(args.merge, args.out_dir)
    else:
        main(out_dir=args.out_dir, benches=args.bench, jobs=args.jobs)
