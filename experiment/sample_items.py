"""Stratified item samples for every benchmark (seed from harness.SEED).

    python experiment/sample_items.py            # writes items/<bench>_{main,pilot}.csv

Main sample: harness.ITEMS_PER_BENCHMARK items. Pilot: 20 items drawn from the items
NOT in the main sample, so the pilot is an independent check.

Strata (chosen so that the benchmark's own aggregate is defined on the subset):
  OCRBench_v2   main: task type (30 types, >=1 each);  pilot: capability group (8 EN + 5 ZH)
  CCOCR         100 per track; inside a track proportional by sub-dataset (>=1 each);
                pilot: 5 per track, simple random inside the track
  OmniDocBench  English and Chinese pages only (overall_EN / overall_CH ignore mixed pages);
                strata = language x data_source; every language must keep at least one
                page with text, a display formula and a table, else redraw (logged)
  olmOCRBench   source category (7 jsonl files, >=1 each)
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ITEMS, harness, items_path, lmu_root, load_config, md5_file  # noqa: E402

OCRB2_GROUP = {  # copied from ocrbench_v2_aggregate_accuracy (VLMEvalKit, pinned commit)
    "text recognition en": "en_text_recognition", "fine-grained text recognition en": "en_text_recognition",
    "full-page OCR en": "en_text_recognition",
    "text grounding en": "en_text_detection", "VQA with position en": "en_text_detection",
    "text spotting en": "en_text_spotting",
    "key information extraction en": "en_relationship_extraction",
    "key information mapping en": "en_relationship_extraction",
    "document parsing en": "en_element_parsing", "chart parsing en": "en_element_parsing",
    "table parsing en": "en_element_parsing", "formula recognition en": "en_element_parsing",
    "math QA en": "en_mathematical_calculation", "text counting en": "en_mathematical_calculation",
    "document classification en": "en_visual_text_understanding", "cognition VQA en": "en_visual_text_understanding",
    "diagram QA en": "en_visual_text_understanding",
    "reasoning VQA en": "en_knowledge_reasoning", "science QA en": "en_knowledge_reasoning",
    "APP agent en": "en_knowledge_reasoning", "ASCII art classification en": "en_knowledge_reasoning",
    "full-page OCR cn": "cn_text_recognition",
    "key information extraction cn": "cn_relationship_extraction",
    "handwritten answer extraction cn": "cn_relationship_extraction",
    "document parsing cn": "cn_element_parsing", "table parsing cn": "cn_element_parsing",
    "formula recognition cn": "cn_element_parsing",
    "cognition VQA cn": "cn_visual_text_understanding",
    "reasoning VQA cn": "cn_knowledge_reasoning", "text translation cn": "cn_knowledge_reasoning",
}
OMNI_TEXT = {"text_block", "title", "code_txt", "code_txt_caption", "reference"}  # scored text classes
OMNI_LANGS = ("english", "simplified_chinese")


def allocate(sizes, n, min_each=1):
    """Largest-remainder allocation of n over strata, at least min_each per stratum."""
    total = sum(sizes.values())
    exact = {k: n * s / total for k, s in sizes.items()}
    q = {k: min(sizes[k], max(min_each, int(exact[k]))) for k in sizes}
    keys = sorted(sizes)
    while sum(q.values()) < n:
        k = max((k for k in keys if q[k] < sizes[k]), key=lambda k: (exact[k] - q[k], k))
        q[k] += 1
    while sum(q.values()) > n:
        k = min((k for k in keys if q[k] > min_each), key=lambda k: (exact[k] - q[k], k))
        q[k] -= 1
    return q


def draw(frame, stratum_col, quotas, rng, exclude=()):
    """frame: DataFrame with 'index' and stratum_col. Returns sampled indices per stratum."""
    out = []
    for s in sorted(quotas):
        pool = sorted(i for i in frame.loc[frame[stratum_col] == s, "index"] if i not in exclude)
        out += rng.sample(pool, quotas[s])
    return out


def read_meta(tsv, columns):
    return pd.read_csv(tsv, sep="\t", usecols=columns, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------- per benchmark frames
def frame_ocrbench(tsv):
    df = read_meta(tsv, ["index", "category"])
    unknown = set(df["category"]) - set(OCRB2_GROUP)
    assert not unknown, f"unknown OCRBench v2 categories: {unknown}"
    df["group"] = df["category"].map(OCRB2_GROUP)
    return df


def frame_ccocr(tsv):
    df = read_meta(tsv, ["index", "image_name", "category", "l2-category", "split"])
    df["track"] = df["category"]
    df["subset"] = df["category"] + "_" + df["l2-category"] + "_" + df["split"]  # = sub_dataset_id
    return df


def frame_omnidoc(tsv):
    df = read_meta(tsv, ["index", "answer"])
    rows = []
    for idx, ans in zip(df["index"], df["answer"]):
        page = json.loads(ans)
        attr = page["page_info"]["page_attribute"]
        cats = {d["category_type"] for d in page["layout_dets"]}
        rows.append({"index": idx, "image_name": Path(page["page_info"]["image_path"]).name,
                     "language": attr.get("language", ""), "data_source": attr.get("data_source", ""),
                     "has_text": bool(cats & OMNI_TEXT), "has_formula": "equation_isolated" in cats,
                     "has_table": "table" in cats})
    out = pd.DataFrame(rows)
    out["lang_source"] = out["language"] + "|" + out["data_source"]
    return out


def frame_olmocr(tsv):
    df = read_meta(tsv, ["index", "answer"])
    tests = [json.loads(a) for a in df["answer"]]
    df["pdf_name"] = [t[0]["pdf"] for t in tests]
    df["category"] = [t[0]["pdf"].split("/")[0] for t in tests]
    df["n_tests"] = [len(t) for t in tests]
    return df.drop(columns=["answer"])


def omni_ok(sample):
    """Every language keeps text, display formula and table pages (else the evaluator gives NaN/KeyError)."""
    for lang in OMNI_LANGS:
        s = sample[sample["language"] == lang]
        if not (len(s) and s["has_text"].any() and s["has_formula"].any() and s["has_table"].any()):
            return False
    return True


# ---------------------------------------------------------------- sampling plans
def plan(bench, frame, n_main, n_pilot, rng):
    notes = {}
    if bench == "OCRBench_v2":
        main = draw(frame, "category", allocate(frame["category"].value_counts().to_dict(), n_main), rng)
        rest = frame[~frame["index"].isin(main)]
        pilot = draw(rest, "group", allocate(rest["group"].value_counts().to_dict(), n_pilot), rng)
        return main, pilot, "category", "group", notes
    if bench == "CCOCR":
        tracks = sorted(frame["track"].unique())
        per_main, per_pilot = n_main // len(tracks), n_pilot // len(tracks)
        assert per_main * len(tracks) == n_main and per_pilot * len(tracks) == n_pilot
        main, pilot = [], []
        for t in tracks:
            ft = frame[frame["track"] == t]
            main += draw(ft, "subset", allocate(ft["subset"].value_counts().to_dict(), per_main), rng)
        taken = set(main)
        for t in tracks:
            pool = sorted(i for i in frame.loc[frame["track"] == t, "index"] if i not in taken)
            pilot += rng.sample(pool, per_pilot)
        return main, pilot, "subset", "track", notes
    if bench == "OmniDocBench":
        fr = frame[frame["language"].isin(OMNI_LANGS)]
        notes["excluded_pages_other_language"] = int((~frame["language"].isin(OMNI_LANGS)).sum())
        q_main = allocate(fr["lang_source"].value_counts().to_dict(), n_main)
        for attempt in range(1, 101):
            main = draw(fr, "lang_source", q_main, rng)
            if omni_ok(fr[fr["index"].isin(main)]):
                break
        notes["main_draws"] = attempt
        rest = fr[~fr["index"].isin(main)]
        q_pilot = allocate(rest["language"].value_counts().to_dict(), n_pilot)
        for attempt in range(1, 1001):
            pilot = draw(rest, "language", q_pilot, rng)
            if omni_ok(rest[rest["index"].isin(pilot)]):
                break
        notes["pilot_draws"] = attempt
        assert omni_ok(fr[fr["index"].isin(main)]) and omni_ok(rest[rest["index"].isin(pilot)])
        return main, pilot, "lang_source", "language", notes
    if bench == "olmOCRBench":
        q = allocate(frame["category"].value_counts().to_dict(), n_main)
        main = draw(frame, "category", q, rng)
        rest = frame[~frame["index"].isin(main)]
        pilot = draw(rest, "category", allocate(rest["category"].value_counts().to_dict(), n_pilot), rng)
        return main, pilot, "category", "category", notes
    raise KeyError(bench)


FRAMES = {"OCRBench_v2": frame_ocrbench, "CCOCR": frame_ccocr,
          "OmniDocBench": frame_omnidoc, "olmOCRBench": frame_olmocr}


def write_items(bench, split, frame, ids, stratum_col, tsv_md5):
    sel = frame.set_index("index").loc[ids].reset_index()
    sel = sel.sort_values("index", key=lambda s: s.astype(int)).reset_index(drop=True)
    sel.insert(1, "stratum", sel[stratum_col])
    sel["tsv_md5"] = tsv_md5
    path = items_path(bench, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


def main(benches=None):
    cfg = load_config()
    n_main, n_pilot = cfg["items"]["main"], cfg["items"]["pilot"]
    summary = {}
    for bench, b in cfg["benchmarks"].items():
        if benches and bench not in benches:
            continue
        tsv = lmu_root() / f"{b['dataset']}.tsv"
        digest = md5_file(tsv)
        assert digest == b["md5"], f"{tsv}: md5 {digest} != pinned {b['md5']}"
        frame = FRAMES[bench](tsv)
        rng = random.Random(f"{harness.SEED}:{bench}")
        main_ids, pilot_ids, s_main, s_pilot, notes = plan(bench, frame, n_main, n_pilot, rng)
        assert len(set(main_ids)) == n_main and len(set(pilot_ids)) == n_pilot
        assert not set(main_ids) & set(pilot_ids)
        for split, ids, col in (("main", main_ids, s_main), ("pilot", pilot_ids, s_pilot)):
            p = write_items(bench, split, frame, ids, col, digest)
            print(f"{bench:13s} {split:5s} {len(ids):4d} items, {frame[col].nunique():3d} strata in frame -> {p}")
        summary[bench] = {"tsv_md5": digest, "frame_rows": len(frame), "main": n_main, "pilot": n_pilot,
                          "main_strata": s_main, "pilot_strata": s_pilot, **notes}
    out = ITEMS / "sampling_summary.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(summary)
    out.write_text(json.dumps(old, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
