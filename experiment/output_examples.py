"""Output examples of the main run -> processed/output_examples_main.csv.

    python experiment/output_examples.py

Five answers of Qwen2.5-VL-7B-Instruct (the model that leads two benchmarks), one or two per
benchmark, chosen to show what a scored answer looks like: correct and scored as correct, and
correct but scored as wrong. For each item the file keeps the task and ground truth from the
benchmark TSV (LMUData/, not committed because of its size), the start of the exact answer from
predictions.jsonl and the item score from processed/item_scores.csv, so the report table can be
checked without the 6 GB of benchmark data.
"""
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROCESSED, RAW, ROOT  # noqa: E402

MODEL = "Qwen2.5-VL-7B-Instruct"
EXAMPLES = [("OCRBench_v2", "1030"), ("OCRBench_v2", "85"), ("CCOCR", "2918"),
            ("OmniDocBench", "153"), ("olmOCRBench", "1177")]
CHARS = 400


def main():
    csv.field_size_limit(sys.maxsize)
    lmu = Path(os.environ.get("LMUData", ROOT / "LMUData"))
    scores = pd.read_csv(PROCESSED / "item_scores.csv", dtype={"item": str, "unit": str}, low_memory=False)
    scores = scores[(scores.split == "main") & (scores.model == MODEL)]
    rows = []
    for bench, item in EXAMPLES:
        tsv = pd.read_csv(lmu / f"{bench}.tsv", sep="\t", dtype=str, usecols=lambda c: c in ("index", "question", "answer"))
        src = tsv.set_index("index").loc[item]
        preds = (RAW / "main" / MODEL / bench / "predictions.jsonl").read_text().splitlines()
        answer = next(str(j["prediction"]) for j in map(json.loads, preds) if str(j["index"]) == item)
        s = scores[(scores.benchmark == bench) & (scores["item"] == item)]
        rows.append({"model": MODEL, "benchmark": bench, "item": item,
                     "question": "" if pd.isna(src.get("question")) else src["question"][:CHARS],
                     "ground_truth": str(src["answer"])[:CHARS], "answer_start": answer[:CHARS],
                     "answer_chars": len(answer),
                     "scores": "; ".join(f"{u}={v:.3f}" for u, v in zip(s.unit, s.score))})
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED / "output_examples_main.csv", index=False)
    print(df.drop(columns=["question", "ground_truth", "answer_start"]).to_string(index=False))


if __name__ == "__main__":
    main()
