"""Run one model on one benchmark subset and log it.

    python experiment/run.py --model GPT4o_MINI --bench CCOCR --split pilot --operator Adilzhan

Steps: build the VLMEvalKit dataset (md5 checked) -> keep only the saved item ids ->
build the model with the common max_tokens / temperature -> infer_data_job (resumes from
checkpoints, retries 'Failed to obtain answer' answers) -> official evaluator ->
raw/rerun/<split>/<model>/<bench>/ + one row in logs/run_log.csv.

Nothing here invents a score: if inference crashes, the row is logged with the error
and no score; if every answer failed, the evaluator is not run.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FAIL_MSG, RAW, ROOT, RUN_LOG, USAGE_DIR, append_run_log, install_redaction,  # noqa: E402
                    load_config, load_env, lmu_root, md5_file, read_items)

load_env()
os.environ.setdefault("LMUData", str(lmu_root()))


# ---------------------------------------------------------------- environment facts
def vlmevalkit_commit():
    import vlmeval
    repo = Path(vlmeval.__file__).resolve().parent.parent
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def system_info():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------- dataset
def build_benchmark(bench, cfg, items):
    """VLMEvalKit dataset restricted (in memory) to the saved item ids, in dataset order."""
    from vlmeval.dataset import build_dataset
    b = cfg["benchmarks"][bench]
    ds = build_dataset(b["dataset"])
    assert ds is not None, f"VLMEvalKit could not build {b['dataset']}"
    tsv_md5 = md5_file(ds.data_path)
    assert tsv_md5 == b["md5"], f"{ds.data_path}: md5 {tsv_md5} != pinned {b['md5']}"
    want = [str(r["index"]) for r in items]
    keep = ds.data["index"].astype(str).isin(set(want))
    ds.data = ds.data[keep].reset_index(drop=True)
    missing = set(want) - set(ds.data["index"].astype(str))
    assert not missing, f"{len(missing)} item ids not in {b['dataset']}: {sorted(missing)[:5]}"
    assert len(ds.data) == len(want)
    return ds, tsv_md5


# ---------------------------------------------------------------- model
def build_model(key, cfg):
    """VLMEvalKit model with the common generation settings (see config.yaml)."""
    from vlmeval import api
    from vlmeval.config import supported_VLM
    m = cfg["models"][key]
    ov = {k: (cfg["max_tokens"] if v == "MAX_TOKENS" else v) for k, v in m.get("overrides", {}).items()}
    if "replace_with" in m:
        r = dict(m["replace_with"])
        model = getattr(api, r.pop("wrapper"))(**{**r, **ov})
    else:
        model = supported_VLM[key](**ov)
    if m.get("provider") == "anthropic":
        # Claude_Wrapper stores temperature but never sends it; pass it with the request.
        model.default_kwargs["temperature"] = cfg["temperature"]
    if cfg.get("prompt") == "benchmark" and hasattr(model, "use_custom_prompt"):
        model.use_custom_prompt = lambda dataset=None: False   # infer_data then uses dataset.build_prompt
    return model


class UsageMeter:
    """Counts API calls and tokens (including retries) so cost can be estimated from the pilot."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = 0
        self.tokens = {"input": 0, "output": 0, "thinking": 0}
        self.finish = {}

    def add(self, inp=0, out=0, think=0, finish=None):
        with self.lock:
            self.calls += 1
            self.tokens["input"] += int(inp or 0)
            self.tokens["output"] += int(out or 0)
            self.tokens["thinking"] += int(think or 0)
            self.finish[str(finish)] = self.finish.get(str(finish), 0) + 1

    def attach(self, model, provider):
        if provider in ("openai", "anthropic"):
            inner = model.generate_inner

            def wrapped(*a, **k):
                ret = inner(*a, **k)
                try:
                    j = ret[2].json()
                    u = j.get("usage", {})
                    if provider == "openai":
                        det = u.get("completion_tokens_details") or {}
                        self.add(u.get("prompt_tokens"), u.get("completion_tokens"), det.get("reasoning_tokens"),
                                 (j.get("choices") or [{}])[0].get("finish_reason"))
                    else:
                        self.add(u.get("input_tokens"), u.get("output_tokens"), 0, j.get("stop_reason"))
                except Exception:
                    self.add(finish="unparsed")
                return ret
            model.generate_inner = wrapped
        elif provider == "google":
            gen = model.client.models.generate_content

            def wrapped(*a, **k):
                try:
                    resp = gen(*a, **k)
                except Exception as err:
                    self.add(finish=f"error:{type(err).__name__}")
                    raise
                u = resp.usage_metadata
                fr = resp.candidates[0].finish_reason if resp.candidates else None
                self.add(getattr(u, "prompt_token_count", 0), getattr(u, "candidates_token_count", 0),
                         getattr(u, "thoughts_token_count", 0), getattr(fr, "name", fr))
                return resp
            model.client.models.generate_content = wrapped

    def as_dict(self):
        return {"calls": self.calls, "tokens": self.tokens, "finish_reasons": self.finish}


# ---------------------------------------------------------------- evaluation
def evaluate(bench, ds, eval_file, work_dir):
    """Run the benchmark's own evaluator on the subset prediction file. Returns {metric: value}."""
    if bench == "OCRBench_v2":
        return {k: float(v) for k, v in ds.evaluate(eval_file).items()}
    if bench == "CCOCR":
        return {k: float(v) for k, v in ds.evaluate(eval_file).items()}
    if bench == "OmniDocBench":
        # The evaluator matches predictions to ground truth BY POSITION in self.data_path,
        # so it gets a TSV with exactly our items in the prediction order.
        from vlmeval.dataset.OmniDocBench.omnidocbench import end2end_evaluator
        from vlmeval.smp import dump, load
        pred = load(eval_file)
        sub = ds.data.set_index(ds.data["index"].astype(str)).loc[pred["index"].astype(str), ["index", "answer"]]
        subset_tsv = Path(work_dir) / "OmniDocBench_subset_gt.tsv"
        dump(sub.reset_index(drop=True), str(subset_tsv))
        # Same call as OmniDocBench.evaluate() makes for its returned metrics; the separate
        # table-attribute report it also builds is not used and fails on small subsets.
        df = end2end_evaluator(eval_file, str(subset_tsv)).score()
        return {k: float(v) for k, v in df.iloc[0].items() if k.endswith(("_EN", "_CH")) and v != "-"}
    if bench == "olmOCRBench":
        # pd.read_excel turns an empty answer into NaN and the evaluator then crashes in
        # f.write(NaN); read it back as an empty string instead (scored like any empty page).
        import importlib  # (vlmeval.dataset.olmOCRBench is shadowed by the class of that name)
        olm = importlib.import_module("vlmeval.dataset.olmOCRBench.evaluator")
        real = olm.pd.read_excel

        def read_excel_empty_as_text(*a, **k):
            df = real(*a, **k)
            df["prediction"] = df["prediction"].fillna("").astype(str)
            return df
        olm.pd = type("pd_shim", (), {"read_excel": staticmethod(read_excel_empty_as_text)})
        try:
            ds.evaluate(eval_file)
        finally:
            olm.pd = pd
        summ = pd.read_csv(Path(eval_file).parent / "olmOCRBench_eval_summary.csv")
        return {str(t): float(s) for t, s in zip(summ["type"], summ["score"])}
    raise KeyError(bench)


# ---------------------------------------------------------------- one run
def run(model_key, bench, split, operator, cfg=None, model=None, out_root=RAW, log_path=RUN_LOG,
        usage_dir=USAGE_DIR, api_nproc=4, note=""):
    cfg = cfg or load_config()
    b = cfg["benchmarks"][bench]
    items = read_items(bench, split)
    work_dir = Path(out_root) / split / model_key / bench
    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PRED_FORMAT"] = b["pred_format"]
    t0 = time.time()
    row = {"operator": operator, "model": model_key, "benchmark": bench, "split": split,
           "n_items": len(items), "max_tokens": cfg["max_tokens"], "temperature": cfg["temperature"]}
    meter = UsageMeter()
    try:
        from vlmeval.inference import infer_data_job
        from vlmeval.smp import get_pred_file_path
        row["vlmevalkit_commit"] = vlmevalkit_commit()
        ds, row["tsv_md5"] = build_benchmark(bench, cfg, items)
        install_redaction()  # before the model is built: the GPT wrapper logs its key in __init__
        if model is None:
            model = build_model(model_key, cfg)
            meter.attach(model, cfg["models"][model_key].get("provider"))
        infer_data_job(model, work_dir=str(work_dir), model_name=model_key, dataset=ds, api_nproc=api_nproc)
        eval_file = get_pred_file_path(str(work_dir), model_key, b["dataset"], use_env_format=True)
        preds = pd.DataFrame({"index": ds.data["index"].astype(str), "prediction": ds.data["prediction"]})
        with open(work_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for r in preds.itertuples(index=False):
                f.write(json.dumps({"index": r.index, "prediction": r.prediction}, ensure_ascii=False) + "\n")
        n_failed = int(preds["prediction"].astype(str).str.contains(FAIL_MSG, regex=False).sum())
        row["n_failed"] = n_failed
        notes = [note] if note else []
        saved = pd.read_excel(eval_file) if eval_file.endswith(".xlsx") else None
        if saved is not None:
            kept = saved["prediction"].fillna("").astype(str)   # an empty cell reads back as NaN
            cut = sum(a != str(p) for a, p in zip(kept, preds["prediction"]))
            if cut:
                notes.append(f"{cut} predictions differ in xlsx (cell limit)")
        if n_failed == len(items):
            notes.append("all answers failed; not evaluated")
            metrics = {}
        else:
            metrics = evaluate(bench, ds, eval_file, work_dir)
            row["primary_score"] = metrics[b["primary_metric"]]
            if n_failed:
                notes.append(f"{n_failed} failed answers scored by the evaluator as given")
        row["notes"] = "; ".join(notes) or "ok"
        (work_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    except Exception as err:
        row["notes"] = f"ERROR {type(err).__name__}: {str(err)[:300]}"
        row.setdefault("n_failed", "")
        traceback.print_exc()
        raise
    finally:
        row["duration_s"] = round(time.time() - t0, 1)
        row["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta = {**row, "system": system_info(), "usage": meter.as_dict(), "items_file": f"items/{bench}_{split}.csv",
                "model_config": cfg["models"].get(model_key), "prompt": cfg.get("prompt", "model default")}
        (work_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        if usage_dir is not None:
            Path(usage_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(usage_dir) / "usage.jsonl", "a") as f:
                f.write(json.dumps({k: meta[k] for k in ("timestamp_utc", "model", "benchmark", "split",
                                                          "n_items", "usage")}) + "\n")
        append_run_log(row, log_path)
    return row


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(cfg["models"]))
    ap.add_argument("--bench", required=True, nargs="+", choices=list(cfg["benchmarks"]))
    ap.add_argument("--split", required=True, choices=["pilot", "main"])
    ap.add_argument("--operator", required=True, choices=["Miras", "Adilzhan"])
    ap.add_argument("--api-nproc", type=int, default=4)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    for bench in a.bench:
        row = run(a.model, bench, a.split, a.operator, cfg=cfg, api_nproc=a.api_nproc, note=a.note)
        print(json.dumps(row, default=str))


if __name__ == "__main__":
    main()
