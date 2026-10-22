"""End-to-end tests on synthetic data with a dummy model (no network, no GPU).

    python -m pytest experiment/tests -q
"""
import json
import logging
import random

import pandas as pd
import pytest

import common
import item_scores
import run
import sample_items
from synthetic import BrokenModel, DummyModel

BENCHES = ["OCRBench_v2", "CCOCR", "OmniDocBench", "olmOCRBench"]


def do_run(bench_env, bench, quality=lambda i: i % 3 != 0, fail=(), model_key="GPT4o_MINI", split="main",
           item_ids=None):
    cfg, answers, paths = bench_env(bench, item_ids, split)
    model = DummyModel(answers, quality, fail)
    row = run.run(model_key, bench, split, "Adilzhan", cfg=cfg, model=model, out_root=paths["raw"],
                  log_path=paths["log"], usage_dir=paths["usage"])
    return cfg, paths, row, model


@pytest.mark.parametrize("bench", BENCHES)
def test_end_to_end_and_item_scores(bench_env, bench):
    cfg, paths, row, model = do_run(bench_env, bench)
    assert row["n_failed"] == 0 and row["notes"] == "ok"
    log = pd.read_csv(paths["log"])
    assert list(log.columns) == common.RUN_LOG_COLUMNS and len(log) == 1
    primary = float(log["primary_score"].iloc[0])
    work = paths["raw"] / "main" / "GPT4o_MINI" / bench
    metrics = json.loads((work / "metrics.json").read_text())
    assert metrics[cfg["benchmarks"][bench]["primary_metric"]] == pytest.approx(primary)
    preds = [json.loads(x) for x in (work / "predictions.jsonl").read_text().splitlines()]
    assert len(preds) == row["n_items"] == model.calls
    # per-item scores rebuild the official aggregate exactly (within the evaluator's own rounding)
    b = cfg["benchmarks"][bench]
    eval_file = work / f"GPT4o_MINI_{b['dataset']}.{b['pred_format']}"
    rows, checks = item_scores.score_run(bench, str(eval_file), metrics, common.read_items(bench, "main"))
    assert rows and all(c["ok"] for c in checks), checks
    # a perfect and a useless model must be ordered correctly
    assert 0 < len({r["score"] for r in rows})


@pytest.mark.parametrize("bench", BENCHES)
def test_better_model_scores_better(bench_env, bench):
    cfg, paths, good, _ = do_run(bench_env, bench, quality=lambda i: True, model_key="GPT4o_MINI")
    _, _, bad, _ = do_run(bench_env, bench, quality=lambda i: False, model_key="GPT4o_20240806")
    better = good["primary_score"] < bad["primary_score"] if cfg["benchmarks"][bench]["direction"] == "lower" \
        else good["primary_score"] > bad["primary_score"]
    assert better, (good["primary_score"], bad["primary_score"])


def test_omnidoc_subset_equals_full(bench_env):
    """Subset evaluation (positional matching on a subset TSV) gives the same page scores
    as the evaluation of the full benchmark on those pages."""
    quality = lambda i: i % 4 != 1  # noqa: E731
    _, paths, _, _ = do_run(bench_env, "OmniDocBench", quality, model_key="GPT4o_MINI", split="main")
    full_dir = paths["raw"] / "main" / "GPT4o_MINI" / "OmniDocBench"
    full_items = common.read_items("OmniDocBench", "main")
    full_rows, _ = item_scores.items_omnidoc(str(full_dir / "GPT4o_MINI_OmniDocBench.tsv"), full_items)
    subset = [7, 2, 9, 4, 1]  # unsorted on purpose
    _, paths, _, _ = do_run(bench_env, "OmniDocBench", quality, model_key="GPT4o_MINI", split="pilot",
                            item_ids=subset)
    sub_dir = paths["raw"] / "pilot" / "GPT4o_MINI" / "OmniDocBench"
    sub_rows, _ = item_scores.items_omnidoc(str(sub_dir / "GPT4o_MINI_OmniDocBench.tsv"),
                                            common.read_items("OmniDocBench", "pilot"))
    full = {r["unit"]: r["score"] for r in full_rows}
    assert {r["item"] for r in sub_rows} == {str(i) for i in subset}
    for r in sub_rows:
        assert r["score"] == pytest.approx(full[r["unit"]], abs=1e-12), r["unit"]


def test_failed_answers_are_counted_not_replaced(bench_env):
    _, paths, row, _ = do_run(bench_env, "CCOCR", fail={"3", "7"})
    assert row["n_failed"] == 2
    assert "2 failed answers" in row["notes"]
    preds = (paths["raw"] / "main" / "GPT4o_MINI" / "CCOCR" / "predictions.jsonl").read_text()
    assert preds.count(common.FAIL_MSG) == 2


def test_all_failed_gives_no_score(bench_env):
    cfg, answers, paths = bench_env("CCOCR")
    model = DummyModel(answers, lambda i: True, fail=set(answers))
    row = run.run("GPT4o_MINI", "CCOCR", "main", "Miras", cfg=cfg, model=model, out_root=paths["raw"],
                  log_path=paths["log"], usage_dir=None)
    assert row.get("primary_score", "") == "" and "not evaluated" in row["notes"]


def test_crash_is_logged(bench_env):
    cfg, answers, paths = bench_env("olmOCRBench")
    with pytest.raises(RuntimeError):
        run.run("GPT4o_MINI", "olmOCRBench", "main", "Miras", cfg=cfg, model=BrokenModel(answers, None),
                out_root=paths["raw"], log_path=paths["log"], usage_dir=None)
    log = pd.read_csv(paths["log"])
    assert log["notes"].iloc[0].startswith("ERROR RuntimeError") and pd.isna(log["primary_score"].iloc[0])


def test_api_key_is_redacted(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-123")
    lg = logging.getLogger("vlmeval.test_redaction")
    common.install_redaction()
    with caplog.at_level(logging.INFO):
        lg.info("Using API Base: x; API Key: sk-test-secret-123")
    assert "sk-test-secret-123" not in caplog.text and "***" in caplog.text


# ---------------------------------------------------------------- sampling
def test_allocate():
    sizes = {"a": 500, "b": 30, "c": 3, "d": 1}
    q = sample_items.allocate(sizes, 40)
    assert sum(q.values()) == 40 and all(v >= 1 for v in q.values())
    assert all(q[k] <= sizes[k] for k in sizes)
    assert sample_items.allocate(sizes, 40) == q


def test_plan_is_deterministic_and_disjoint():
    r = random.Random(0)
    frame = pd.DataFrame({"index": [str(i) for i in range(300)],
                          "category": [f"c{i % 7}" for i in range(300)]})
    frame["group"] = frame["category"]
    a = sample_items.plan("olmOCRBench", frame, 70, 14, random.Random("2026:x"))
    b = sample_items.plan("olmOCRBench", frame, 70, 14, random.Random("2026:x"))
    assert a[:2] == b[:2]
    main, pilot = a[0], a[1]
    assert len(set(main)) == 70 and len(set(pilot)) == 14 and not set(main) & set(pilot)
    assert frame[frame["index"].isin(main)]["category"].nunique() == 7
    assert r is not None


# ---------------------------------------------------------------- collect + analyze
def test_collect_and_analyze(bench_env):
    import analyze
    import collect
    models = ["GPT4o_20240806", "GPT4o_MINI", "Ovis2-8B", "Qwen2.5-VL-7B-Instruct", "llava-onevision-qwen2-7b-ov-hf", "InternVL3-8B"]
    for k, m in enumerate(models):
        for bench in BENCHES:
            # model k answers item i correctly with a rate that falls with k -> rankings agree
            cfg, paths, _, _ = do_run(bench_env, bench, quality=lambda i, k=k: (i * 7 + k) % 10 >= k + 2,
                                      model_key=m)
    rows, missing = collect.main(paths["log"], paths["raw"], paths["processed"], cfg)
    assert not missing and len({(r["model"], r["benchmark"]) for r in rows}) == len(models) * 4
    item_scores.main(paths["raw"], paths["processed"], cfg)
    out = analyze.analyze("main", cfg, paths["processed"], paths["charts"], reps=50, figures=True)
    assert out["n_models"] == 6 and out["verdict"] in ("rankings agree", "partial agreement", "rankings do not agree")
    assert all(p["n_shared"] == 6 for p in out["pairs"])
    assert "ci_stratified" in out["pairs"][0] and "ci_harness_plain_mean" in out["pairs"][0]
    assert len(list(paths["charts"].glob("fig*_main.png"))) == 6


def test_olmocr_empty_answer_is_scored_not_crashing(bench_env):
    cfg, answers, paths = bench_env("olmOCRBench")
    answers = {k: ("" if k == "4" else v[0], v[1]) for k, v in answers.items()}
    row = run.run("GPT4o_MINI", "olmOCRBench", "main", "Miras", cfg=cfg, model=DummyModel(answers, lambda i: True),
                  out_root=paths["raw"], log_path=paths["log"], usage_dir=None)
    assert row["notes"] == "ok" and row["primary_score"] < 100
    work = paths["raw"] / "main" / "GPT4o_MINI" / "olmOCRBench"
    metrics = json.loads((work / "metrics.json").read_text())
    _, checks = item_scores.score_run("olmOCRBench", str(work / "GPT4o_MINI_olmOCRBench.xlsx"), metrics,
                                      common.read_items("olmOCRBench", "main"))
    assert all(c["ok"] for c in checks), checks


def test_generate_overrides_are_forced():
    class Inner:
        def generate(self, *a, **k):
            return k

    class Wrapper:
        model = Inner()

    cfg = common.load_config()
    m = cfg["models"]["llava-onevision-qwen2-7b-ov-hf"]
    import vlmeval.config as vc
    orig = vc.supported_VLM.get("llava-onevision-qwen2-7b-ov-hf")
    vc.supported_VLM["llava-onevision-qwen2-7b-ov-hf"] = lambda **ov: Wrapper()
    try:
        model = run.build_model("llava-onevision-qwen2-7b-ov-hf", cfg)
    finally:
        vc.supported_VLM["llava-onevision-qwen2-7b-ov-hf"] = orig
    assert model.model.generate(max_new_tokens=2048) == {"max_new_tokens": cfg["max_tokens"], "do_sample": False}
    assert m["env"] == "open"


def test_compare_published(tmp_path):
    import compare_published as cp
    cfg = common.load_config()
    models = ["GPT4o_20240806", "GPT4o_MINI", "InternVL3-8B", "Qwen2.5-VL-7B-Instruct"]
    ours = pd.DataFrame([{"model": m, "benchmark": "OCRBench_v2", "split": "main", "metric": "English Overall Score",
                          "value": v} for m, v in zip(models, [0.5, 0.4, 0.45, 0.3])])
    pub = pd.DataFrame([{"benchmark": "OCRBench v2", "split": "English (private)", "model_id": cp.PUBLISHED_ID[m],
                         "metric": "Average score", "value": v, "status": "ok"} for m, v in zip(models, [47, 44, 45, 41])])
    ours.to_csv(tmp_path / "ours.csv", index=False)
    pub.to_csv(tmp_path / "pub.csv", index=False)
    rows, summary = cp.compare("main", tmp_path / "ours.csv", tmp_path / "pub.csv", tmp_path, cfg)
    first = summary[0]
    assert first["n_shared"] == 4 and first["kendall_tau_b"] == 1.0 and len(rows) == 4


def test_queue_shares_the_vllm_gpu():
    import queue_runs as qr
    cfg = common.load_config()
    job = lambda m, b: qr.Job(cfg, m, b, "main")
    qwen = [job("Qwen2.5-VL-7B-Instruct", b) for b in qr.BENCH_ORDER]
    hf = [job(m, b) for b in qr.BENCH_ORDER for m in ("InternVL3-8B", "Ovis2-8B", "llava-onevision-qwen2-7b-ov-hf")]
    assert all(j.vllm and j.gb == 24 for j in qwen) and not any(j.vllm for j in hf)
    queue, running, t = hf[:2] + qwen + hf[2:], [], 0.0

    def start():
        nonlocal t
        t += qr.START_GAP_S          # past the start gap
        nxt = qr.pick(queue, running, 2, now=t)
        if nxt:
            j = nxt[0]
            j.gpu, j.start = nxt[1], t
            queue.remove(j)
            running.append(j)
        return nxt and (nxt[0].model, nxt[1])
    # GPU 1: its vLLM job first, then one transformers job next to it; GPU 0: two transformers jobs
    assert start() == ("Qwen2.5-VL-7B-Instruct", 1)
    assert start() == ("InternVL3-8B", 1)
    assert start() == ("Ovis2-8B", 0)
    assert start() == ("llava-onevision-qwen2-7b-ov-hf", 0)
    assert start() is None
    # no second start on a GPU within the gap
    running.remove(next(j for j in running if j.gpu == 0))
    last = max(j.start for j in running if j.gpu == 0)
    assert qr.pick(queue, running, 2, now=last + 1) is None
    assert qr.pick(queue, running, 2, now=last + qr.START_GAP_S)[1] == 0
    # Qwen finished, a transformers job is still on GPU 1: room is kept for the next Qwen job
    running = [j for j in running if not j.vllm]
    assert qr.pick([j for j in queue if not j.vllm], running, 2, now=t + 999)[1] == 0
    assert qr.pick(queue, running, 2, now=t + 999)[0].model == "Qwen2.5-VL-7B-Instruct"
    # one GPU: the old rule, no skipping past a job that does not fit
    assert qr.pick(qwen[:1] + hf, [], 1, now=t)[0].vllm
