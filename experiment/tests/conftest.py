import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import synthetic  # noqa: E402

DATASET_CLASS = {"OCRBench_v2": ("vlmeval.dataset", "OCRBench_v2"),
                 "CCOCR": ("vlmeval.dataset", "CCOCRDataset"),
                 "OmniDocBench": ("vlmeval.dataset", "OmniDocBench"),
                 "olmOCRBench": ("vlmeval.dataset", "olmOCRBench")}


@pytest.fixture
def bench_env(tmp_path, monkeypatch):
    """Returns setup(bench, item_ids=None) -> (cfg, answers, paths). Everything lives in tmp_path."""
    lmu = tmp_path / "LMUData"
    lmu.mkdir()
    monkeypatch.setenv("LMUData", str(lmu))
    monkeypatch.setattr(common, "ITEMS", tmp_path / "items")
    paths = {"raw": tmp_path / "raw", "log": tmp_path / "logs" / "run_log.csv",
             "usage": tmp_path / "logs" / "usage", "processed": tmp_path / "processed",
             "charts": tmp_path / "charts", "lmu": lmu}
    cfg = copy.deepcopy(common.load_config())

    def setup(bench, item_ids=None, split="main"):
        import importlib
        tsv, answers = synthetic.write_tsv(bench, lmu)
        digest = common.md5_file(tsv)
        mod, cls = DATASET_CLASS[bench]
        klass = getattr(importlib.import_module(mod), cls)
        monkeypatch.setitem(klass.DATASET_MD5, bench, digest)
        cfg["benchmarks"][bench]["md5"] = digest
        df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
        if item_ids is not None:
            df = df[df["index"].isin([str(i) for i in item_ids])]
        items = pd.DataFrame({"index": df["index"]})
        if bench == "OmniDocBench":
            pages = [json.loads(a) for a in df["answer"]]
            items["image_name"] = [p["page_info"]["image_path"] for p in pages]
            items["language"] = [p["page_info"]["page_attribute"]["language"] for p in pages]
            items["stratum"] = items["language"]
        elif bench == "CCOCR":
            items["stratum"] = (df["category"] + "_" + df["l2-category"] + "_" + df["split"]).values
        elif bench == "OCRBench_v2":
            items["stratum"] = df["category"].values
        else:
            items["stratum"] = [json.loads(a)[0]["pdf"].split("/")[0] for a in df["answer"]]
        p = common.items_path(bench, split)
        p.parent.mkdir(parents=True, exist_ok=True)
        items.to_csv(p, index=False)
        return cfg, answers, paths

    return setup
