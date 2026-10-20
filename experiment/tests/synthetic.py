"""Tiny synthetic TSVs shaped like each benchmark, plus a dummy model.

Every row gets a prediction chosen by the test (some right, some wrong), so the official
evaluators produce non-trivial scores. Nothing here is real data or a real model.
"""
import base64
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

FAIL_MSG = "Failed to obtain answer via API. "


def png_b64(seed):
    im = Image.new("RGB", (16, 16), (255 - seed % 200, 255, 255))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- OCRBench v2
def ocrbench_rows():
    spec = [  # category, question, answers, good prediction, bad prediction
        ("text recognition en", "What is written?", ["hello"], "hello", "goodbye"),
        ("math QA en", "What is 6 x 7?", ["42"], "42", "41"),
        ("document classification en", "Which type of document is this?", ["letter"], "letter", "memo"),
        ("cognition VQA cn", "这是哪个城市?", ["北京"], "北京", "上海"),
        ("full-page OCR cn", "请识别图中文字", ["今天天气很好我们去公园散步"], "今天天气很好我们去公园散步", "明天下雨"),
    ]
    rows = []
    for i in range(20):
        cat, q, ans, good, bad = spec[i % len(spec)]
        rows.append({"index": i, "image": png_b64(i), "question": q, "answer": str(ans), "category": cat,
                     "eval": "without eval", "bbox": "without bbox", "content": "without content",
                     "_good": good, "_bad": bad})
    return rows


# ---------------------------------------------------------------- CC-OCR
def ccocr_rows():
    spec = [  # category, l2-category, split, answer, good, bad
        ("multi_lan_ocr", "French", "French", "bonjour le monde entier", "bonjour le monde entier", "bonsoir"),
        ("multi_lan_ocr", "German", "German", "guten tag welt", "guten tag welt", "tag"),
        ("multi_scene_ocr", "scene_text", "IC15", "STOP 24 EXIT", "STOP 24 EXIT", "STOP"),
        ("kie", "constrained_category", "sroie2019_word",
         json.dumps({"company": "ABC SHOP", "total": "9.00"}),
         json.dumps({"company": "ABC SHOP", "total": "9.00"}), json.dumps({"company": "XYZ", "total": "1.00"})),
        ("doc_parsing", "doc", "doc_photo_eng", "Thesumofsquaresisalwayspositive",
         "The sum of squares is always positive", "Nothing here"),
    ]
    rows = []
    for i in range(20):
        cat, l2, split, ans, good, bad = spec[i % len(spec)]
        rows.append({"index": i, "image": png_b64(i), "image_name": f"{split}_{i}.jpg",
                     "question": "Read the image.", "answer": ans, "category": cat, "l2-category": l2,
                     "split": split, "_good": good, "_bad": bad})
    return rows


# ---------------------------------------------------------------- OmniDocBench
def omni_page(i, lang):
    zh = lang == "simplified_chinese"
    text = "这是一个关于文档解析的测试段落" if zh else "This paragraph is a test of document parsing quality"
    title = "第一章 引言" if zh else "Chapter one introduction"
    table_attr = {"table_layout": "horizontal", "with_span": False, "line": "full_line",
                  "language": "table_simplified_chinese" if zh else "table_en", "include_equation": False,
                  "include_photo": False, "include_background": False, "with_structured_text": False}
    dets = [
        {"category_type": "title", "ignore": False, "order": 1, "anno_id": 0, "text": title,
         "attribute": {"text_language": "text_simplified_chinese" if zh else "text_english"}},
        {"category_type": "text_block", "ignore": False, "order": 2, "anno_id": 1, "text": f"{text} {i}",
         "attribute": {"text_language": "text_simplified_chinese" if zh else "text_english"}},
        {"category_type": "equation_isolated", "ignore": False, "order": 3, "anno_id": 2,
         "latex": f"$$x^{{{i % 3 + 2}}}+y^2=z^2$$", "attribute": {"formula_type": "print"}},
        {"category_type": "table", "ignore": False, "order": 4, "anno_id": 3,
         "html": f"<table><tr><td>a</td><td>b</td></tr><tr><td>{i}</td><td>2</td></tr></table>",
         "latex": "", "attribute": table_attr},
    ]
    page = {"layout_dets": dets,
            "page_info": {"page_attribute": {"data_source": "book" if i % 2 else "academic_literature",
                                             "language": lang, "layout": "single_column", "special_issue": []},
                          "page_no": i, "height": 1000, "width": 800, "image_path": f"page_{i}.jpg"},
            "extra": {"relation": []}}
    good = (f"# {title}\n\n{text} {i}\n\n\\[x^{{{i % 3 + 2}}}+y^2=z^2\\]\n\n"
            f"<table><tr><td>a</td><td>b</td></tr><tr><td>{i}</td><td>2</td></tr></table>\n")
    bad = f"{text[:12]}\n\n<table><tr><td>z</td></tr></table>\n"
    return page, good, bad


def omnidoc_rows():
    rows = []
    for i in range(12):
        page, good, bad = omni_page(i, "english" if i % 2 == 0 else "simplified_chinese")
        rows.append({"index": i, "image": png_b64(i), "question": "", "answer": json.dumps(page, ensure_ascii=False),
                     "_good": good, "_bad": bad})
    return rows


# ---------------------------------------------------------------- olmOCR-Bench
def olmocr_rows():
    cats = ["tables", "multi_column", "old_scans"]
    rows = []
    for i in range(12):
        cat = cats[i % 3]
        pdf = f"{cat}/doc{i}_pg1.pdf"
        tests = [{"pdf": pdf, "page": 1, "id": f"doc{i}_present", "type": "present", "text": f"Invoice number {i}",
                  "max_diffs": 0},
                 {"pdf": pdf, "page": 1, "id": f"doc{i}_absent", "type": "absent", "text": "CONFIDENTIAL DRAFT",
                  "max_diffs": 0},
                 {"pdf": pdf, "page": 1, "id": f"doc{i}_order", "type": "order", "before": "First section",
                  "after": "Second section", "max_diffs": 0}]
        good = f"First section\n\nInvoice number {i}\n\nSecond section\n"
        bad = "Second section\n\nCONFIDENTIAL DRAFT\n\nFirst section\n"
        rows.append({"index": i, "image": png_b64(i), "pdf": "JVBERi0=", "question": "NaN",
                     "answer": json.dumps(tests), "_good": good, "_bad": bad})
    return rows


ROWS = {"OCRBench_v2": ocrbench_rows, "CCOCR": ccocr_rows, "OmniDocBench": omnidoc_rows,
        "olmOCRBench": olmocr_rows}


def write_tsv(bench, root):
    rows = ROWS[bench]()
    df = pd.DataFrame(rows)
    pred_cols = ["_good", "_bad"]
    path = Path(root) / f"{bench}.tsv"
    df.drop(columns=pred_cols).to_csv(path, sep="\t", index=False)
    return path, {str(r["index"]): (r["_good"], r["_bad"]) for r in rows}


class DummyModel:
    """Local-style model: answers from a table keyed by item index (image file stem)."""

    is_api = False

    def __init__(self, answers, quality, fail=()):
        self.answers, self.quality, self.fail = answers, quality, set(fail)
        self.calls = 0

    def set_dump_image(self, f):
        self.dump_image_func = f

    def generate(self, message, dataset=None):
        self.calls += 1
        img = next(x["value"] for x in message if x["type"] == "image")
        idx = Path(img).stem          # dump_image writes <index>.png
        if idx in self.fail:
            return FAIL_MSG
        good, bad = self.answers[idx]
        return good if self.quality(int(idx)) else bad


class BrokenModel(DummyModel):
    def generate(self, message, dataset=None):
        raise RuntimeError("model endpoint gone")
