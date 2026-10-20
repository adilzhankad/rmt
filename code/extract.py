"""Build the long-format score dataset from the saved leaderboard sources.

Usage:  python -I extract.py <raw_dir> <out_csv>

Every row = one score of one model on one benchmark split (unit of analysis:
a model-benchmark score). Nothing is typed by hand except the CC-OCR README
table, which is parsed from markdown like the others.
"""
import csv, html, re, sys
from pathlib import Path

RAW = Path(sys.argv[1]); OUT = Path(sys.argv[2])
COLLECTED = "2026-10-13"

def cells(row):
    return [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
            for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)]

def html_table(text, start):
    s = text.find("<table", start); e = text.find("</table>", s)
    return [cells(r) for r in re.findall(r"<tr.*?</tr>", text[s:e], re.S)], e

# ---- model name normalisation: raw label -> (model_id, match_level) ----------
# match_level "exact": same model and size; "family": same model line, version
# not stated identically (used only in the sensitivity analysis).
CANON = {
    "gemini-1.5-pro": "gemini-1.5-pro", "gemini1.5-pro": "gemini-1.5-pro",
    "gpt-4o": "gpt-4o", "gpt4o": "gpt-4o",
    "claude3.5-sonnet": "claude-3.5-sonnet",
    "qwen-vl-72b": "qwen2-vl-72b", "qwen2-vl-72b": "qwen2-vl-72b",
    "internvl2-76b": "internvl2-76b",
    "got": "got-ocr2.0", "got-ocr": "got-ocr2.0",
    "textmonkey": "textmonkey",
    "gemini-2.5-pro": "gemini-2.5-pro", "gemini2.5-pro": "gemini-2.5-pro", "gemini-2.5 pro": "gemini-2.5-pro",
    "gemini2.0-flash": "gemini-2.0-flash",
    "qwen2.5-vl-7b": "qwen2.5-vl-7b", "qwen2.5-vl-72b": "qwen2.5-vl-72b",
    "qwen2-vl-7b": "qwen2-vl-7b",
    "internvl3-78b": "internvl3-78b",
    "mistral ocr": "mistral-ocr", "mistral ocr api": "mistral-ocr",
    "nanonets-ocr-s": "nanonets-ocr-s", "nanonets-ocr2-3b": "nanonets-ocr2-3b",
    "deepseek-ocr": "deepseek-ocr", "paddleocr-vl": "paddleocr-vl", "paddleocr-vl*": "paddleocr-vl",
    "mineru2.5": "mineru-2.5", "mineru 2.5.4*": "mineru-2.5",
    "olmocr": "olmocr", "olmocr v0.4.0": "olmocr", "olmocr-sglang": "olmocr",
    "gpt5-2025-08-07": "gpt-5-2025-08-07", "xcomposer2-4khd": "xcomposer2-4khd",
    "paddleocr pp-structurev3": "pp-structurev3", "dolphin-0.3b": "dolphin",
}
FAMILY_ONLY = {("olmOCR-Bench", "mineru-2.5"), ("olmOCR-Bench", "olmocr"), ("OmniDocBench", "olmocr")}  # version differs or is not stated

def norm(raw, size=""):
    k = re.sub(r"[🥇🥈🥉]", "", raw).strip().lower()
    if k in CANON:
        return CANON[k]
    if size and size not in ("-", "") and not re.search(r"\d+b", k):
        k = f"{k}-{size.lower()}"
    return CANON.get(k, k)

STATED = {  # versions stated by the source (CC-OCR README footnote)
    ("CC-OCR", "gpt-4o"): "2024-08-06", ("CC-OCR", "gemini-1.5-pro"): "002",
    ("CC-OCR", "claude-3.5-sonnet"): "20241022",
}

rows = []
def add(bench, ver, split, raw, mid, metric, direction, value, source, snapshot, note=""):
    try:
        v = float(str(value).split("±")[0]); status = "ok"
    except ValueError:
        v = ""; status = "missing"
    if v == 0.0 and bench == "CC-OCR":
        status = "zero_unsupported"   # README: 0.00 = model cannot do this track
    m = re.search(r"(\d{4}-?\d{2}-?\d{2}|v\d[\d.]*|\d+\.\d+\.\d+)", raw)
    stated = STATED.get((bench, mid), m.group(1) if m else "not stated")
    level = "family" if (bench, mid) in FAMILY_ONLY else "exact"
    rows.append(dict(benchmark=bench, benchmark_version=ver, split=split, model_raw=raw,
                     model_id=mid, model_version_stated=stated, match_level=level,
                     metric=metric, direction=direction, value=v, status=status,
                     source=source, snapshot=snapshot, collected_on=COLLECTED, note=note))

# ---- OCRBench v2: private-data leaderboard, 2025-09 release -------------------
h = (RAW / "v2_main.html").read_text(encoding="utf8")
pos = h.find("leaderboard-2025-09")
for split in ("English (private)", "Chinese (private)"):
    t, pos = html_table(h, pos)
    head = t[0]
    for r in t[1:]:
        add("OCRBench v2", "private set, 2025-09 release", split, r[1], norm(r[1], r[4]),
            "Average score", "higher", r[head.index("Average")],
            "github.com/99Franklin/ocrbench_v2 (official leaderboard)", "2025-09")

# ---- CC-OCR: README leaderboard (same numbers as Yang et al., 2025, Tables 2-5)
c = (RAW / "ccocr.md").read_text(encoding="utf8")
tracks = ["Multi-scene", "Multilingual", "Document parsing", "KIE", "Total"]
for line in c.splitlines():
    p = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(p) == 6 and re.match(r"^\d", p[1]):
        for name, val in zip(tracks, p[1:]):
            add("CC-OCR", "v1 (Nov 2024 runs)", name, p[0], norm(p[0]), "Track score", "higher", val,
                "github.com/AlibabaResearch/AdvancedLiterateMachinery, CC-OCR README", "2024-11")

# ---- OmniDocBench v1.0 (CVPR 2025 version): Overall edit distance, EN and ZH --
o = (RAW / "omni_v1_0.md").read_text(encoding="utf8")
t, _ = html_table(o, o.find("### End-to-End Evaluation"))
for r in t[2:]:
    r = r[1:] if r[0] in ("Pipeline Tools", "Expert VLMs", "General VLMs") else r
    for split, val in (("English pages", r[1]), ("Chinese pages", r[2])):
        add("OmniDocBench", "v1.0 (CVPR 2025)", split, r[0], norm(r[0]), "Overall edit distance",
            "lower", val, "github.com/opendatalab/OmniDocBench, branch v1_0 README", "2025-08")

# ---- OmniDocBench v1.5: Overall (all pages) -- used only for the version check -
o5 = (RAW / "omni_v1_5.md").read_text(encoding="utf8")
t, _ = html_table(o5, o5.find("### End-to-End Evaluation"))
for r in t[1:]:
    r = r[1:] if r[0] in ("SpecializedVLMs", "GeneralVLMs", "PipelineTools") else r
    add("OmniDocBench", "v1.5", "All pages", r[0], norm(r[0], r[1]), "Overall score", "higher", r[2],
        "github.com/opendatalab/OmniDocBench, branch v1_5 README", "2026-03")

# ---- olmOCR-Bench: README leaderboard (comparison systems + latest olmOCR) ----
b = (RAW / "olmbench.md").read_text(encoding="utf8")
t, _ = html_table(b, 0)
for r in t[1:]:
    if len(r) < 10 or (r[0].startswith(("v0.", "olmOCR (first")) and not r[0].startswith("v0.4.0")):
        continue
    if r[0].startswith("v0.4.0"):
        r = ["olmOCR v0.4.0"] + r[1:]
    add("olmOCR-Bench", "README leaderboard", "Overall", r[0], norm(r[0]), "Unit-test pass rate (%)",
        "higher", r[9], "github.com/allenai/olmocr, olmocr/bench README", "2025-10",
        note="* = score reported by the system's authors" if "*" in r[0] else "")

with OUT.open("w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows -> {OUT}")
