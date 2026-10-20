"""Figures for Assignment 3. Usage: python -I figures.py <overlap.csv> <out_dir>"""
import csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 11})
INK, MUTED = "#222222", "#6b6b6b"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

ov = list(csv.DictReader(open(sys.argv[1])))
out = sys.argv[2]
B = ["OCRBench v2", "CC-OCR", "OmniDocBench", "olmOCR-Bench"]
N = {"OCRBench v2": 60, "CC-OCR": 9, "OmniDocBench": 27, "olmOCR-Bench": 9}
S = {(r["benchmark_a"], r["benchmark_b"]): int(r["shared_models"]) for r in ov}
S.update({(b, a): v for (a, b), v in list(S.items())})

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for i, a in enumerate(B):
    for j, b in enumerate(B):
        if j > i: continue
        if i == j:
            ax.add_patch(Rectangle((j, i), 1, 1, fc="#f0efec", ec="white", lw=2))
            ax.text(j + .5, i + .5, f"{N[a]}\nmodels", ha="center", va="center", fontsize=10, color=MUTED)
            continue
        v = S[(a, b)]
        ax.add_patch(Rectangle((j, i), 1, 1, fc=SEQ[min(v, 5)], ec="white", lw=2))
        ax.text(j + .5, i + .42, str(v), ha="center", va="center", fontsize=18, fontweight="bold", color=INK)
        ax.text(j + .5, i + .74, "shared" if v >= 5 else "below 5", ha="center", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, 4); ax.set_ylim(4, 0)
ax.set_xticks([k + .5 for k in range(4)]); ax.set_xticklabels(["OCRBench\nv2", "CC-OCR", "OmniDoc-\nBench", "olmOCR-\nBench"], fontsize=10)
ax.set_yticks([k + .5 for k in range(4)]); ax.set_yticklabels(B)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.text(4.15, 0.25, "Models in all four\npublished tables: 0", fontsize=11, fontweight="bold", va="top")
ax.text(4.15, 1.05, "A pair needs at least 5\nshared models before\nKendall's τ is computed.\nNo pair reaches 5.", fontsize=9.5, color=MUTED, va="top")
fig.suptitle("Models shared by the published leaderboards (same model after name normalisation)", x=0.03, ha="left", fontsize=12, fontweight="bold")
fig.subplots_adjust(left=0.2, right=0.72, top=0.88, bottom=0.13)
fig.savefig(f"{out}/fig1_overlap.png", dpi=220); plt.close(fig)

# Design flow
fig, ax = plt.subplots(figsize=(10, 1.9)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0.25, 1.95)
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#9a9a9a", lw=1))
    ax.text(x + .12, y + h - .18, title, fontsize=10.5, fontweight="bold", va="top", color=INK)
    ax.text(x + .12, y + h - .52, body, fontsize=9, va="top", color=INK, linespacing=1.35)
def arrow(x1, x2, y=1.12):
    ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4))
box(0.05, 0.35, 2.25, 1.55, "Stage 1 · done", "Published scores\n269 rows, 4 benchmarks\nAudit: 0 models on all\nfour; max 4 per pair", "#f0efec")
box(2.65, 0.35, 2.25, 1.55, "Stage 2 · inputs", "8 fixed MLLMs\n× 4 benchmarks\n400 stratified items each\n(seed 2026)", "#cde2fb")
box(5.25, 0.35, 2.25, 1.55, "Stage 2 · scoring", "Official scorer of each\nbenchmark, same prompt,\ntemperature 0, pinned\nmodel versions", "#cde2fb")
box(7.85, 0.35, 2.1, 1.55, "Analysis", "8 × 4 score matrix\n6 pairwise Kendall's τ-b\n95% bootstrap CI\nverdict vs τ ≥ 0.70", "#9ec5f4")
arrow(2.3, 2.65); arrow(4.9, 5.25); arrow(7.5, 7.85)
fig.savefig(f"{out}/fig2_design.png", dpi=220, bbox_inches="tight"); plt.close(fig)
print("ok")
