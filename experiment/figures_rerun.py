"""Figures of the re-run (fig3-fig8), same look as code/figures.py (fig1, fig2).

Called by analyze.py. Model identity is never colour-only: every mark is labelled.
Colour encodes model type (closed API vs open weights) or tau ranges.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from common import harness  # noqa: E402

plt.rcParams.update({"font.family": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 11})
INK, MUTED, GRID, SURFACE = "#222222", "#6b6b6b", "#e4e3df", "#ffffff"
CLOSED, OPEN = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated palette)
ZONE_LOW, ZONE_MID, ZONE_HIGH = "#f6dede", "#f3efe4", "#dcefe5"
DIV = ["#b2382f", "#e08a7e", "#f0efec", "#7fb2e8", "#256abf"]   # diverging, gray midpoint at tau = 0
SHORT = {"OCRBench v2": "OCRBench v2", "CC-OCR": "CC-OCR", "OmniDocBench": "OmniDocBench",
         "olmOCR-Bench": "olmOCR-Bench"}


def kind(cfg, m):
    return cfg["models"].get(m, {}).get("kind", "api")


def colour(cfg, m):
    return CLOSED if kind(cfg, m) == "api" else OPEN


def legend_kinds(ax, loc="lower right"):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=CLOSED, label="closed (API)"),
         Line2D([], [], marker="o", ls="", color=OPEN, label="open weights")]
    ax.legend(handles=h, loc=loc, frameon=False, fontsize=9)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, length=3)


def save(fig, charts, name):
    Path(charts).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(charts) / name, dpi=220, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- fig3: scores per benchmark
def fig_scores(split, cfg, raw, charts):
    benches = list(cfg["benchmarks"].items())
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharey=False)
    for ax, (key, b) in zip(axes, benches):
        vals = raw.get((key, b["primary_metric"]), {})
        if not vals:
            ax.set_axis_off()
            continue
        better_high = b["direction"] == "higher"
        order = sorted(vals, key=lambda m: vals[m][0], reverse=not better_high)  # best at the top
        y = list(range(len(order)))[::-1]
        ax.barh(y, [vals[m][0] for m in order], height=0.62, color=[colour(cfg, m) for m in order],
                edgecolor=SURFACE, linewidth=2)
        for yi, m in zip(y, order):
            ax.text(vals[m][0], yi, f" {vals[m][0]:.3g}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=8.5)
        unit = "edit distance, lower = better" if not better_high else "higher = better"
        ax.set_title(f"{b['label']}\n{b['primary_metric']} ({unit})", fontsize=10, loc="left")
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        clean(ax)
    legend_kinds(axes[-1])
    fig.suptitle(f"Primary score of each model on each benchmark ({split} split; best model at the top)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, charts, f"fig3_scores_{split}.png")


# ---------------------------------------------------------------- fig4: rank slope chart
def fig_ranks(split, cfg, table, charts):
    labels = [b["label"] for b in cfg["benchmarks"].values() if table.get(b["label"])]
    models = sorted({m for lab in labels for m in table[lab]})
    if len(labels) < 2:
        return
    rank = {lab: {m: 1 + sum(v > table[lab][m] for v in table[lab].values()) for m in table[lab]} for lab in labels}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for m in models:
        xs = [i for i, lab in enumerate(labels) if m in rank[lab]]
        ys = [rank[labels[i]][m] for i in xs]
        ax.plot(xs, ys, color=colour(cfg, m), lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
        ax.text(xs[0] - 0.08, ys[0], m, ha="right", va="center", fontsize=8.5, color=INK)
        ax.text(xs[-1] + 0.08, ys[-1], m, ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([SHORT[lab] for lab in labels])
    ax.set_ylim(len(models) + 0.5, 0.5)
    ax.set_yticks(range(1, len(models) + 1))
    ax.set_ylabel("rank (1 = best)")
    ax.set_xlim(-1.3, len(labels) - 1 + 1.3)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    clean(ax)
    legend_kinds(ax, loc="lower center")
    ax.set_title(f"Rank of each model on each benchmark ({split}); crossing lines = disagreement",
                 loc="left", fontsize=12, fontweight="bold")
    save(fig, charts, f"fig4_ranks_{split}.png")


# ---------------------------------------------------------------- fig5: tau matrix
def div_colour(t):
    if t is None or math.isnan(t):
        return "#f0efec"
    edges = [-1, -0.6, -0.2, 0.2, 0.6, 1.01]
    for i in range(5):
        if edges[i] <= t < edges[i + 1]:
            return DIV[i]
    return DIV[-1]


def fig_tau_matrix(split, cfg, out, charts):
    labels = [b["label"] for b in cfg["benchmarks"].values()]
    tau = {}
    for p in out["pairs"]:
        tau[(p["a"], p["b"])] = tau[(p["b"], p["a"])] = p
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j > i:
                continue
            if i == j:
                ax.add_patch(Rectangle((j, i), 1, 1, fc="#f7f7f5", ec=SURFACE, lw=2))
                continue
            p = tau[(a, b)]
            t = p["tau_b"]
            ax.add_patch(Rectangle((j, i), 1, 1, fc=div_colour(t), ec=SURFACE, lw=2))
            txt = "n/a" if t is None else f"{t:.2f}"
            ax.text(j + .5, i + .42, txt, ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
            ci = p.get("ci_stratified")
            sub = f"n = {p['n_shared']}" if not ci or any(map(math.isnan, ci)) else f"[{ci[0]:.2f}, {ci[1]:.2f}]"
            ax.text(j + .5, i + .76, sub, ha="center", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 4)
    ax.set_ylim(4, 0)
    ax.set_xticks([k + .5 for k in range(4)])
    ax.set_xticklabels(["OCRBench\nv2", "CC-OCR", "OmniDoc-\nBench", "olmOCR-\nBench"], fontsize=10)
    ax.set_yticks([k + .5 for k in range(4)])
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    mt = out.get("mean_tau")
    side = [f"Mean τ-b: {mt:.2f}" if mt is not None else "Mean τ-b: n/a",
            f"Min τ-b: {out['min_tau']:.2f}" if out.get("min_tau") is not None else "",
            f"Verdict: {out['verdict']}"]
    ax.text(4.15, 0.25, "\n".join(s for s in side if s), fontsize=11, fontweight="bold", va="top")
    ax.text(4.15, 1.45, f"Kendall's τ-b between the\nrankings of {out['n_models']} models.\n"
            "Brackets: 95% stratified\nbootstrap CI.\nSuccess: mean ≥ 0.70\nand min ≥ 0.40.",
            fontsize=9.5, color=MUTED, va="top")
    fig.suptitle(f"Agreement between benchmark rankings ({split})", x=0.03, ha="left", fontsize=12,
                 fontweight="bold")
    fig.subplots_adjust(left=0.2, right=0.7, top=0.88, bottom=0.13)
    save(fig, charts, f"fig5_tau_matrix_{split}.png")


# ---------------------------------------------------------------- fig6: tau with CIs
def fig_tau_forest(split, out, charts):
    rows = [(f"{SHORT[p['a']]} × {SHORT[p['b']]}", p["tau_b"], p.get("ci_stratified"), p.get("ci_harness_plain_mean"))
            for p in out["pairs"] if p["tau_b"] is not None]
    if not rows:
        return
    if out.get("mean_tau") is not None:
        rows.append(("Mean of the six pairs", out["mean_tau"], out.get("mean_tau_ci"), None))
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(rows) + 1.6))
    ax.axvspan(-1, harness.LOW_AGREEMENT, color=ZONE_LOW, lw=0)
    ax.axvspan(harness.LOW_AGREEMENT, harness.HIGH_AGREEMENT, color=ZONE_MID, lw=0)
    ax.axvspan(harness.HIGH_AGREEMENT, 1, color=ZONE_HIGH, lw=0)
    for k, (name, t, ci, ci2) in enumerate(rows):
        y = len(rows) - 1 - k
        if ci2 and not any(map(math.isnan, ci2)):
            ax.plot(ci2, [y - 0.18] * 2, color=MUTED, lw=2, solid_capstyle="round")
        if ci and not any(map(math.isnan, ci)):
            ax.plot(ci, [y] * 2, color=INK, lw=2, solid_capstyle="round")
        ax.plot([t], [y], "o", color=CLOSED, ms=9, mec=SURFACE, mew=2, zorder=3)
        ax.text(1.02, y, f"{t:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Kendall's τ-b")
    for x, lab in ((harness.LOW_AGREEMENT, "0.40"), (harness.HIGH_AGREEMENT, "0.70")):
        ax.axvline(x, color=MUTED, lw=1, ls="--")
    ax.text(-0.98, len(rows) - 0.35, "low", fontsize=8.5, color=MUTED)
    ax.text(0.42, len(rows) - 0.35, "partial", fontsize=8.5, color=MUTED)
    ax.text(0.72, len(rows) - 0.35, "high", fontsize=8.5, color=MUTED)
    clean(ax)
    ax.set_title(f"Pairwise τ-b with 95% bootstrap intervals ({split})\n"
                 "black = stratified, official aggregate; grey = harness plain-mean bootstrap",
                 loc="left", fontsize=11, fontweight="bold")
    save(fig, charts, f"fig6_tau_ci_{split}.png")


# ---------------------------------------------------------------- fig7: language
def fig_language(split, cfg, lang_tab, out, charts):
    panels = [("OCRBench v2", "OCRBench v2 EN", "OCRBench v2 ZH"), ("OmniDocBench", "OmniDocBench EN", "OmniDocBench ZH")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    lt = {(p["a"], p["b"]): p for p in out["language"]}
    for ax, (title, a, b) in zip(axes, panels):
        shared = sorted(set(lang_tab[a]) & set(lang_tab[b]))
        if not shared:
            ax.set_axis_off()
            continue
        ra = {m: 1 + sum(lang_tab[a][x] > lang_tab[a][m] for x in shared) for m in shared}
        rb = {m: 1 + sum(lang_tab[b][x] > lang_tab[b][m] for x in shared) for m in shared}
        n = len(shared)
        ax.plot([0.5, n + 0.5], [0.5, n + 0.5], color=GRID, lw=2, zorder=1)
        for m in shared:
            ax.plot(ra[m], rb[m], "o", color=colour(cfg, m), ms=9, mec=SURFACE, mew=2, zorder=3)
            ax.annotate(m, (ra[m], rb[m]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
        ax.set_xlim(n + 0.7, 0.3)
        ax.set_ylim(n + 0.7, 0.3)
        ax.set_xlabel("rank on English pages / items (1 = best)")
        ax.set_ylabel("rank on Chinese pages / items")
        p = lt[(a, b)]
        t = p["tau_b"]
        ttxt = "n/a" if t is None or math.isnan(t) else f"{t:.2f}"
        ax.set_title(f"{title}: EN vs ZH, τ-b = {ttxt}", loc="left", fontsize=11)
        clean(ax)
    legend_kinds(axes[-1], loc="lower left")
    fig.suptitle(f"Language: does a model's rank change between English and Chinese? ({split}; on the diagonal = same rank)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, charts, f"fig7_language_{split}.png")


# ---------------------------------------------------------------- fig8: bootstrap mean tau
def fig_bootstrap(split, out, boot, charts):
    if not boot or not boot["mean"]:
        return
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.axvspan(-1, harness.LOW_AGREEMENT, color=ZONE_LOW, lw=0)
    ax.axvspan(harness.LOW_AGREEMENT, harness.HIGH_AGREEMENT, color=ZONE_MID, lw=0)
    ax.axvspan(harness.HIGH_AGREEMENT, 1, color=ZONE_HIGH, lw=0)
    ax.hist(boot["mean"], bins=40, range=(-1, 1), color=CLOSED, edgecolor=SURFACE, linewidth=1)
    ax.axvline(out["mean_tau"], color=INK, lw=2)
    ax.text(out["mean_tau"], ax.get_ylim()[1] * 0.95, f"  observed {out['mean_tau']:.2f}", fontsize=9, va="top")
    share = out.get("verdict_share", {})
    txt = "\n".join(f"{k}: {100 * v:.0f}% of replicates" for k, v in share.items())
    ax.text(-0.97, ax.get_ylim()[1] * 0.95, txt, fontsize=9, va="top", color=INK)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("mean pairwise τ-b in a bootstrap replicate")
    ax.set_ylabel("replicates")
    clean(ax)
    ax.set_title(f"How stable is the verdict? {len(boot['mean'])} stratified bootstrap replicates ({split})",
                 loc="left", fontsize=11, fontweight="bold")
    save(fig, charts, f"fig8_bootstrap_mean_tau_{split}.png")


def draw_all(split, cfg, table, out, raw, lang_tab, boot, charts):
    fig_scores(split, cfg, raw, charts)
    fig_ranks(split, cfg, table, charts)
    fig_tau_matrix(split, cfg, out, charts)
    fig_tau_forest(split, out, charts)
    fig_language(split, cfg, lang_tab, out, charts)
    fig_bootstrap(split, out, boot, charts)
