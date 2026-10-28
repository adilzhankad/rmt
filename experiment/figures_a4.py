"""Figures of the A4 report (Results and Discussion) -> charts/a4_fig*.png.

    python experiment/figures_a4.py

Everything is read from processed/ (written by collect.py, analyze.py, compare_published.py and
output_checks.py); nothing is typed in by hand. Same look as figures_rerun.py.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHARTS, PROCESSED, harness, load_config  # noqa: E402
from figures_rerun import CLOSED, GRID, INK, MUTED, OPEN, ZONE_HIGH, ZONE_LOW, ZONE_MID, clean  # noqa: E402

plt.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 11})
NAME = {"GPT4o_20240806": "GPT-4o", "GPT4o_MINI": "GPT-4o-mini", "InternVL3-8B": "InternVL3-8B",
        "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B", "Ovis2-8B": "Ovis2-8B", "Gemma3-4B": "Gemma3-4B",
        "llava-onevision-qwen2-7b-ov-hf": "LLaVA-OV-7B"}
BENCH = {"OCRBench_v2": "OCRBench v2", "CCOCR": "CC-OCR", "OmniDocBench": "OmniDocBench", "olmOCRBench": "olmOCR-Bench"}
LABELS = list(BENCH.values())
SAME_TYPE = {("OCRBench v2", "CC-OCR"), ("OmniDocBench", "olmOCR-Bench")}


def colour(cfg, m):
    return CLOSED if cfg["models"][m]["kind"] == "api" else OPEN


def zones(ax, horizontal=True):
    f = ax.axvspan if horizontal else ax.axhspan
    f(-1, 0.40, color=ZONE_LOW, zorder=0)
    f(0.40, 0.70, color=ZONE_MID, zorder=0)
    f(0.70, 1.0, color=ZONE_HIGH, zorder=0)


def kinds_legend(ax, loc):
    h = [Line2D([], [], marker="o", ls="", color=CLOSED, label="closed (API)"),
         Line2D([], [], marker="o", ls="", color=OPEN, label="open weights")]
    ax.legend(handles=h, loc=loc, frameon=False, fontsize=9)


def save(fig, name):
    CHARTS.mkdir(exist_ok=True)
    fig.savefig(CHARTS / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", CHARTS / name)


def primary(cfg):
    s = pd.read_csv(PROCESSED / "scores_rerun.csv")
    s = s[(s.split == "main") & s.is_primary]
    return {BENCH[b]: dict(zip(g.model, g.value)) for b, g in s.groupby("benchmark")}


def fig1_scores(cfg, scores):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    units = {"OCRBench v2": "English Overall (0-1, higher better)", "CC-OCR": "total (0-1, higher better)",
             "OmniDocBench": "edit distance EN (0-1, lower better)", "olmOCR-Bench": "tests passed, % (higher better)"}
    for ax, b in zip(axes, LABELS):
        lower = b == "OmniDocBench"
        items = sorted(scores[b].items(), key=lambda kv: kv[1], reverse=not lower)[::-1]
        for i, (m, v) in enumerate(items):
            only = m == "llava-onevision-qwen2-7b-ov-hf"
            ax.barh(i, v, color="white" if only else colour(cfg, m), edgecolor=colour(cfg, m), hatch="///" if only else None,
                    height=0.62)
            ax.text(v, i, f" {v:.3f}" if v < 1.5 else f" {v:.1f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(items)), [NAME[m] for m, _ in items], fontsize=9)
        ax.set_title(b, fontsize=11, loc="left", fontweight="bold")
        ax.set_xlabel(units[b], fontsize=9, color=MUTED)
        ax.set_xlim(0, max(scores[b].values()) * 1.25)
        clean(ax)
    h = [Line2D([], [], marker="s", ls="", color=CLOSED, label="closed (API)"),
         Line2D([], [], marker="s", ls="", color=OPEN, label="open weights")]
    fig.legend(handles=h, loc="lower right", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(1, -0.06))
    fig.suptitle("Primary score of every model on every benchmark (main run, 400 items each); best model at the top",
                 x=0.01, ha="left", fontsize=11)
    fig.text(0.01, -0.04, "Hatched bar: LLaVA-OV-7B, run on OCRBench v2 only (stopped). Source: processed/scores_rerun.csv",
             fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, "a4_fig1_scores.png")


def fig2_ranks(cfg):
    r = pd.read_csv(PROCESSED / "rank_table_main.csv").dropna()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    # ranks among the six complete models (dropna drops LLaVA-OV), so OCRBench v2 is ranked again
    six = r.copy()
    six["OCRBench v2 rank"] = six["OCRBench v2 score"].rank(ascending=False).astype(int)
    for row in six.itertuples(index=False):
        m = row[0]
        ranks = [row[2], row[4], row[6], row[8]]
        ax.plot(range(4), ranks, color=colour(cfg, m), lw=2, marker="o", ms=7, mec="white", zorder=3)
        ax.text(-0.08, ranks[0], NAME[m], ha="right", va="center", fontsize=9.5)
        ax.text(3.08, ranks[3], NAME[m], ha="left", va="center", fontsize=9.5)
    ax.set_xticks(range(4), LABELS)
    ax.set_yticks(range(1, 7))
    ax.set_ylim(6.5, 0.5)
    ax.set_xlim(-0.9, 3.9)
    ax.set_ylabel("rank (1 = best)")
    ax.yaxis.grid(True, color=GRID)
    clean(ax)
    h = [Line2D([], [], marker="o", ls="", color=CLOSED, label="closed (API)"),
         Line2D([], [], marker="o", ls="", color=OPEN, label="open weights")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=9)
    ax.set_title("Rank of the six complete models on each benchmark; crossing lines = disagreement",
                 loc="left", fontsize=11)
    save(fig, "a4_fig2_ranks.png")


def fig3_matrix():
    p = pd.read_csv(PROCESSED / "tau_pairs_main.csv")
    m = pd.DataFrame(1.0, index=LABELS, columns=LABELS)
    for r in p.itertuples():
        m.loc[r.a, r.b] = m.loc[r.b, r.a] = r.tau_b
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#f0efec")
    shown = m.values.copy()
    for i in range(4):
        shown[i, i] = float("nan")
    im = ax.imshow(shown, cmap=cmap, vmin=-1, vmax=1)
    for i in range(4):
        for j in range(4):
            v = m.values[i, j]
            ax.text(j, i, "—" if i == j else f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 and i != j else INK, fontsize=12)
    ax.set_xticks(range(4), LABELS, rotation=20, ha="right")
    ax.set_yticks(range(4), LABELS)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Kendall's τ-b (6 shared models)")
    ax.set_title("Kendall's τ-b between the rankings of every benchmark pair", loc="left", fontsize=11)
    save(fig, "a4_fig3_tau_matrix.png")


def fig4_pairs():
    p = pd.read_csv(PROCESSED / "tau_pairs_main.csv")
    j = pd.read_json(PROCESSED / "agreement_main.json", typ="series")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    zones(ax)
    rows = [(f"{r.a} × {r.b}", r.tau_b, eval(r.ci_stratified), (r.a, r.b) in SAME_TYPE) for r in p.itertuples()]
    rows.append(("Mean of the six pairs", j["mean_tau"], tuple(j["mean_tau_ci"]), None))
    for i, (lab, t, (lo, hi), same) in enumerate(rows[::-1]):
        ax.plot([lo, hi], [i, i], color=INK, lw=2, zorder=2)
        face = INK if same is None else (CLOSED if same else "white")
        ax.plot(t, i, "o", ms=9, mfc=face, mec=CLOSED if same is not None else INK, mew=2, zorder=3)
        ax.text(1.02, i, f"{t:.2f}  [{lo:.2f}, {hi:.2f}]", va="center", fontsize=9.5, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Kendall's τ-b; bar = 95% stratified bootstrap interval (1,000 replicates)")
    for x, t in ((0.20, "below 0.40"), (0.55, "0.40-0.70"), (0.85, "≥ 0.70")):
        ax.text(x, len(rows) - 0.4, t, ha="center", fontsize=9, color=MUTED)
    h = [Line2D([], [], marker="o", ls="", mfc=CLOSED, mec=CLOSED, label="same task type"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=CLOSED, mew=2, label="different task type")]
    ax.legend(handles=h, loc="lower left", frameon=False, fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    clean(ax)
    ax.set_title("Agreement of every benchmark pair, with its sampling uncertainty", loc="left", fontsize=11)
    save(fig, "a4_fig4_tau_pairs_ci.png")


def fig5_bootstrap():
    b = pd.read_csv(PROCESSED / "bootstrap_main.csv")
    j = pd.read_json(PROCESSED / "agreement_main.json", typ="series")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    zones(ax)
    vals = b["mean_tau"].round(4)
    counts = vals.value_counts().sort_index()
    ax.bar(counts.index, counts.values, width=0.018, color=CLOSED, zorder=2)
    ax.axvline(j["mean_tau"], color=INK, lw=1.5, zorder=3)
    ax.text(j["mean_tau"], counts.max() * 1.02, f" observed {j['mean_tau']:.2f}", fontsize=9.5)
    share = j["verdict_share"]
    txt = "\n".join(f"{k}: {v * 1000:.0f} of 1,000 replicates" for k, v in sorted(share.items(), key=lambda kv: -kv[1]))
    ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top", fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean pairwise τ-b in one bootstrap replicate (verdict zones: < 0.40, 0.40-0.70, ≥ 0.70)")
    ax.set_ylabel("replicates")
    clean(ax)
    ax.set_title("How stable is the verdict? Mean τ-b in 1,000 stratified bootstrap replicates", loc="left", fontsize=11)
    save(fig, "a4_fig5_bootstrap.png")


def slope(ax, cfg, left, right, title, lnames):
    shared = sorted(set(left) & set(right))
    rl = pd.Series({m: left[m] for m in shared}).rank(ascending=False)
    rr = pd.Series({m: right[m] for m in shared}).rank(ascending=False)
    for m in shared:
        ax.plot([0, 1], [rl[m], rr[m]], color=colour(cfg, m), lw=2, marker="o", ms=7, mec="white")
        ax.text(-0.05, rl[m], NAME[m], ha="right", va="center", fontsize=9)
        ax.text(1.05, rr[m], NAME[m], ha="left", va="center", fontsize=9)
    ax.set_xticks([0, 1], lnames)
    ax.set_yticks(range(1, len(shared) + 1))
    ax.set_ylim(len(shared) + 0.5, 0.5)
    ax.set_xlim(-0.8, 1.8)
    ax.yaxis.grid(True, color=GRID)
    ax.set_title(title, fontsize=10.5, loc="left")
    clean(ax)


def fig6_language(cfg):
    s = pd.read_csv(PROCESSED / "scores_rerun.csv")
    s = s[s.split == "main"]
    get = lambda b, met: dict(zip(s[(s.benchmark == b) & (s.metric == met)].model, s[(s.benchmark == b) & (s.metric == met)].value))
    lt = pd.read_csv(PROCESSED / "tau_language_main.csv").set_index(["a", "b"])["tau_b"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ocr_en, ocr_zh = get("OCRBench_v2", "English Overall Score"), get("OCRBench_v2", "Chinese Overall Score")
    omni_en = {m: -v for m, v in get("OmniDocBench", "overall_EN").items()}
    omni_zh = {m: -v for m, v in get("OmniDocBench", "overall_CH").items()}
    slope(axes[0], cfg, ocr_en, ocr_zh, f"OCRBench v2 (7 models): τ-b = {lt[('OCRBench v2 EN', 'OCRBench v2 ZH')]:.2f}",
          ["English", "Chinese"])
    slope(axes[1], cfg, omni_en, omni_zh, f"OmniDocBench (6 models): τ-b = {lt[('OmniDocBench EN', 'OmniDocBench ZH')]:.2f}",
          ["English pages", "Chinese pages"])
    axes[0].set_ylabel("rank (1 = best)")
    fig.suptitle("Rank on English vs Chinese material inside one benchmark", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "a4_fig6_language.png")


def fig7_published(cfg):
    p = pd.read_csv(PROCESSED / "published_vs_ours.csv")
    p = p[p.benchmark == "OCRBench_v2"]
    j = pd.read_json(PROCESSED / "published_vs_ours.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, met, lab in zip(axes, ["English Overall Score", "Chinese Overall Score"], ["English", "Chinese"]):
        q = p[p.metric == met]
        tau = harness.kendall_tau_b(list(q.ours), list(q.published))
        slope(ax, cfg, dict(zip(q.model, q.published)), dict(zip(q.model, q.ours)),
              f"OCRBench v2 {lab} ({len(q)} models): τ-b = {tau:.2f}", ["published leaderboard\n(private set)", "our re-run\n(400 public items)"])
    axes[0].set_ylabel("rank (1 = best)")
    fig.suptitle("Our OCRBench v2 ranking against the published leaderboard", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "a4_fig7_published.png")


def fig8_failures():
    o = pd.read_csv(PROCESSED / "output_checks_main.csv")
    o = o[o.model != "llava-onevision-qwen2-7b-ov-hf"]
    piv = o.pivot(index="model", columns="benchmark", values="loop").reindex(columns=list(BENCH))
    piv = piv.loc[piv.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(piv.values, cmap="Oranges", vmin=0, vmax=max(80, piv.values.max()), aspect="auto")
    for i in range(piv.shape[0]):
        for k in range(piv.shape[1]):
            v = int(piv.values[i, k])
            ax.text(k, i, f"{v} of 400 ({v / 4:.1f}%)", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 45 else INK)
    ax.set_xticks(range(4), LABELS)
    ax.set_yticks(range(piv.shape[0]), [NAME[m] for m in piv.index])
    fig.colorbar(im, ax=ax, fraction=0.04, label="answers ending in a repetition loop (of 400)")
    ax.set_title("Answers that end in a repetition loop, per model and benchmark (main run)", loc="left", fontsize=11)
    save(fig, "a4_fig8_loops.png")


def fig9_leave_one_out(cfg):
    import json
    n = json.loads((PROCESSED / "a4_numbers.json").read_text())
    j = pd.read_json(PROCESSED / "agreement_main.json", typ="series")
    loo = sorted(n["leave_one_model_out"].items(), key=lambda kv: kv[1]["mean_tau"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    zones(ax)
    for i, (m, v) in enumerate(loo):
        ax.plot([v["min_tau"], v["mean_tau"]], [i, i], color=MUTED, lw=1.2, zorder=2)
        ax.plot(v["min_tau"], i, "|", ms=12, color=INK, zorder=3)
        ax.plot(v["mean_tau"], i, "o", ms=9, color=colour(cfg, m), mec="white", zorder=3)
        ax.text(1.02, i, f"mean {v['mean_tau']:.2f}, min {v['min_tau']:.2f}: {v['verdict']}", va="center",
                fontsize=9, transform=ax.get_yaxis_transform())
    ax.axvline(j["mean_tau"], color=INK, lw=1, ls="--", zorder=1)
    ax.text(j["mean_tau"], len(loo) - 0.45, f" all six: {j['mean_tau']:.2f}", fontsize=9)
    ax.set_yticks(range(len(loo)), [f"without {NAME[m]}" for m, _ in loo])
    ax.set_xlim(-0.05, 1)
    ax.set_ylim(-0.6, len(loo) - 0.2)
    ax.set_xlabel("Kendall's τ-b over the six benchmark pairs on five models (dot = mean, bar = minimum)")
    clean(ax)
    ax.set_title("Sensitivity to the model set: verdict when one of the six models is left out", loc="left", fontsize=11)
    save(fig, "a4_fig9_leave_one_out.png")


def main():
    cfg = load_config()
    scores = primary(cfg)
    fig1_scores(cfg, scores)
    fig2_ranks(cfg)
    fig3_matrix()
    fig4_pairs()
    fig5_bootstrap()
    fig6_language(cfg)
    fig7_published(cfg)
    fig8_failures()
    fig9_leave_one_out(cfg)


if __name__ == "__main__":
    main()
