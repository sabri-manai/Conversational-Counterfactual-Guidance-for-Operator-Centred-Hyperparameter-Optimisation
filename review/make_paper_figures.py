"""Regenerate the manuscript figures from the frozen review tables.

usage: python make_paper_figures.py <repo-root> <output-dir>
Figures: surrogate-vs-confirmed (both datasets), pooled-vs-per-category AUROC, Task-3 trade-off.
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO, OUT = sys.argv[1], sys.argv[2]
T = f"{REPO}/review/outputs/tables"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"ACFX": "#1f77b4", "CFNOW": "#ff7f0e", "DiCE": "#2ca02c"}
MK = {"ACFX": "o", "CFNOW": "s", "DiCE": "^"}
LBL = {"ACFX": "ACFX", "CFNOW": "LogTopK", "DiCE": "DiCE"}

# ---- Figure 2: surrogate vs confirmed, two panels, legend below the axes ----
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.55))
for ax, ds, title, tgt in [(axes[0], "mvtec", "MVTec, per-category protocol (target $\\geq$ 0.70)", (0.70, None)),
                           (axes[1], "proditec", "Proditec (target band 0.75–0.95)", (0.75, 0.95))]:
    a = pd.read_csv(f"{T}/R02_{ds}_candidates_aggregated.csv")
    lo = min(a.surrogate.min(), a.f1_test.min()) - 0.02
    hi = max(a.surrogate.max(), a.f1_test.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="grey", label="Ideal agreement")
    for t in tgt:
        if t is not None and lo < t < hi:
            ax.axhline(t, ls=":", lw=0.8, color="black")
    for m, g in a.groupby("method"):
        ax.errorbar(g.surrogate, g.f1_test, yerr=g.f1_test_sd, fmt=MK[m], ms=3.5, color=COL[m], ecolor=COL[m],
                    elinewidth=0.6, capsize=1.2, label=f"{LBL[m]} (n={len(g)})", alpha=0.9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Surrogate F1"); ax.set_title(title)
axes[0].set_ylabel("Confirmed PatchCore test F1")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(f"{OUT}/fig_surrogate_vs_confirmed_both.pdf", bbox_inches="tight")

# ---- Figure 3: pooled vs per-category AUROC, individual points, full axis ----
c = pd.read_csv(f"{T}/R01_candidates_per_category.csv")
fig, ax = plt.subplots(figsize=(3.35, 1.9))
rng = np.random.default_rng(0)
xs = {"ACFX": 0, "CFNOW": 1, "DiCE": 2}
for m, g in c.groupby("method"):
    j = rng.uniform(-0.12, 0.12, len(g)); x = xs[m]
    ax.scatter(x - 0.2 + j, g.pooled_auroc_test, s=12, marker=MK[m], facecolors="none", edgecolors="#d62728",
               label="Pooled AUROC" if m == "ACFX" else None)
    ax.scatter(x + 0.2 + j, g.auroc_test, s=12, marker=MK[m], color="#1f77b4",
               label="Mean per-category AUROC" if m == "ACFX" else None)
ax.axhline(0.5, ls=":", lw=0.8, color="black")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["ACFX", "LogTopK", "DiCE"])
ax.set_ylim(0, 1); ax.set_ylabel("Test AUROC")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_mvtec_pooled_vs_percategory_auroc.pdf", bbox_inches="tight")

# ---- Figure 4: Task-3 quality-effort trade-off ----
d = pd.read_csv(f"{REPO}/user_study/revision_outputs/task_level_answers_export.csv")
d = d[d.task_name == "task_3"].copy()
BASE_F1, BASE_B = 0.84984, 0.375
d["dq"] = d.selected_quality - BASE_F1
d["red"] = (BASE_B - d.selected_review_budget) / BASE_B * 100
for cnd, g in d.groupby("condition"):
    print(cnd, "n", len(g), "mean dF1", round(g.dq.mean(), 3), "mean reduction %", round(g.red.mean(), 2))
fig, ax = plt.subplots(figsize=(3.35, 2.1))
ax.axhline(0, ls="--", lw=0.7, color="grey"); ax.axvline(5, ls=":", lw=0.8, color="black")
ax.text(5.6, -0.030, "5% budget\nreduction", fontsize=6, va="top")
for cnd, lab, col, mk in [("condition_a", "API (A)", "#1f77b4", "o"), ("condition_llm_interface", "Conversational (B)", "#ff7f0e", "s")]:
    g = d[d.condition == cnd]
    ax.scatter(g.red, g.dq, s=18, marker=mk, color=col, edgecolors="black", linewidths=0.4, label=lab, alpha=0.9)
ax.set_xlabel("Review-budget reduction from base configuration (%)")
ax.set_ylabel("Predicted F1 change\nfrom base configuration")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_user_study_task3_quality_budget_tradeoff.pdf", bbox_inches="tight")
print("done")
