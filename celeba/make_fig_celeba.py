"""CelebA results figure: ceiling-normalized reverse predictivity and spectral
concentration, pretrained vs. matched random initializations.

Reads results_celeba.csv (output of celeba_diagnostics.py), so rerun it after
the CLIP QuickGELU fix and the figure updates automatically.

Usage:  python make_fig_celeba.py results_celeba.csv fig_celeba_results.pdf [--horizontal] [--h=3.6]
        (default: two stacked panels sized for a wrapfigure of ~0.38 of the text width)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
CSV = ARGS[0] if len(ARGS) > 0 else "results_celeba.csv"
OUT = ARGS[1] if len(ARGS) > 1 else "fig_celeba_results.pdf"

PRE, RAND = "#1f3b73", "#f28e2b"          # dark blue / orange, as in Fig. 3
CELLS = [("dinov3_b16", "cls", "DINOv3\nCLS"),
         ("dinov3_b16", "patches_mean", "DINOv3\npatches"),
         ("clip_b16", "cls", "CLIP\nimage")]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

df = pd.read_csv(CSV)


def stats(enc, feat, norm, col):
    g = df[(df.encoder == enc) & (df.features == feat) & (df.normalization == norm)]
    pre = g[g.weights == "pre"].iloc[0]
    rnd = g[g.weights == "rand"]
    return pre, rnd


# panel (a): R2_rev / C_r ; pretrained error = bootstrap CI / ceiling
ratio_pre, ratio_pre_err, ratio_rnd, ratio_rnd_err = [], [], [], []
# panel (b): SC = reverse R2 on top-r whitened PCs ; pretrained is a single fit
sc_pre, sc_rnd, sc_rnd_err = [], [], []
for enc, feat, _ in CELLS:
    pre, rnd = stats(enc, feat, "raw", "rev_over_ceiling")
    v = pre.rev_over_ceiling
    lo, hi = pre.rev_lo / pre.ceiling, pre.rev_hi / pre.ceiling
    ratio_pre.append(v); ratio_pre_err.append([v - lo, hi - v])
    ratio_rnd.append(rnd.rev_over_ceiling.mean())
    ratio_rnd_err.append(rnd.rev_over_ceiling.std(ddof=1))
    pre_w, rnd_w = stats(enc, feat, "pca_whitened", "R2_rev")
    sc_pre.append(pre_w.R2_rev)
    sc_rnd.append(rnd_w.R2_rev.mean()); sc_rnd_err.append(rnd_w.R2_rev.std(ddof=1))

VERTICAL = "--horizontal" not in sys.argv   # default: stacked panels for a side column
# --h=3.6 sets the vertical figure height in inches (width stays 2.05); LaTeX then
# scales it by height to match the text beside it
FIG_H = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--h=")), 2.75))
if VERTICAL:
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 6.5})
x = np.arange(len(CELLS))
w = 0.36
if VERTICAL:
    fig, axes = plt.subplots(2, 1, figsize=(2.05, FIG_H), sharex=True,
                             constrained_layout=True)
else:
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.85), constrained_layout=True)
panels = [
    (axes[0], ratio_pre, np.array(ratio_pre_err).T, ratio_rnd, ratio_rnd_err,
     r"$R^2_{\mathrm{rev}} / C_r$", r"(a) Share of the ceiling" if VERTICAL else r"(a) Share of the ceiling reached"),
    (axes[1], sc_pre, None, sc_rnd, sc_rnd_err,
     r"SC", r"(b) Spectral concentration"),
]
for ax, vp, ep, vr, er, ylab, title in panels:
    b1 = ax.bar(x - w / 2, vp, w, color=PRE, yerr=ep, capsize=1.5,
                error_kw=dict(lw=0.6, capthick=0.6), label="pretrained")
    b2 = ax.bar(x + w / 2, vr, w, color=RAND, yerr=er, capsize=1.5,
                error_kw=dict(lw=0.6, capthick=0.6), label="random init")
    top = max(vp) * 1.28
    for rect, val in list(zip(b1, vp)) + list(zip(b2, vr)):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + top * 0.025,
                f"{val:.2f}", ha="center", va="bottom", fontsize=5.8 if VERTICAL else 6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in CELLS])
    ax.set_ylim(0, top)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="x", length=0)

if VERTICAL:   # shared x axis: labels only under the bottom panel
    axes[0].tick_params(labelbottom=False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
           ncol=2, frameon=False, handlelength=1.0, columnspacing=1.5)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print("saved", OUT)
