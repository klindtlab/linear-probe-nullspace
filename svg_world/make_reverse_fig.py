"""
Figure 8: reverse predictivity (D1) and spectral concentration (D2) on
SVG-World features.

Two panels, one per feature mode (CLS, mean-pooled patches). Each panel
shows R^2_rev on the y-axis as a function of PCA-whitening cutoff k on the
x-axis. Pretrained and random-init curves shown.

The story:
  - At k=32 (matching the latent dimension), pretrained R^2_rev is close
    to its raw value. Most of the latent-aligned variance is already in
    the top ~32 PCs.
  - As k grows, R^2_rev decays toward 0. Adding low-variance PCs to the
    whitening expands the denominator faster than the numerator, because
    those low-variance PCs are largely z-irrelevant.
  - The pretrained-vs-random gap is largest at small k. This is what (D2)
    spectral concentration tests: pretrained has its z-aligned variance
    concentrated in a small number of high-variance directions; random
    spreads it out diffusely.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects

PLOT_SIZE = 0.55
PRE_COLOR  = "#1f4e9b"
RAND_COLOR = "#d97a1a"

# Aggregated numbers (mean across AquaWorld and WestWorld).
# Source: 4_reverse_predictivity.py runs across whitening k.
#   raw == no whitening (i.e. plotted at the right edge / 768 for CLS).
# Format: cls_pre, cls_rand, pat_pre, pat_rand
DATA = {
    "raw": {"k": None,
            "cls_pre": 0.1812, "cls_rand": 0.0194,
            "pat_pre": 0.2759, "pat_rand": 0.0225},
    32:    {"cls_pre": 0.1744, "cls_rand": 0.0227,
            "pat_pre": 0.1930, "pat_rand": 0.0289},
    64:    {"cls_pre": 0.1141, "cls_rand": 0.0216,
            "pat_pre": 0.1273, "pat_rand": 0.0251},
    128:   {"cls_pre": 0.0667, "cls_rand": 0.0178,
            "pat_pre": 0.0713, "pat_rand": 0.0194},
    256:   {"cls_pre": 0.0353, "cls_rand": 0.0140,
            "pat_pre": 0.0362, "pat_rand": 0.0137},
    "full":{"k": 768,
            "cls_pre": 0.0091, "cls_rand": 0.0056,
            "pat_pre": 0.0093, "pat_rand": 0.0049},
}

# Build x and y arrays. We treat raw and full as endpoints in different ways:
#   - "raw" reports R^2_rev on raw features, no whitening; we plot it as a
#     horizontal reference line, since it doesn't have a meaningful k.
#   - The whitening sweep runs through k=32, 64, 128, 256.
#   - "full" whitening with all components is plotted at k=768.
SWEEP_KS = [32, 64, 128, 256, 768]


def get_curve(mode, variant):
    """Return (ks, vals) arrays for the whitening sweep including the
    full-whitening endpoint."""
    ks = []
    vals = []
    for k in [32, 64, 128, 256]:
        ks.append(k)
        vals.append(DATA[k][f"{mode}_{variant}"])
    ks.append(768)
    vals.append(DATA["full"][f"{mode}_{variant}"])
    return np.array(ks), np.array(vals)


def main():
    fig, axes = plt.subplots(
        1, 2,
        figsize=PLOT_SIZE * np.array((11, 4.2)),
        gridspec_kw=dict(wspace=0.28),
        sharey=True,
    )

    for ax, mode, title in [
        (axes[0], "cls",  "CLS token"),
        (axes[1], "pat",  "Mean-pooled patches"),
    ]:
        # Whitening curves
        ks, pre_vals  = get_curve(mode, "pre")
        _,  rand_vals = get_curve(mode, "rand")

        ax.plot(ks, pre_vals,  marker="o", lw=2.0, color=PRE_COLOR,
                label="pretrained")
        ax.plot(ks, rand_vals, marker="s", lw=2.0, color=RAND_COLOR,
                label="random init")

        # Raw values as horizontal reference lines (left side annotations)
        raw_pre  = DATA["raw"][f"{mode}_pre"]
        raw_rand = DATA["raw"][f"{mode}_rand"]
        ax.axhline(raw_pre,  color=PRE_COLOR,  ls="--", lw=1.0, alpha=0.6)
        ax.axhline(raw_rand, color=RAND_COLOR, ls="--", lw=1.0, alpha=0.6)
        ax.text(750, raw_pre + 0.005, f"raw: {raw_pre:.2f}",
                color=PRE_COLOR, fontsize=8, ha="right", va="bottom")
        ax.text(750, raw_rand + 0.005, f"raw: {raw_rand:.2f}",
                color=RAND_COLOR, fontsize=8, ha="right", va="bottom")

        ax.set_xscale("log")
        ax.set_xticks([32, 64, 128, 256, 768])
        ax.set_xticklabels(["32", "64", "128", "256", "768"])
        ax.set_xlabel(r"PCA-whitening cutoff $k$")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-0.01, 0.32)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel(r"$R^2_\mathrm{rev}$")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9,
                   bbox_to_anchor=(0.02, 0.62))

    fig.savefig("fig_reverse_whitening.pdf",
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig("fig_reverse_whitening.png",
                bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print("wrote fig_reverse_whitening.{pdf,png}")


if __name__ == "__main__":
    main()
