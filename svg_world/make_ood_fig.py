"""
Figure 9: Out-of-distribution generalization via cross-domain transfer.

Pre vs random, ρ only. Two groups of bars per dim: within-domain on the
left, cross-domain on the right; pretrained dark, random light.

The story:
  - Within-domain pretrained ρ ≈ 0.91, random ρ ≈ 0.51. Big gap.
  - Cross-domain pretrained ρ ≈ 0.65, random ρ ≈ 0.25. The gap
    persists across visual-style transfer.
  - Pretraining's advantage is style-invariant: the encoder learns
    position-tracking directions that survive the AquaWorld ↔
    WestWorld stylistic gap.

Numbers averaged over both transfer directions (Aqua→West, West→Aqua)
to get a single per-dim cross-domain number.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PLOT_SIZE = 0.55
PRE_COLOR     = "#1f4e9b"
PRE_LIGHT     = "#88a0c4"
RAND_COLOR    = "#d97a1a"
RAND_LIGHT    = "#f0b988"

DIMS = [
    ("p0_x",      "person"),
    ("p0_z",      "person"),
    ("p1_x",      "person"),
    ("p1_z",      "person"),
    ("p2_x",      "person"),
    ("p2_z",      "person"),
    ("p3_x",      "person"),
    ("p3_z",      "person"),
    ("dolphin_x", "animal"),
    ("dolphin_z", "animal"),
    ("turtle_x",  "animal"),
    ("turtle_z",  "animal"),
]

# Mean-pooled patches numbers averaged over both worlds (within) and both
# transfer directions (cross). Source: --features patches_mean --cross.

# Within-domain ρ
PAT_WITHIN_PRE = {
    "p0_x": 0.9301, "p0_z": 0.9008, "p1_x": 0.8192, "p1_z": 0.9017,
    "p2_x": 0.8609, "p2_z": 0.9021, "p3_x": 0.9103, "p3_z": 0.8823,
    "dolphin_x": 0.9540, "dolphin_z": 0.9784,
    "turtle_x":  0.9396, "turtle_z":  0.9699,
}
PAT_WITHIN_RAND = {
    "p0_x": 0.6792, "p0_z": 0.7620, "p1_x": 0.4382, "p1_z": 0.7029,
    "p2_x": 0.4590, "p2_z": 0.5783, "p3_x": 0.6259, "p3_z": 0.7330,
    "dolphin_x": 0.2386, "dolphin_z": 0.4539,
    "turtle_x":  0.1292, "turtle_z":  0.2937,
}

# Cross-domain pretrained: average of (island→western) and (western→island)
CROSS_PRE_IW = {
    "p0_x": 0.4956, "p0_z": 0.5653, "p1_x": 0.4457, "p1_z": 0.6921,
    "p2_x": 0.3739, "p2_z": 0.5599, "p3_x": 0.5167, "p3_z": 0.5553,
    "dolphin_x": 0.7736, "dolphin_z": 0.8592,
    "turtle_x":  0.7331, "turtle_z":  0.8116,
}
CROSS_PRE_WI = {
    "p0_x": 0.8164, "p0_z": 0.6904, "p1_x": 0.5824, "p1_z": 0.6981,
    "p2_x": 0.5727, "p2_z": 0.5717, "p3_x": 0.6373, "p3_z": 0.6207,
    "dolphin_x": 0.6309, "dolphin_z": 0.8214,
    "turtle_x":  0.6380, "turtle_z":  0.8263,
}

# Cross-domain random: average of (island→western) and (western→island)
CROSS_RAND_IW = {
    "p0_x": 0.4153, "p0_z": 0.4750, "p1_x": 0.2226, "p1_z": 0.3559,
    "p2_x": 0.1557, "p2_z": 0.2737, "p3_x": 0.2515, "p3_z": 0.4146,
    "dolphin_x":  0.0239, "dolphin_z": -0.0781,
    "turtle_x":   0.0350, "turtle_z":  -0.0268,
}
CROSS_RAND_WI = {
    "p0_x": 0.4451, "p0_z": 0.5882, "p1_x": 0.2343, "p1_z": 0.4519,
    "p2_x": 0.1810, "p2_z": 0.3139, "p3_x": 0.3899, "p3_z": 0.4860,
    "dolphin_x": -0.0348, "dolphin_z": -0.0345,
    "turtle_x":  -0.0158, "turtle_z":  -0.0101,
}


def avg(d1, d2):
    return {k: 0.5 * (d1[k] + d2[k]) for k in d1}


def main():
    labels = [d[0] for d in DIMS]
    groups = [d[1] for d in DIMS]

    within_pre  = np.array([PAT_WITHIN_PRE[d]  for d, _ in DIMS])
    within_rand = np.array([PAT_WITHIN_RAND[d] for d, _ in DIMS])
    cross_pre   = avg(CROSS_PRE_IW, CROSS_PRE_WI)
    cross_pre   = np.array([cross_pre[d] for d, _ in DIMS])
    cross_rand  = avg(CROSS_RAND_IW, CROSS_RAND_WI)
    cross_rand  = np.array([cross_rand[d] for d, _ in DIMS])

    # 4 bars per dim: [within_pre, within_rand, cross_pre, cross_rand]
    fig, ax = plt.subplots(figsize=PLOT_SIZE * np.array((13, 4.5)))

    x = np.arange(len(labels))
    bw = 0.2
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bw

    ax.bar(x + offsets[0], within_pre,  width=bw, color=PRE_COLOR,
           edgecolor="white", linewidth=0.4)
    ax.bar(x + offsets[1], within_rand, width=bw, color=PRE_LIGHT,
           edgecolor="white", linewidth=0.4)
    ax.bar(x + offsets[2], cross_pre,   width=bw, color=RAND_COLOR,
           edgecolor="white", linewidth=0.4)
    ax.bar(x + offsets[3], cross_rand,  width=bw, color=RAND_LIGHT,
           edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"Pearson $\rho$")
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    legend_elems = [
        Patch(facecolor=PRE_COLOR,  label="within-domain, pretrained"),
        Patch(facecolor=PRE_LIGHT,  label="within-domain, random"),
        Patch(facecolor=RAND_COLOR, label="cross-domain, pretrained"),
        Patch(facecolor=RAND_LIGHT, label="cross-domain, random"),
    ]
    fig.legend(
        handles=legend_elems, loc="upper center",
        bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=False,
        fontsize=9,
    )

    fig.savefig("fig_ood_cross_domain.pdf",
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig("fig_ood_cross_domain.png",
                bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print("wrote fig_ood_cross_domain.{pdf,png}")

    # Print summary stats
    print("\nMean-pooled patches, mean over all 12 position dims:")
    print(f"  within pretrained ρ = {within_pre.mean():.3f}")
    print(f"  within random     ρ = {within_rand.mean():.3f}")
    print(f"  cross  pretrained ρ = {cross_pre.mean():.3f}")
    print(f"  cross  random     ρ = {cross_rand.mean():.3f}")
    print(f"  within Δ (pre - rand)  = {(within_pre - within_rand).mean():.3f}")
    print(f"  cross  Δ (pre - rand)  = {(cross_pre  - cross_rand ).mean():.3f}")


if __name__ == "__main__":
    main()
