"""
SVG-World forward decoding figures.

Numbers are pulled from the cls --cross probe run (within-domain only for these
figures; cross-domain is for Fig 9).

Average over both worlds for the random-vs-pretrained comparison: each dim
shows the mean of (AquaWorld, WestWorld) since the within-domain story should
be world-agnostic.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects

# ----- Numbers from the cls within-domain probe runs -----
# Format: dim_label -> dict(island_pre_rho, island_pre_r2, island_rand_rho, ...,
#                           western_pre_rho, ...)
# Source: the output of 3_linear_probe.py with both --features cls and --cross.
# Within-domain only (the "AquaWorld | pre", "AquaWorld | rand", "WestWorld | pre",
# "WestWorld | rand" blocks).

DIMS = [
    # (label, group)  where group is 'person' or 'animal'
    ("p0_x",       "person"),
    ("p0_z",       "person"),
    ("p1_x",       "person"),
    ("p1_z",       "person"),
    ("p2_x",       "person"),
    ("p2_z",       "person"),
    ("p3_x",       "person"),
    ("p3_z",       "person"),
    ("dolphin_x",  "animal"),
    ("dolphin_z",  "animal"),
    ("turtle_x",   "animal"),
    ("turtle_z",   "animal"),
]

# Pearson rho per dim: (aqua_pre, aqua_rand, west_pre, west_rand)
RHO = {
    "p0_x":      (0.9746, 0.7273, 0.9603, 0.6912),
    "p0_z":      (0.9549, 0.8346, 0.9357, 0.7571),
    "p1_x":      (0.9123, 0.5256, 0.9020, 0.4568),
    "p1_z":      (0.9509, 0.7619, 0.9392, 0.6840),
    "p2_x":      (0.9369, 0.4996, 0.9163, 0.4918),
    "p2_z":      (0.9555, 0.6384, 0.9398, 0.5619),
    "p3_x":      (0.9642, 0.7305, 0.9455, 0.6204),
    "p3_z":      (0.9550, 0.7952, 0.9140, 0.6924),
    "dolphin_x": (0.9117, 0.0829, 0.9533, 0.7607),
    "dolphin_z": (0.9662, 0.2145, 0.9750, 0.7788),
    "turtle_x":  (0.8684, 0.1485, 0.8899, 0.2115),
    "turtle_z":  (0.9477, 0.3194, 0.9522, 0.3194),
}

# R^2 per dim, same structure:
R2 = {
    "p0_x":      (0.9494, 0.5009, 0.9218, 0.4448),
    "p0_z":      (0.9111, 0.6788, 0.8752, 0.5478),
    "p1_x":      (0.8304, 0.2441, 0.8127, 0.1855),
    "p1_z":      (0.9033, 0.5564, 0.8818, 0.4378),
    "p2_x":      (0.8758, 0.2065, 0.8383, 0.2032),
    "p2_z":      (0.9122, 0.3551, 0.8826, 0.2752),
    "p3_x":      (0.9282, 0.4924, 0.8929, 0.3460),
    "p3_z":      (0.9108, 0.5913, 0.8333, 0.4520),
    "dolphin_x": (0.8210, 0.0025, 0.9080, 0.4937),
    "dolphin_z": (0.9317, 0.0456, 0.9504, 0.5675),
    "turtle_x":  (0.7400, 0.0216, 0.7866, 0.0446),
    "turtle_z":  (0.8963, 0.0897, 0.9063, 0.0964),
}


# Style constants. The figure size scales with PLOT_SIZE.
PLOT_SIZE = 0.55
PERSON_COLOR     = "#1f4e9b"
PERSON_RAND_COL  = "#a8b8d4"   # muted blue for random
ANIMAL_COLOR     = "#d97a1a"
ANIMAL_RAND_COL  = "#f0c89e"   # muted orange for random


def aggregate(d):
    """Mean across both worlds for each dim. Returns (pre_mean, rand_mean) lists
    in the order of DIMS."""
    pre, rand = [], []
    for label, _ in DIMS:
        aqua_pre, aqua_rand, west_pre, west_rand = d[label]
        pre.append(0.5 * (aqua_pre + west_pre))
        rand.append(0.5 * (aqua_rand + west_rand))
    return np.array(pre), np.array(rand)


# ============================================================================
# Figure 6: forward decoding within-domain, pretrained only.
# Single-panel ρ figure. R² goes to an appendix table.
# ============================================================================

def fig_forward_decoding():
    rho_pre, _ = aggregate(RHO)
    labels = [d[0] for d in DIMS]
    groups = [d[1] for d in DIMS]
    colors = [PERSON_COLOR if g == "person" else ANIMAL_COLOR for g in groups]

    fig, ax = plt.subplots(figsize=PLOT_SIZE * np.array((11, 4.5)))

    x = np.arange(len(labels))
    ax.bar(x, rho_pre, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"Pearson $\rho$")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.95, color="gray", lw=0.7, ls="--", alpha=0.6, zorder=0)
    ax.text(len(labels) - 0.4, 0.96, r"$\rho{=}0.95$",
            color="gray", ha="right", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=PERSON_COLOR, label="person position"),
        Patch(facecolor=ANIMAL_COLOR, label="animal position"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=False,
              fontsize=9)

    fig.savefig("fig_forward_decoding.pdf",
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig("fig_forward_decoding.png",
                bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print("wrote fig_forward_decoding.{pdf,png}")


# ============================================================================
# Figure 7: pretrained vs random, paired bars, ρ only.
# ============================================================================

def fig_random_comparison():
    rho_pre, rho_rand = aggregate(RHO)
    labels = [d[0] for d in DIMS]
    groups = [d[1] for d in DIMS]
    pre_color  = [PERSON_COLOR    if g == "person" else ANIMAL_COLOR    for g in groups]
    rand_color = [PERSON_RAND_COL if g == "person" else ANIMAL_RAND_COL for g in groups]

    fig, ax = plt.subplots(figsize=PLOT_SIZE * np.array((11, 4.5)))

    x = np.arange(len(labels))
    bar_w = 0.4

    ax.bar(x - bar_w/2, rho_pre,  width=bar_w, color=pre_color,
           edgecolor="white", linewidth=0.6)
    ax.bar(x + bar_w/2, rho_rand, width=bar_w, color=rand_color,
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"Pearson $\rho$")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=PERSON_COLOR,    label="person, pretrained"),
        Patch(facecolor=PERSON_RAND_COL, label="person, random"),
        Patch(facecolor=ANIMAL_COLOR,    label="animal, pretrained"),
        Patch(facecolor=ANIMAL_RAND_COL, label="animal, random"),
    ]
    ax.legend(handles=legend_elems, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), frameon=False,
              fontsize=8, ncol=4)

    fig.savefig("fig_random_comparison.pdf",
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig("fig_random_comparison.png",
                bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print("wrote fig_random_comparison.{pdf,png}")


# Print summary stats for the caption / paper text
def print_summary():
    rho_pre, rho_rand = aggregate(RHO)
    r2_pre,  r2_rand  = aggregate(R2)
    person_idx = [i for i, (_, g) in enumerate(DIMS) if g == "person"]
    animal_idx = [i for i, (_, g) in enumerate(DIMS) if g == "animal"]
    all_idx    = list(range(len(DIMS)))

    def show(name, idx):
        print(f"  {name:<20} ρ pre {rho_pre[idx].mean():.3f}  "
              f"ρ rand {rho_rand[idx].mean():.3f}  "
              f"Δρ {(rho_pre[idx]-rho_rand[idx]).mean():.3f}   "
              f"R² pre {r2_pre[idx].mean():.3f}  "
              f"R² rand {r2_rand[idx].mean():.3f}")
    print("\nWithin-domain summary (mean over AquaWorld and WestWorld):")
    show("all 12 dims",   np.array(all_idx))
    show("person dims",   np.array(person_idx))
    show("animal dims",   np.array(animal_idx))


if __name__ == "__main__":
    fig_forward_decoding()
    fig_random_comparison()
    print_summary()
