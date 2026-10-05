"""
Forward decoding table for SVG-World.

Aggregates per-dim metrics into a compact table showing pretrained vs random
init for both CLS and mean-pooled patch features. Outputs LaTeX (booktabs)
and a plain-text version for inspection.

Numbers from the within-domain blocks of:
    python 3_linear_probe.py --features cls --cross
    python 3_linear_probe.py --features patches_mean --cross
averaged over AquaWorld and WestWorld.
"""

import numpy as np

# ============================================================================
# Numbers from the probe runs, averaged over the two worlds.
# Format per dim: (rho_cls_pre, rho_cls_rand, r2_cls_pre, r2_cls_rand,
#                  rho_pat_pre, rho_pat_rand, r2_pat_pre, r2_pat_rand)
# ============================================================================

DIMS = [
    "p0_x", "p0_z",
    "p1_x", "p1_z",
    "p2_x", "p2_z",
    "p3_x", "p3_z",
    "dolphin_x", "dolphin_z",
    "turtle_x",  "turtle_z",
]

# CLS numbers: mean of (Aqua, West) from the cls --cross within-domain blocks.
CLS_RHO_PRE = {
    "p0_x": 0.9675, "p0_z": 0.9453, "p1_x": 0.9072, "p1_z": 0.9451,
    "p2_x": 0.9266, "p2_z": 0.9477, "p3_x": 0.9549, "p3_z": 0.9345,
    "dolphin_x": 0.9325, "dolphin_z": 0.9706,
    "turtle_x":  0.8792, "turtle_z":  0.9500,
}
CLS_RHO_RAND = {
    "p0_x": 0.7093, "p0_z": 0.7959, "p1_x": 0.4912, "p1_z": 0.7230,
    "p2_x": 0.4957, "p2_z": 0.6002, "p3_x": 0.6755, "p3_z": 0.7438,
    "dolphin_x": 0.4218, "dolphin_z": 0.4967,
    "turtle_x":  0.1800, "turtle_z":  0.3194,
}
CLS_R2_PRE = {
    "p0_x": 0.9356, "p0_z": 0.8932, "p1_x": 0.8216, "p1_z": 0.8926,
    "p2_x": 0.8571, "p2_z": 0.8974, "p3_x": 0.9106, "p3_z": 0.8721,
    "dolphin_x": 0.8645, "dolphin_z": 0.9411,
    "turtle_x":  0.7633, "turtle_z":  0.9013,
}
CLS_R2_RAND = {
    "p0_x": 0.4729, "p0_z": 0.6133, "p1_x": 0.2148, "p1_z": 0.4971,
    "p2_x": 0.2049, "p2_z": 0.3152, "p3_x": 0.4192, "p3_z": 0.5217,
    "dolphin_x": 0.2481, "dolphin_z": 0.3066,
    "turtle_x":  0.0331, "turtle_z":  0.0931,
}

# patches_mean numbers: mean of (Aqua, West) from the patches_mean --cross
# within-domain blocks.
PAT_RHO_PRE = {
    "p0_x": 0.9301, "p0_z": 0.9008, "p1_x": 0.8192, "p1_z": 0.9017,
    "p2_x": 0.8609, "p2_z": 0.9021, "p3_x": 0.9103, "p3_z": 0.8823,
    "dolphin_x": 0.9540, "dolphin_z": 0.9784,
    "turtle_x":  0.9396, "turtle_z":  0.9699,
}
PAT_RHO_RAND = {
    "p0_x": 0.6792, "p0_z": 0.7620, "p1_x": 0.4382, "p1_z": 0.7029,
    "p2_x": 0.4590, "p2_z": 0.5783, "p3_x": 0.6259, "p3_z": 0.7330,
    "dolphin_x": 0.2386, "dolphin_z": 0.4539,
    "turtle_x":  0.1292, "turtle_z":  0.2937,
}
PAT_R2_PRE = {
    "p0_x": 0.8534, "p0_z": 0.7974, "p1_x": 0.6495, "p1_z": 0.8022,
    "p2_x": 0.7131, "p2_z": 0.7941, "p3_x": 0.8050, "p3_z": 0.7585,
    "dolphin_x": 0.8944, "dolphin_z": 0.9554,
    "turtle_x":  0.8622, "turtle_z":  0.9379,
}
PAT_R2_RAND = {
    "p0_x": 0.4074, "p0_z": 0.5502, "p1_x": 0.1617, "p1_z": 0.4475,
    "p2_x": 0.1579, "p2_z": 0.2527, "p3_x": 0.2976, "p3_z": 0.4191,
    "dolphin_x": 0.0713, "dolphin_z": 0.2481,
    "turtle_x":  0.0175, "turtle_z":  0.0557,
}


def render_latex():
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{\textbf{DINOv3-B/16 forward decoding on SVG-World, per-dim within-domain.} "
                 r"Pearson $\rho$ and $R^2$ from a ridge probe trained on each world separately, "
                 r"averaged over AquaWorld and WestWorld. Two feature modes: CLS token and mean-pooled "
                 r"patch tokens. \emph{Pre}: pretrained DINOv3. \emph{Rand}: same architecture, "
                 r"freshly initialised weights. Position latents only; facing and skin-hue are reported in App.~\ref{app:experiments}.}")
    lines.append(r"\label{tab:dino-forward}")
    lines.append(r"\begin{tabular}{l rrrr rrrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{4}{c}{CLS token} & \multicolumn{4}{c}{Mean-pooled patches} \\")
    lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    lines.append(r" & \multicolumn{2}{c}{$\rho$} & \multicolumn{2}{c}{$R^2$} & \multicolumn{2}{c}{$\rho$} & \multicolumn{2}{c}{$R^2$} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(r"latent & pre & rand & pre & rand & pre & rand & pre & rand \\")
    lines.append(r"\midrule")

    person_dims = [d for d in DIMS if d.startswith("p")]
    animal_dims = [d for d in DIMS if not d.startswith("p")]

    for d in person_dims:
        row = (
            d.replace("_", r"\_"),
            CLS_RHO_PRE[d], CLS_RHO_RAND[d], CLS_R2_PRE[d], CLS_R2_RAND[d],
            PAT_RHO_PRE[d], PAT_RHO_RAND[d], PAT_R2_PRE[d], PAT_R2_RAND[d],
        )
        lines.append(r"\texttt{%s} & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\" % row)

    lines.append(r"\midrule")
    for d in animal_dims:
        row = (
            d.replace("_", r"\_"),
            CLS_RHO_PRE[d], CLS_RHO_RAND[d], CLS_R2_PRE[d], CLS_R2_RAND[d],
            PAT_RHO_PRE[d], PAT_RHO_RAND[d], PAT_R2_PRE[d], PAT_R2_RAND[d],
        )
        lines.append(r"\texttt{%s} & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\" % row)

    # Aggregate row
    def agg(d_list, table):
        return np.mean([table[d] for d in d_list])
    lines.append(r"\midrule")
    for label, d_list in [("person mean", person_dims),
                          ("animal mean", animal_dims),
                          ("all mean",    DIMS)]:
        lines.append(r"\textbf{%s} & \textbf{%.2f} & \textbf{%.2f} & \textbf{%.2f} & \textbf{%.2f} & "
                     r"\textbf{%.2f} & \textbf{%.2f} & \textbf{%.2f} & \textbf{%.2f} \\" % (
            label,
            agg(d_list, CLS_RHO_PRE), agg(d_list, CLS_RHO_RAND),
            agg(d_list, CLS_R2_PRE),  agg(d_list, CLS_R2_RAND),
            agg(d_list, PAT_RHO_PRE), agg(d_list, PAT_RHO_RAND),
            agg(d_list, PAT_R2_PRE),  agg(d_list, PAT_R2_RAND),
        ))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_text():
    lines = []
    lines.append("Per-dim forward decoding (mean over AquaWorld and WestWorld)")
    lines.append("=" * 72)
    h1 = f"{'':12} {'CLS token':>30} {'Mean-pooled patches':>30}"
    h2 = (f"{'latent':12} "
          f"{'rho-pre':>7} {'rho-rand':>8} {'R2-pre':>7} {'R2-rand':>7} "
          f"{'rho-pre':>7} {'rho-rand':>8} {'R2-pre':>7} {'R2-rand':>7}")
    lines.append(h1)
    lines.append(h2)
    lines.append("-" * len(h2))
    for d in DIMS:
        lines.append(f"{d:12} "
                     f"{CLS_RHO_PRE[d]:>7.3f} {CLS_RHO_RAND[d]:>8.3f} "
                     f"{CLS_R2_PRE[d]:>7.3f} {CLS_R2_RAND[d]:>7.3f} "
                     f"{PAT_RHO_PRE[d]:>7.3f} {PAT_RHO_RAND[d]:>8.3f} "
                     f"{PAT_R2_PRE[d]:>7.3f} {PAT_R2_RAND[d]:>7.3f}")
    lines.append("-" * len(h2))

    person_dims = [d for d in DIMS if d.startswith("p")]
    animal_dims = [d for d in DIMS if not d.startswith("p")]
    def agg(d_list, table):
        return np.mean([table[d] for d in d_list])
    for label, d_list in [("person mean", person_dims),
                          ("animal mean", animal_dims),
                          ("all mean",    DIMS)]:
        lines.append(f"{label:12} "
                     f"{agg(d_list, CLS_RHO_PRE):>7.3f} {agg(d_list, CLS_RHO_RAND):>8.3f} "
                     f"{agg(d_list, CLS_R2_PRE):>7.3f}  {agg(d_list, CLS_R2_RAND):>7.3f} "
                     f"{agg(d_list, PAT_RHO_PRE):>7.3f} {agg(d_list, PAT_RHO_RAND):>8.3f} "
                     f"{agg(d_list, PAT_R2_PRE):>7.3f}  {agg(d_list, PAT_R2_RAND):>7.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    text = render_text()
    print(text)
    print()
    latex = render_latex()
    with open("table_forward_decoding.tex", "w") as f:
        f.write(latex + "\n")
    with open("table_forward_decoding.txt", "w") as f:
        f.write(text + "\n")
    print(f"\nwrote table_forward_decoding.{{tex,txt}}")
