"""
Render Scene-World datasets at 224x224.

Generates a single Z (n_total x 32) and renders BOTH themes from it. Same Z
across themes is what makes the cross-domain probe well-defined: row i in
island/ and row i in western/ share latents.

The island/western SVGs use a 600x420 viewBox. We post-process the SVG string
to extend the viewBox to 600x600 (90px water pad top and bottom) so the final
rasterization to 224x224 has no aspect distortion.

Inputs:  svg_island.py, svg_western.py in PYTHONPATH (or same dir)
Outputs: scene_world/Z.npy
         scene_world/latent_layout.json
         scene_world/island/{svgs,pngs}/island_NNNNN.{svg,png}
         scene_world/western/{svgs,pngs}/western_NNNNN.{svg,png}

Usage:   python 1_render_dataset.py --n_total 10000

Single-process, ~10 min for 20k images on a modern CPU. cairosvg is the
bottleneck; SVG generation is fast.
"""
import os
import json
import argparse
import numpy as np
import cairosvg
from tqdm import tqdm

import svg_island
import svg_western


def square_pad_svg(svg_str, vb_w=600, vb_h=420):
    """Pad an SVG with a vb_w x vb_h viewBox to a square (vb_w x vb_w) viewBox
    by extending top and bottom equally. The existing 100% water rect fills
    the new area automatically.
    """
    pad = (vb_w - vb_h) // 2  # 90
    new_h = vb_w               # 600 (square)
    svg_str = svg_str.replace(
        f'viewBox="0 0 {vb_w} {vb_h}"',
        f'viewBox="0 {-pad} {vb_w} {new_h}"',
    )
    # Outer width/height attrs: replace both. cairosvg honors output_width
    # regardless, but a self-consistent SVG is nicer for any other consumer.
    svg_str = svg_str.replace(
        f'width="{vb_w}" height="{vb_h}"',
        f'width="{new_h}" height="{new_h}"',
    )
    return svg_str


def render_theme(theme_module, theme_name, Z, out_root, output_size):
    svg_dir = os.path.join(out_root, theme_name, "svgs")
    png_dir = os.path.join(out_root, theme_name, "pngs")
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    n = Z.shape[0]
    print(f"Rendering {n} '{theme_name}' scenes -> {out_root}/{theme_name}")

    for i in tqdm(range(n), desc=theme_name, ncols=80):
        sid = f"{theme_name}_{i:05d}"
        svg_str = theme_module.generate_scene_svg(Z[i])
        svg_str = square_pad_svg(svg_str)

        with open(os.path.join(svg_dir, f"{sid}.svg"), "w") as f:
            f.write(svg_str)

        cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            write_to=os.path.join(png_dir, f"{sid}.png"),
            output_width=output_size,
            output_height=output_size,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_total", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_size", type=int, default=224)
    parser.add_argument("--out_root", default="scene_world")
    parser.add_argument("--themes", nargs="+", default=["island", "western"],
                        choices=["island", "western"])
    args = parser.parse_args()

    assert svg_island.LATENT_DIM == svg_western.LATENT_DIM, \
        "themes must share latent dim"
    LATENT_DIM = svg_island.LATENT_DIM

    rng = np.random.default_rng(args.seed)
    Z = rng.uniform(0, 1, (args.n_total, LATENT_DIM)).astype(np.float32)

    os.makedirs(args.out_root, exist_ok=True)
    np.save(os.path.join(args.out_root, "Z.npy"), Z)

    layout = {
        "n_total": args.n_total,
        "seed": args.seed,
        "latent_dim": LATENT_DIM,
        "output_size": args.output_size,
        "groups": {
            "person_0": list(range(0, 7)),
            "person_1": list(range(7, 14)),
            "person_2": list(range(14, 21)),
            "person_3": list(range(21, 28)),
            "dolphin_or_horse": [28, 29],
            "turtle_or_cow":    [30, 31],
        },
        "per_person_dim_meaning": {
            "0": "lane_x", "1": "lane_z", "2": "facing",
            "3": "skin_h",
            "4": "shirt_h_unused",
            "5": "shirt_s_unused",
            "6": "shirt_v_unused",
        },
        "used_dims_for_probing": (
            [p * 7 + i for p in range(4) for i in (0, 1, 2, 3)]
            + [28, 29, 30, 31]
        ),
        "notes": (
            "z[b+4..6] (shirt h/s/v) are unused: shirt color is fixed per "
            "lane in both themes. R^2 on those dims is uninformative."
        ),
    }
    with open(os.path.join(args.out_root, "latent_layout.json"), "w") as f:
        json.dump(layout, f, indent=2)

    print(f"Wrote Z {Z.shape} to {args.out_root}/Z.npy")
    print(f"Wrote latent layout to {args.out_root}/latent_layout.json\n")

    theme_module = {"island": svg_island, "western": svg_western}
    for t in args.themes:
        render_theme(theme_module[t], t, Z, args.out_root, args.output_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
