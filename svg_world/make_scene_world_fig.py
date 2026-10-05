"""
Scene-World latent illustration figure.

Two side-by-side panels (island left, western right). Each shows the canonical
scene with all latents at z=0.5 (everyone in the middle of their lane), and
arrows overlay the spatial extent each position latent can reach.

Both worlds share an identical 32-dim latent layout:
    z[7i + 0]  : person i lane-x      (i = 0..3)
    z[7i + 1]  : person i lane-z
    z[7i + 2]  : person i facing      (4-way discrete; not drawn as arrow)
    z[7i + 3]  : person i skin-hue    (color; not drawn as arrow)
    z[7i + 4..6]: unused shirt h/s/v
    z[28]      : animal-A lane-x      (dolphin / brown horse)
    z[29]      : animal-A lane-z
    z[30]      : animal-B lane-x      (turtle  / white horse)
    z[31]      : animal-B lane-z

The same arrows overlay both worlds because both share the layout.
This is the visual that earns "SVG World" its name as a benchmark and motivates
the cross-domain transfer experiment in section 9.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import patheffects

sys.path.insert(0, ".")
import svg_island
import svg_western


# Plot size factor; the whole figure scales with this through figsize,
# with no individual font/linewidth tweaks.
PLOT_SIZE = 0.55

# Canvas dims from the SVG modules (both share the same 600x420 canvas)
CW, CH = svg_island.CANVAS_W, svg_island.CANVAS_H

# Color scheme: people positions in blue, animals in orange.
PERSON_COLOR = "#1f4e9b"
ANIMAL_COLOR = "#d97a1a"


def project(x, y, z, mod):
    """Project a scene-coord point to image-pixel coords using the module's
    own projection. Image y is downward, matching cairosvg output."""
    sx, sy = mod.project(x, y, z)
    return sx, sy


def lane_corners(lane, mod, y=0.0):
    """Return projected screen-pixel corners of a lane rectangle on the ground."""
    xs = [lane["x_min"], lane["x_max"], lane["x_max"], lane["x_min"]]
    zs = [lane["z_min"], lane["z_min"], lane["z_max"], lane["z_max"]]
    pts = [project(x, y, z, mod) for x, z in zip(xs, zs)]
    return np.array(pts)


def axis_indicator_endpoints(anchor_xz, mod, length_units=0.6,
                              start_frac=0.5):
    """Two pairs of screen-coord endpoints, one for the x-latent and one for
    the z-latent. With every latent at 0 the object sits at the lane minimum
    in (x, z); the arrows point in the +x and +z directions, indicating where
    increasing the latent moves the object.

    `start_frac` shifts the arrow's starting point that fraction of the way
    toward its tip, so the arrow doesn't visually emerge from the object."""
    ax, az = anchor_xz
    s = start_frac * length_units
    e = length_units
    x_endpts = (project(ax + s, 0.0, az, mod),
                project(ax + e, 0.0, az, mod))
    z_endpts = (project(ax, 0.0, az + s, mod),
                project(ax, 0.0, az + e, mod))
    return x_endpts, z_endpts


def draw_arrow(ax, p0, p1, color, label=None, label_offset=(0, 0)):
    """Single-headed arrow from p0 to p1 in data coordinates. Body is a
    Line2D, arrowhead is a small filled polygon drawn in data coords. This
    avoids matplotlib's display-coord arrow scaling issues at small figsize."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return
    unit = direction / norm
    perp = np.array([-unit[1], unit[0]])

    head_len = 9.0  # data-coord pixels
    head_w   = 5.5
    # Body runs from anchor to the base of the arrowhead
    body1 = p1 - unit * head_len * 0.6
    line = Line2D([p0[0], body1[0]], [p0[1], body1[1]],
                  color=color, linewidth=1.6, zorder=10,
                  solid_capstyle="round")
    line.set_path_effects([
        patheffects.Stroke(linewidth=3.6, foreground="white"),
        patheffects.Normal(),
    ])
    ax.add_line(line)

    # Single arrowhead at p1
    base_center = p1 - unit * head_len
    verts = np.array([
        p1,
        base_center + perp * head_w,
        base_center - perp * head_w,
    ])
    head = Polygon(verts, closed=True, facecolor=color,
                   edgecolor="white", linewidth=1.0, zorder=11)
    ax.add_patch(head)

    if label is not None:
        mid = ((p0[0] + p1[0]) / 2 + label_offset[0],
               (p0[1] + p1[1]) / 2 + label_offset[1])
        ax.text(mid[0], mid[1], label,
                color=color, ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=12,
                path_effects=[
                    patheffects.Stroke(linewidth=3, foreground="white"),
                    patheffects.Normal(),
                ])


def render_canonical_image(mod):
    """Render with all latents at 0 (one extreme of each lane). Arrows then
    point in the single direction each latent can take the object."""
    z = np.zeros(mod.LATENT_DIM)
    svg = mod.generate_scene_svg(z)
    return mod.svg_to_png_array(svg)


def draw_world(ax, mod, title, animal_a_label, animal_b_label):
    img = render_canonical_image(mod)
    ax.imshow(img, extent=(0, CW, CH, 0), interpolation="bicubic")
    ax.set_xlim(0, CW)
    ax.set_ylim(CH, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=4)

    # Each object's anchor: lane minimum (where the object actually is when
    # all latents are zero). Arrows point in the +x and +z directions, showing
    # which way the latent moves things.
    person_anchors = [(lane["x_min"], lane["z_min"]) for lane in mod.PERSON_LANES]

    if mod is svg_island:
        a_lane, b_lane = mod.DOLPHIN_LANE, mod.TURTLE_LANE
    else:
        a_lane, b_lane = mod.HORSE_BROWN_LANE, mod.HORSE_WHITE_LANE
    animal_anchors = [(a_lane["x_min"], a_lane["z_min"]),
                      (b_lane["x_min"], b_lane["z_min"])]

    # People: 4 of them. Arrow length = 1.1 scene units in each direction.
    for i, anchor in enumerate(person_anchors):
        x_end, z_end = axis_indicator_endpoints(anchor, mod, length_units=1.1)
        draw_arrow(ax, x_end[0], x_end[1], PERSON_COLOR)
        draw_arrow(ax, z_end[0], z_end[1], PERSON_COLOR)
        # Centered label above each person's head.
        head_x, head_y = project(anchor[0], 1.95, anchor[1], mod)
        ax.text(head_x, head_y - 4, f"p{i}",
                color=PERSON_COLOR, ha="center", va="bottom",
                fontsize=11, fontweight="bold", zorder=12,
                path_effects=[
                    patheffects.Stroke(linewidth=3, foreground="white"),
                    patheffects.Normal(),
                ])

    # Animals: same arrow length. Two animals sit at canvas-bottom-left and
    # canvas-bottom-right. Their labels sit just below-and-outside, so each
    # is comfortably inside the panel edge but doesn't overlap the other's.
    for i, (anchor, lbl) in enumerate(zip(animal_anchors,
                                          [animal_a_label, animal_b_label])):
        x_end, z_end = axis_indicator_endpoints(anchor, mod, length_units=1.1)
        draw_arrow(ax, x_end[0], x_end[1], ANIMAL_COLOR)
        draw_arrow(ax, z_end[0], z_end[1], ANIMAL_COLOR)
        cx, cy = project(anchor[0], 0, anchor[1], mod)
        # Place each animal's label on the OUTSIDE of the animal (away from
        # canvas center). Leftmost: label ends just left of the animal
        # (anchored "right"); rightmost: label starts just right of the
        # animal (anchored "left"). This keeps the two labels far apart and
        # comfortably inside the panel edges.
        if i == 0:
            x_offset, ha = -6, "right"
        else:
            x_offset, ha = 6, "left"
        ax.text(cx + x_offset, cy + 28, lbl,
                color=ANIMAL_COLOR, ha=ha, va="top",
                fontsize=10, fontweight="bold", zorder=12,
                path_effects=[
                    patheffects.Stroke(linewidth=3, foreground="white"),
                    patheffects.Normal(),
                ])


def main():
    fig = plt.figure(figsize=PLOT_SIZE * np.array((22, 5.6)))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 0.32, 1.0],
        wspace=0.04,
    )
    ax_left   = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_right  = fig.add_subplot(gs[0, 2])

    draw_world(ax_left,  svg_island,  "AquaWorld", "dolphin",     "turtle")
    draw_world(ax_right, svg_western, "WestWorld", "brown horse", "white horse")

    # Center legend column: arrow swatches + descriptive text for each
    # latent group, between the two scene panels.
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.set_aspect("auto")
    ax_legend.set_xticks([]); ax_legend.set_yticks([])
    for s in ax_legend.spines.values():
        s.set_visible(False)

    # Title for the legend column
    ax_legend.text(0.5, 0.96, "shared latents",
                   ha="center", va="top", fontsize=11, fontweight="bold")

    # Layout: two groups (people / animals), each with a header and two rows
    # (horizontal / depth), each row showing a small arrow swatch + label.
    def draw_legend_arrow(x, y, color, direction):
        """Draw a small arrow swatch at axis-frac (x, y). direction = 'h' or 'z'."""
        # Convert to figure-fraction-ish data coords (axis is unit-square).
        if direction == "h":
            # Horizontal right-pointing arrow
            p0 = np.array([x - 0.04, y])
            p1 = np.array([x + 0.04, y])
        else:
            # Up-and-left, mimicking the +z projection direction
            dx, dy = -np.sin(np.pi / 6) * 0.05, np.cos(np.pi / 6) * 0.05
            p0 = np.array([x - dx * 0.5, y - dy * 0.5])
            p1 = np.array([x + dx * 0.5, y + dy * 0.5])

        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon
        unit = (p1 - p0) / np.linalg.norm(p1 - p0)
        perp = np.array([-unit[1], unit[0]])
        head_len = 0.022
        head_w   = 0.012
        body1 = p1 - unit * head_len * 0.6
        line = Line2D([p0[0], body1[0]], [p0[1], body1[1]],
                      color=color, linewidth=1.6, zorder=10,
                      solid_capstyle="round",
                      transform=ax_legend.transAxes)
        ax_legend.add_line(line)
        base_center = p1 - unit * head_len
        verts = np.array([
            p1,
            base_center + perp * head_w,
            base_center - perp * head_w,
        ])
        head = Polygon(verts, closed=True, facecolor=color,
                       edgecolor=color, linewidth=0.5, zorder=11,
                       transform=ax_legend.transAxes)
        ax_legend.add_patch(head)

    # People group
    ax_legend.text(0.5, 0.84, "person 1$-$4 position",
                   ha="center", va="center", fontsize=10,
                   color=PERSON_COLOR, fontweight="bold")
    draw_legend_arrow(0.16, 0.74, PERSON_COLOR, "h")
    ax_legend.text(0.24, 0.74, "horizontal",
                   ha="left", va="center", fontsize=9, color=PERSON_COLOR)
    draw_legend_arrow(0.16, 0.64, PERSON_COLOR, "z")
    ax_legend.text(0.24, 0.64, "depth",
                   ha="left", va="center", fontsize=9, color=PERSON_COLOR)

    # Animal group
    ax_legend.text(0.5, 0.46, "animal 1$-$2 position",
                   ha="center", va="center", fontsize=10,
                   color=ANIMAL_COLOR, fontweight="bold")
    draw_legend_arrow(0.16, 0.36, ANIMAL_COLOR, "h")
    ax_legend.text(0.24, 0.36, "horizontal",
                   ha="left", va="center", fontsize=9, color=ANIMAL_COLOR)
    draw_legend_arrow(0.16, 0.26, ANIMAL_COLOR, "z")
    ax_legend.text(0.24, 0.26, "depth",
                   ha="left", va="center", fontsize=9, color=ANIMAL_COLOR)

    fig.savefig("scene_world_latents.pdf",
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig("scene_world_latents.png",
                bbox_inches="tight", pad_inches=0.05, dpi=200)
    print("wrote scene_world_latents.{pdf,png}")


if __name__ == "__main__":
    main()
