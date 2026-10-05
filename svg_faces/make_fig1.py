"""
Figure 1: SVG world.

4 subplots, each a 3D unit-cube plot with face renderings at 7 corners
(all 8 corners minus (1,1,1)). Each subplot varies one group of 3 latents
while holding the other 12 latents fixed at 0.5.

Panels:
  (a) Face latents       : z[0] face_radius, z[1] cx, z[2] cy
  (b) Eye latents        : z[3] eye_radius, z[4] eye_spacing, z[5] eye_y_offset
  (c) Mouth latents      : z[6] mouth_width, z[7] mouth_y_offset, z[8] mouth_curve
  (d) Skin color (HSV)   : z[9] hue, z[10] saturation, z[11] value
"""
import io
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import proj3d  # noqa: F401  (registers 3d projection)
import cairosvg
from PIL import Image

CANVAS_W = 224
CANVAS_H = 224


# ---------- SVG generation (mirrors user's generate_face_svg) ----------
def hsv_to_hex(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def make_mouth_path(cx, cy, width, curve):
    x1 = cx - width / 2
    x2 = cx + width / 2
    control_y = cy + curve
    return f"M {x1:.2f} {cy:.2f} Q {cx:.2f} {control_y:.2f} {x2:.2f} {cy:.2f}"


def generate_face_svg_from_z(z, framed=True):
    """Generate face SVG given 15-d z in [0,1]. Mirrors user's generator.

    If framed=True, render an opaque white canvas with a thin dark border so
    the face's position within its bounding frame is visible.
    """
    # Geometry: face
    face_radius = 55 + z[0] * (75 - 55)
    cx = (face_radius + 10) + z[1] * ((CANVAS_W - face_radius - 10) - (face_radius + 10))
    cy = (face_radius + 10) + z[2] * ((CANVAS_H - face_radius - 10) - (face_radius + 10))

    # Geometry: eyes
    eye_radius = 4 + z[3] * (10 - 4)
    eye_spacing = 15 + z[4] * (30 - 15)
    eye_y_offset = 10 + z[5] * (25 - 10)

    # Geometry: mouth
    mouth_width = 25 + z[6] * (50 - 25)
    mouth_y_offset = 15 + z[7] * (30 - 15)
    mouth_curve = -20 + z[8] * 40.0

    left_eye_cx, left_eye_cy = cx - eye_spacing, cy - eye_y_offset
    right_eye_cx, right_eye_cy = cx + eye_spacing, cy - eye_y_offset
    mouth_cx, mouth_cy = cx, cy + mouth_y_offset

    # Colors (skin + eye), each h in [0,1], s,v in [0.5,1]
    skin_hex = hsv_to_hex(z[9], 0.5 + z[10] * 0.5, 0.5 + z[11] * 0.5)
    eye_hex = hsv_to_hex(z[12], 0.5 + z[13] * 0.5, 0.5 + z[14] * 0.5)

    mouth_path = make_mouth_path(mouth_cx, mouth_cy, mouth_width, mouth_curve)

    if framed:
        # Opaque white fill + thin dark border drawn just inside the canvas edge
        bg = (
            '<rect width="100%" height="100%" fill="white" />'
            '<rect x="2" y="2" width="220" height="220" '
            'fill="none" stroke="#444" stroke-width="3" />'
        )
    else:
        bg = ""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">
  {bg}
  <g id="face">
    <circle cx="{cx}" cy="{cy}" r="{face_radius}" fill="{skin_hex}" stroke="#111" stroke-width="2" />
    <circle cx="{left_eye_cx}" cy="{left_eye_cy}" r="{eye_radius}" fill="{eye_hex}" />
    <circle cx="{right_eye_cx}" cy="{right_eye_cy}" r="{eye_radius}" fill="{eye_hex}" />
    <path d="{mouth_path}" fill="none" stroke="#111" stroke-width="3" stroke-linecap="round" />
  </g>
</svg>"""


def svg_to_array(svg_str, size=200):
    png_bytes = cairosvg.svg2png(
        bytestring=svg_str.encode("utf-8"),
        output_width=size,
        output_height=size,
    )
    return np.array(Image.open(io.BytesIO(png_bytes)))


# ---------- Figure setup ----------
PANELS = [
    {
        "title": "(a) Face latents",
        "indices": [0, 1, 2],
        "labels": ["face radius", "center x", "center y"],
    },
    {
        "title": "(b) Eye latents",
        "indices": [3, 4, 5],
        "labels": ["eye radius", "eye spacing", "eye y-offset"],
    },
    {
        "title": "(c) Mouth latents",
        "indices": [6, 7, 8],
        "labels": ["mouth width", "mouth y-offset", "mouth curve"],
    },
    {
        "title": "(d) Skin color (HSV)",
        "indices": [9, 10, 11],
        "labels": ["hue", "saturation", "value"],
    },
]

# 7 corners of unit cube, omitting (1,1,1)
CORNERS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
]

# Cube edges (12 of them)
CUBE_EDGES = [
    ((0, 0, 0), (1, 0, 0)),
    ((0, 0, 0), (0, 1, 0)),
    ((0, 0, 0), (0, 0, 1)),
    ((1, 0, 0), (1, 1, 0)),
    ((1, 0, 0), (1, 0, 1)),
    ((0, 1, 0), (1, 1, 0)),
    ((0, 1, 0), (0, 1, 1)),
    ((0, 0, 1), (1, 0, 1)),
    ((0, 0, 1), (0, 1, 1)),
    ((1, 1, 0), (1, 1, 1)),
    ((1, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (1, 1, 1)),
]

DEFAULT_Z = np.full(15, 0.5)
# Override defaults so eyes contrast strongly with skin in panels (a)-(c).
# (Panel (d) overrides z[9..11] anyway, so skin defaults don't affect it.)
DEFAULT_Z[9] = 0.08   # skin hue ~ warm orange
DEFAULT_Z[10] = 0.55  # skin saturation
DEFAULT_Z[11] = 0.85  # skin value (lightness)
DEFAULT_Z[12] = 0.6   # eye hue ~ blue
DEFAULT_Z[13] = 0.9   # eye saturation high
DEFAULT_Z[14] = 0.0   # eye value low -> dark blue eyes
ZOOM = 0.22

fig = plt.figure(figsize=(18, 5.0))

for p_idx, panel in enumerate(PANELS):
    ax = fig.add_subplot(1, 4, p_idx + 1, projection="3d")
    ax.set_title(panel["title"], fontsize=15, fontweight="bold", pad=10)

    # Cube wireframe
    for s, e in CUBE_EDGES:
        xs, ys, zs = zip(s, e)
        ax.plot(xs, ys, zs, color="#888", alpha=0.55, linewidth=1.0)

    # Mark the 7 image corners; mark (1,1,1) distinctly as the "no image" corner.
    cx_pts = [c[0] for c in CORNERS]
    cy_pts = [c[1] for c in CORNERS]
    cz_pts = [c[2] for c in CORNERS]
    ax.scatter(cx_pts, cy_pts, cz_pts, c="black", s=18, zorder=2)
    ax.scatter([1], [1], [1], facecolor="white", edgecolor="#888", s=30, zorder=2, linewidth=1.0)

    # Axes styling
    ax.set_xlabel(panel["labels"][0], fontsize=11, labelpad=8)
    ax.set_ylabel(panel["labels"][1], fontsize=11, labelpad=8)
    # Manually place the z-axis label so it isn't occluded by face boxes.
    # Positioned at the upper-left of each subplot in axes coords, vertically.
    ax.text2D(
        -0.04, 0.55, panel["labels"][2],
        transform=ax.transAxes,
        rotation=90, ha="center", va="center",
        fontsize=11, zorder=10,
    )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.tick_params(axis="both", which="major", labelsize=9, pad=2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_zlim(-0.05, 1.05)
    ax.view_init(elev=18, azim=35)

    # Place a face image at each corner
    for corner in CORNERS:
        z_vec = DEFAULT_Z.copy()
        for axis_i, latent_i in enumerate(panel["indices"]):
            z_vec[latent_i] = corner[axis_i]

        svg_str = generate_face_svg_from_z(z_vec)
        img = svg_to_array(svg_str, size=220)

        # Project 3D corner -> 2D data coords of this 3D axis
        x2, y2, _ = proj3d.proj_transform(corner[0], corner[1], corner[2], ax.get_proj())

        ab = AnnotationBbox(
            OffsetImage(img, zoom=ZOOM),
            (x2, y2),
            xycoords="data",
            frameon=False,
            pad=0.0,
        )
        ax.add_artist(ab)

plt.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.06, wspace=0.05)

out_path = "svg_faces_latents.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")