"""
Figure 4a: Reverse predictability concept cartoon.

4 panels:
  (a) 2D latent space Z         : scatter of grid points in (z1, z2)
  (b) Rendered images X         : 2D grid of face cartoons at the same points
  (c) Linear representation     : dots on a flat 2D plane embedded in 3D
  (d) Nonlinear representation  : dots on a curved 2D surface in 3D
                                  (linear projection onto y1,y2 still recovers
                                   z, but y3 also encodes a nonlinear function
                                   of z)

Latent dims varied: z[8] mouth_curve (sad↔happy), z[0] face_radius (small↔big).
All other latents fixed at the Figure-1 defaults.
"""
import io
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cairosvg
from PIL import Image

CANVAS_W = 224
CANVAS_H = 224


# ---------- Face SVG generator (from Fig 1 code) ----------
def hsv_to_hex(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def make_mouth_path(cx, cy, width, curve):
    x1 = cx - width / 2
    x2 = cx + width / 2
    control_y = cy + curve
    return f"M {x1:.2f} {cy:.2f} Q {cx:.2f} {control_y:.2f} {x2:.2f} {cy:.2f}"


def generate_face_svg_from_z(z, framed=False):
    face_radius = 55 + z[0] * (75 - 55)
    cx = (face_radius + 10) + z[1] * ((CANVAS_W - face_radius - 10) - (face_radius + 10))
    cy = (face_radius + 10) + z[2] * ((CANVAS_H - face_radius - 10) - (face_radius + 10))
    eye_radius = 4 + z[3] * (10 - 4)
    eye_spacing = 15 + z[4] * (30 - 15)
    eye_y_offset = 10 + z[5] * (25 - 10)
    mouth_width = 25 + z[6] * (50 - 25)
    mouth_y_offset = 15 + z[7] * (30 - 15)
    mouth_curve = -20 + z[8] * 40.0

    left_eye_cx, left_eye_cy = cx - eye_spacing, cy - eye_y_offset
    right_eye_cx, right_eye_cy = cx + eye_spacing, cy - eye_y_offset
    mouth_cx, mouth_cy = cx, cy + mouth_y_offset

    skin_hex = hsv_to_hex(z[9], 0.5 + z[10] * 0.5, 0.5 + z[11] * 0.5)
    eye_hex = hsv_to_hex(z[12], 0.5 + z[13] * 0.5, 0.5 + z[14] * 0.5)
    mouth_path = make_mouth_path(mouth_cx, mouth_cy, mouth_width, mouth_curve)

    bg = ""
    if framed:
        bg = (
            '<rect width="100%" height="100%" fill="white" />'
            '<rect x="2" y="2" width="220" height="220" '
            'fill="none" stroke="#444" stroke-width="3" />'
        )

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


# Defaults matching Fig 1
DEFAULT_Z = np.full(15, 0.5)
DEFAULT_Z[9]  = 0.08   # warm skin hue
DEFAULT_Z[10] = 0.55
DEFAULT_Z[11] = 0.85
DEFAULT_Z[12] = 0.6    # blue eyes
DEFAULT_Z[13] = 0.9
DEFAULT_Z[14] = 0.0


# ---------- Latent grid ----------
LATENT_DIM_1 = 8   # mouth_curve: sad → happy   (mapped to z1)
LATENT_DIM_2 = 0   # face_radius: small → big   (mapped to z2)

N = 5
z1_vals = np.linspace(0.12, 0.88, N)
z2_vals = np.linspace(0.12, 0.88, N)
Z1, Z2 = np.meshgrid(z1_vals, z2_vals)


# ---------- Embedding hypotheses (z -> y in R^3) ----------
def linear_y3(z1, z2):
    # flat plane in y3
    return np.full_like(z1, 0.5)


def nonlinear_y3(z1, z2):
    # parabolic bowl: y3 minimised at the centre of latent space, rises at edges.
    # this is monotone-nothing-in-particular as a function of (z1,z2), clearly
    # nonlinear, but (z1,z2) is still recoverable by linear projection onto
    # (y1,y2) = (z1,z2).
    return 0.20 + 0.55 * 2.0 * ((z1 - 0.5) ** 2 + (z2 - 0.5) ** 2)


# ---------- Plot ----------
plt.rcParams["font.size"] = 11

fig = plt.figure(figsize=(19, 4.8))
gs = fig.add_gridspec(1, 4, wspace=0.32, left=0.035, right=0.985, top=0.90, bottom=0.10)

# -- (a) 2D latent space --
ax_a = fig.add_subplot(gs[0, 0])
ax_a.scatter(Z1.flatten(), Z2.flatten(),
             s=70, c="black", edgecolor="white", linewidth=0.6, zorder=3)
ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
ax_a.set_xticks([0, 1]); ax_a.set_yticks([0, 1])
ax_a.set_xlabel(r"$z_1$", fontsize=13)
ax_a.set_ylabel(r"$z_2$", fontsize=13)
ax_a.set_title(r"(a) Latent space  $\mathbf{z}$", fontsize=14, pad=10)
ax_a.set_aspect("equal")
for sp in ax_a.spines.values():
    sp.set_linewidth(1.0)

# -- (b) Rendered face grid --
ax_b = fig.add_subplot(gs[0, 1])
for i in range(N):           # row index -> z2
    for j in range(N):       # col index -> z1
        z1 = z1_vals[j]
        z2 = z2_vals[i]
        z = DEFAULT_Z.copy()
        z[LATENT_DIM_1] = z1
        z[LATENT_DIM_2] = z2
        svg = generate_face_svg_from_z(z, framed=False)
        img = svg_to_array(svg, size=200)
        ab = AnnotationBbox(OffsetImage(img, zoom=0.27),
                            (z1, z2), frameon=False, pad=0.0)
        ax_b.add_artist(ab)
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)
ax_b.set_xticks([0, 1]); ax_b.set_yticks([0, 1])
ax_b.set_xlabel(r"$z_1$", fontsize=13)
ax_b.set_ylabel(r"$z_2$", fontsize=13)
ax_b.set_title(r"(b) Rendered images  $\mathbf{x}$", fontsize=14, pad=10)
ax_b.set_aspect("equal")
for sp in ax_b.spines.values():
    sp.set_linewidth(1.0)


def style_3d_axes(ax, title):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.set_xlabel(r"$y_1$", fontsize=12, labelpad=-4)
    ax.set_ylabel(r"$y_2$", fontsize=12, labelpad=-4)
    ax.set_zlabel(r"$y_3$", fontsize=12, labelpad=-2)
    ax.tick_params(axis="both", which="major", labelsize=9, pad=0)
    ax.set_title(title, fontsize=14, pad=10)
    ax.view_init(elev=22, azim=-58)
    # leave a bit more room on the right so the y3 label & ticks aren't clipped
    ax.set_box_aspect(None, zoom=0.92)


# -- (c) Linear representation: flat plane --
ax_c = fig.add_subplot(gs[0, 2], projection="3d")
yy1, yy2 = np.meshgrid(np.linspace(0.04, 0.96, 30), np.linspace(0.04, 0.96, 30))
yy3_lin = linear_y3(yy1, yy2)
ax_c.plot_surface(yy1, yy2, yy3_lin,
                  alpha=0.45, color="#3B7DB6", linewidth=0,
                  rstride=1, cstride=1, antialiased=True)
# grid lines on the surface for legibility
for i in range(N):
    ax_c.plot(z1_vals, [z2_vals[i]] * N, [0.5] * N,
              color="#1f3b58", alpha=0.6, linewidth=0.7, zorder=4)
    ax_c.plot([z1_vals[i]] * N, z2_vals, [0.5] * N,
              color="#1f3b58", alpha=0.6, linewidth=0.7, zorder=4)
ax_c.scatter(Z1, Z2, linear_y3(Z1, Z2),
             c="black", s=26, depthshade=False, zorder=10,
             edgecolor="white", linewidth=0.6)
style_3d_axes(ax_c, "(c) Linear representation")

# -- (d) Nonlinear representation: curved surface --
ax_d = fig.add_subplot(gs[0, 3], projection="3d")
yy3_nl = nonlinear_y3(yy1, yy2)
ax_d.plot_surface(yy1, yy2, yy3_nl,
                  alpha=0.45, color="#D85A30", linewidth=0,
                  rstride=1, cstride=1, antialiased=True)
for i in range(N):
    line_y3 = nonlinear_y3(z1_vals, np.full(N, z2_vals[i]))
    ax_d.plot(z1_vals, [z2_vals[i]] * N, line_y3,
              color="#5a2410", alpha=0.6, linewidth=0.7, zorder=4)
    line_y3 = nonlinear_y3(np.full(N, z1_vals[i]), z2_vals)
    ax_d.plot([z1_vals[i]] * N, z2_vals, line_y3,
              color="#5a2410", alpha=0.6, linewidth=0.7, zorder=4)
ax_d.scatter(Z1, Z2, nonlinear_y3(Z1, Z2),
             c="black", s=26, depthshade=False, zorder=10,
             edgecolor="white", linewidth=0.6)
style_3d_axes(ax_d, "(d) Nonlinear representation")


out_path = "svg_faces_reverse_concept.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")