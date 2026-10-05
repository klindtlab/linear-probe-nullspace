"""
SVG-world village generator.

Oblique projection (cabinet-style, 30° angle). Scene coordinates:
  +x : east (right)
  +y : up
  +z : north (depth, upper-left on screen)

Latents (32 dims, all in [0,1]); only people and animals vary. The village
(island, 3 houses, 4 trees, fixed sky/water/grass colors, decorative scatter,
wave marks) is fixed.

Latent layout:
   z[0:7]    person 0  (lane-x, lane-z, facing, skin-h, _shirt h/s/v unused_)
   z[7:14]   person 1
   z[14:21]  person 2
   z[21:28]  person 3
   z[28:30]  dolphin (lane-x, lane-z), swims in the front-left water
   z[30:32]  turtle  (lane-x, lane-z), swims in the front-right water
"""

import io
import os
import colorsys
import numpy as np
import cairosvg
from PIL import Image
import matplotlib.pyplot as plt

# ---------- Projection ----------
CANVAS_W = 600
CANVAS_H = 420
SCALE = 50  # pixels per scene unit
OFFSET_X = CANVAS_W / 2
OFFSET_Y = 230  # vertical anchor; nudged up so the island sits high and there's water below for animals
COS30 = np.cos(np.pi / 6)
SIN30 = np.sin(np.pi / 6)


def project(x, y, z):
    """Oblique projection. +x→right, +y→up (screen), +z→upper-left."""
    sx = (x - z * SIN30) * SCALE
    sy = -(y + z * COS30) * SCALE
    return OFFSET_X + sx, OFFSET_Y + sy


# ---------- Color helpers ----------
def hsv_hex(h, s, v):
    h = h % 1.0
    s = max(0.0, min(1.0, s))
    v = max(0.0, min(1.0, v))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def shade(hex_color, factor=0.75):
    """Multiply RGB by factor for a darker/lighter variant."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


# ---------- SVG primitive builders ----------
def poly3(corners, fill, stroke="#222", sw=1.0):
    pts = " ".join(f"{project(*p)[0]:.1f},{project(*p)[1]:.1f}" for p in corners)
    return (f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linejoin="round" />')


def circle3(x, y, z, r, fill, stroke="#222", sw=1.0):
    cx, cy = project(x, y, z)
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * SCALE:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />')


# ---------- Object renderers ----------
def render_ground(half_size, color):
    return poly3(
        [(-half_size, 0, -half_size), (half_size, 0, -half_size),
         (half_size, 0, half_size), (-half_size, 0, half_size)],
        fill=color, stroke=shade(color, 0.7), sw=1.0,
    )


def shadow(x, z, rx, rz=None):
    """Soft ground shadow centered at (x, 0, z). rx/rz in scene units."""
    rz = rz if rz is not None else rx * 0.6
    cx, cy = project(x, 0, z)
    # Approximate the projected ellipse by scaling its axes; close enough at this angle.
    return (f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" '
            f'rx="{rx * SCALE:.1f}" ry="{rz * SCALE * COS30:.1f}" '
            f'fill="rgba(0,0,0,0.18)" />')


def render_grass_tuft(x, z, color):
    cx, cy = project(x, 0, z)
    return (f'<path d="M{cx - 4:.1f},{cy} L{cx - 2:.1f},{cy - 5} '
            f'M{cx:.1f},{cy} L{cx:.1f},{cy - 6} '
            f'M{cx + 4:.1f},{cy} L{cx + 2:.1f},{cy - 5}" '
            f'stroke="{color}" stroke-width="1.5" stroke-linecap="round" fill="none" />')


def render_flower(x, z, petal_color):
    cx, cy = project(x, 0.05, z)
    return (
        f'<g>'
        f'<line x1="{cx:.1f}" y1="{cy + 6:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
        f'stroke="#3a6b1f" stroke-width="1.5" />'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{petal_color}" '
        f'stroke="#222" stroke-width="0.5" />'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1" fill="#f5d000" />'
        f'</g>'
    )


def render_house(x, z, wx=1.0, wz=0.65, h=1.25, wall="#e8d290", roof="#a23838", door="#5d4037"):
    """House centered on (x, 0, z). wx = x half-width, wz = z half-depth.
    From a south-east-above oblique view, the visible faces are: south wall
    (front), BOTH side walls (east extends up-and-left from south-east corner;
    west extends up-and-left from south-west corner into the back-left of the
    silhouette), both roof slopes, and BOTH gable triangles."""
    parts = []
    roof_h = 0.50  # steeper roof = cottage / dollhouse vibe
    eave = 0.10
    # Window/shutter constants (used for both south- and west-wall windows)
    win_h = 0.13
    shutter_w = 0.05
    shutter_pad = 0.012
    shutter_color = shade(roof, 0.85)
    # East wall (right side): overlaps south wall in screen, must draw first.
    parts.append(poly3(
        [(x + wx, 0, z - wz), (x + wx, 0, z + wz),
         (x + wx, h, z + wz), (x + wx, h, z - wz)],
        fill=shade(wall, 0.78),
    ))
    # West wall (left side): extends up-and-left from the south-west corner of
    # the south wall. Without this, grass+sky show through the building's
    # left silhouette.
    parts.append(poly3(
        [(x - wx, 0, z - wz), (x - wx, 0, z + wz),
         (x - wx, h, z + wz), (x - wx, h, z - wz)],
        fill=shade(wall, 0.78),
    ))
    # West-wall window: one centered window with shutters. The wall lies in the
    # y-z plane at constant x, so shutters flank the window in the z direction.
    west_x = x - wx
    ww_y = h - 0.4
    ww_zc = z  # centered along wall depth
    for sh_lo_z, sh_hi_z in [
        (ww_zc - win_h - shutter_pad - shutter_w, ww_zc - win_h - shutter_pad),
        (ww_zc + win_h + shutter_pad,             ww_zc + win_h + shutter_pad + shutter_w),
    ]:
        parts.append(poly3(
            [(west_x, ww_y - win_h, sh_lo_z),
             (west_x, ww_y - win_h, sh_hi_z),
             (west_x, ww_y + win_h, sh_hi_z),
             (west_x, ww_y + win_h, sh_lo_z)],
            fill=shutter_color, sw=0.8,
        ))
    parts.append(poly3(
        [(west_x, ww_y - win_h, ww_zc - win_h),
         (west_x, ww_y - win_h, ww_zc + win_h),
         (west_x, ww_y + win_h, ww_zc + win_h),
         (west_x, ww_y + win_h, ww_zc - win_h)],
        fill="#a8d8ec", sw=1.0,
    ))
    # Vertical mullion through the west window (line of constant x, constant z, y varies)
    wmt_x, wmt_y = project(west_x, ww_y - win_h, ww_zc)
    wmb_x, wmb_y = project(west_x, ww_y + win_h, ww_zc)
    parts.append(f'<line x1="{wmt_x:.1f}" y1="{wmt_y:.1f}" x2="{wmb_x:.1f}" y2="{wmb_y:.1f}" stroke="#222" stroke-width="1" />')
    # North roof slope: back of roof, behind the south wall.
    parts.append(poly3(
        [(x - wx, h, z + wz), (x + wx, h, z + wz),
         (x + wx, h + roof_h, z), (x - wx, h + roof_h, z)],
        fill=shade(roof, 0.62),
    ))
    # South wall
    parts.append(poly3(
        [(x - wx, 0, z - wz), (x + wx, 0, z - wz),
         (x + wx, h, z - wz), (x - wx, h, z - wz)],
        fill=wall,
    ))
    # Door on south wall
    door_w = 0.18
    door_h = 0.55
    parts.append(poly3(
        [(x - door_w, 0, z - wz), (x + door_w, 0, z - wz),
         (x + door_w, door_h, z - wz), (x - door_w, door_h, z - wz)],
        fill=door,
    ))
    # Door knob
    kx, ky = project(x + door_w * 0.7, door_h * 0.55, z - wz)
    parts.append(f'<circle cx="{kx:.1f}" cy="{ky:.1f}" r="2" fill="#f0c419" />')
    # Two windows on south wall, with cute shutters
    for win_x in (x - 0.5, x + 0.5):
        # Shutters: one on each side of the window. Drawn BEFORE the window glass
        # so the shutters appear flush with the wall (not on top of glass).
        for sh_left, sh_right in [
            (win_x - win_h - shutter_pad - shutter_w, win_x - win_h - shutter_pad),
            (win_x + win_h + shutter_pad,             win_x + win_h + shutter_pad + shutter_w),
        ]:
            parts.append(poly3(
                [(sh_left,  h - 0.4 - win_h, z - wz),
                 (sh_right, h - 0.4 - win_h, z - wz),
                 (sh_right, h - 0.4 + win_h, z - wz),
                 (sh_left,  h - 0.4 + win_h, z - wz)],
                fill=shutter_color, sw=0.8,
            ))
        # Window glass
        parts.append(poly3(
            [(win_x - win_h, h - 0.4 - win_h, z - wz),
             (win_x + win_h, h - 0.4 - win_h, z - wz),
             (win_x + win_h, h - 0.4 + win_h, z - wz),
             (win_x - win_h, h - 0.4 + win_h, z - wz)],
            fill="#a8d8ec", sw=1.0,
        ))
        wcx, wcy = project(win_x, h - 0.4, z - wz)
        parts.append(f'<line x1="{wcx:.1f}" y1="{wcy - win_h * SCALE:.1f}" '
                     f'x2="{wcx:.1f}" y2="{wcy + win_h * SCALE:.1f}" stroke="#222" stroke-width="1" />')
    # South roof slope (front of roof)
    parts.append(poly3(
        [(x - wx, h, z - wz - eave),
         (x + wx, h, z - wz - eave),
         (x + wx, h + roof_h, z),
         (x - wx, h + roof_h, z)],
        fill=roof,
    ))
    # East gable triangle (caps east end of ridge)
    parts.append(poly3(
        [(x + wx, h, z - wz), (x + wx, h, z + wz), (x + wx, h + roof_h, z)],
        fill=shade(roof, 0.78),
    ))
    # West gable triangle (caps west end of ridge)
    parts.append(poly3(
        [(x - wx, h, z - wz), (x - wx, h, z + wz), (x - wx, h + roof_h, z)],
        fill=shade(roof, 0.78),
    ))
    # Chimney
    cw = 0.07
    cx_, cz_ = x + wx * 0.45, z + 0.05
    parts.append(poly3(
        [(cx_ - cw, h + roof_h * 0.7, cz_ - cw),
         (cx_ + cw, h + roof_h * 0.7, cz_ - cw),
         (cx_ + cw, h + roof_h * 0.7 + 0.25, cz_ - cw),
         (cx_ - cw, h + roof_h * 0.7 + 0.25, cz_ - cw)],
        fill="#7a5b4f",
    ))
    return "\n".join(parts)


def render_tree(x, z, height=1.5, trunk_color="#a08770",
                leaf_color="#3f8a3a", leaf_radius=0.6):
    """Stylized palm tree: a slim, ringed, tapered trunk with eight fronds
    radiating from the top (six in the upper hemisphere, two drooping forward)
    and a small cluster of coconuts."""
    parts = []
    # Tapered trunk
    trunk_w_base = 0.06
    trunk_w_top  = 0.04
    bl = project(x - trunk_w_base, 0, z)
    br = project(x + trunk_w_base, 0, z)
    tr = project(x + trunk_w_top,  height, z)
    tl = project(x - trunk_w_top,  height, z)
    parts.append(
        f'<polygon points="{bl[0]:.1f},{bl[1]:.1f} {br[0]:.1f},{br[1]:.1f} '
        f'{tr[0]:.1f},{tr[1]:.1f} {tl[0]:.1f},{tl[1]:.1f}" '
        f'fill="{trunk_color}" stroke="{shade(trunk_color, 0.62)}" stroke-width="1" />'
    )
    # Trunk rings (palm texture)
    for i in range(1, 5):
        ry = height * (i / 5)
        w  = trunk_w_base * (1 - 0.35 * (i / 5)) * 0.85
        rl = project(x - w, ry, z)
        rr = project(x + w, ry, z)
        parts.append(
            f'<line x1="{rl[0]:.1f}" y1="{rl[1]:.1f}" '
            f'x2="{rr[0]:.1f}" y2="{rr[1]:.1f}" '
            f'stroke="{shade(trunk_color, 0.62)}" stroke-width="0.7" />'
        )
    # Fronds: 8 elongated rotated ellipses radiating from the trunk top.
    cx_top, cy_top = project(x, height, z)
    fl = leaf_radius * SCALE * 1.5  # frond length (px)
    frond_tips = [
        (-fl * 1.05, -fl * 0.10),   # far left, just barely above horizontal
        (-fl * 0.70, -fl * 0.65),   # up-left
        (-fl * 0.22, -fl * 0.95),   # near-vertical, slight L
        ( fl * 0.22, -fl * 0.95),   # near-vertical, slight R
        ( fl * 0.70, -fl * 0.65),   # up-right
        ( fl * 1.05, -fl * 0.10),   # far right
        (-fl * 0.55,  fl * 0.40),   # drooping front-left
        ( fl * 0.55,  fl * 0.40),   # drooping front-right
    ]
    edge = shade(leaf_color, 0.62)
    for tip_dx, tip_dy in frond_tips:
        tip_x = cx_top + tip_dx
        tip_y = cy_top + tip_dy
        mid_x = (cx_top + tip_x) / 2
        mid_y = (cy_top + tip_y) / 2
        length = np.hypot(tip_dx, tip_dy)
        angle  = np.degrees(np.arctan2(tip_dy, tip_dx))
        parts.append(
            f'<ellipse cx="{mid_x:.1f}" cy="{mid_y:.1f}" '
            f'rx="{length / 2:.1f}" ry="4.4" '
            f'transform="rotate({angle:.1f}, {mid_x:.1f}, {mid_y:.1f})" '
            f'fill="{leaf_color}" stroke="{edge}" stroke-width="1" />'
        )
    # Coconuts (drawn last so they sit on top of any frond they overlap)
    parts.append(
        f'<circle cx="{cx_top - 4:.1f}" cy="{cy_top + 5:.1f}" r="2.7" '
        f'fill="#5d3f2a" stroke="#3a2418" stroke-width="0.8" />'
    )
    parts.append(
        f'<circle cx="{cx_top + 4:.1f}" cy="{cy_top + 5:.1f}" r="2.7" '
        f'fill="#5d3f2a" stroke="#3a2418" stroke-width="0.8" />'
    )
    return "\n".join(parts)


def render_person(x, z, skin_color="#f1c27d", shirt_color="#3a78b5",
                  pant_color="#2a3b6c", hair_color="#3e2c1a", facing="south"):
    """Small chibi-style character. Total height ~0.6 scene units (under house wall).
       facing ∈ {'south', 'east', 'north', 'west'} controls eye/hair placement."""
    parts = []
    head_y = 0.50
    head_r = 0.11
    body_top = 0.42
    body_bot = 0.20
    body_w = 0.10
    leg_split = 0.035

    # Legs (two)
    parts.append(poly3(
        [(x - body_w, 0, z), (x - leg_split, 0, z),
         (x - leg_split, body_bot, z), (x - body_w, body_bot, z)],
        fill=pant_color, sw=1.0,
    ))
    parts.append(poly3(
        [(x + leg_split, 0, z), (x + body_w, 0, z),
         (x + body_w, body_bot, z), (x + leg_split, body_bot, z)],
        fill=pant_color, sw=1.0,
    ))
    # Body / shirt
    parts.append(poly3(
        [(x - body_w, body_bot, z), (x + body_w, body_bot, z),
         (x + body_w, body_top, z), (x - body_w, body_top, z)],
        fill=shirt_color, sw=1.0,
    ))
    # Head (skin circle)
    hcx, hcy = project(x, head_y, z)
    rr = head_r * SCALE
    parts.append(
        f'<circle cx="{hcx:.1f}" cy="{hcy:.1f}" r="{rr:.1f}" '
        f'fill="{skin_color}" stroke="#222" stroke-width="1.0" />'
    )
    # Hair + eyes depend on facing
    eye_y = hcy + rr * 0.20
    if facing == "north":
        # Back to viewer: hair covers head fully (slightly inset), no eyes.
        parts.append(
            f'<circle cx="{hcx:.1f}" cy="{hcy:.1f}" r="{rr * 0.92:.1f}" '
            f'fill="{hair_color}" stroke="#222" stroke-width="1.0" />'
        )
    elif facing == "east":
        # Hair top + back-of-head flap on the WEST (left) side. Eyes on east (right).
        parts.append(
            f'<path d="M{hcx - rr:.1f},{hcy:.1f} '
            f'A{rr},{rr} 0 0 1 {hcx + rr:.1f},{hcy:.1f} Z" '
            f'fill="{hair_color}" stroke="#222" stroke-width="1.0" />'
        )
        parts.append(
            f'<ellipse cx="{hcx - rr * 0.55:.1f}" cy="{hcy + rr * 0.25:.1f}" '
            f'rx="{rr * 0.30:.1f}" ry="{rr * 0.45:.1f}" '
            f'fill="{hair_color}" stroke="#222" stroke-width="0.6" />'
        )
        parts.append(
            f'<circle cx="{hcx + rr * 0.30:.1f}" cy="{eye_y:.1f}" r="1.4" fill="#222" />'
        )
    elif facing == "west":
        # Mirror of east.
        parts.append(
            f'<path d="M{hcx - rr:.1f},{hcy:.1f} '
            f'A{rr},{rr} 0 0 1 {hcx + rr:.1f},{hcy:.1f} Z" '
            f'fill="{hair_color}" stroke="#222" stroke-width="1.0" />'
        )
        parts.append(
            f'<ellipse cx="{hcx + rr * 0.55:.1f}" cy="{hcy + rr * 0.25:.1f}" '
            f'rx="{rr * 0.30:.1f}" ry="{rr * 0.45:.1f}" '
            f'fill="{hair_color}" stroke="#222" stroke-width="0.6" />'
        )
        parts.append(
            f'<circle cx="{hcx - rr * 0.30:.1f}" cy="{eye_y:.1f}" r="1.4" fill="#222" />'
        )
    else:  # south (face viewer)
        parts.append(
            f'<path d="M{hcx - rr:.1f},{hcy:.1f} '
            f'A{rr},{rr} 0 0 1 {hcx + rr:.1f},{hcy:.1f} Z" '
            f'fill="{hair_color}" stroke="#222" stroke-width="1.0" />'
        )
        eye_dx = rr * 0.42
        parts.append(
            f'<circle cx="{hcx - eye_dx:.1f}" cy="{eye_y:.1f}" r="1.4" fill="#222" />'
            f'<circle cx="{hcx + eye_dx:.1f}" cy="{eye_y:.1f}" r="1.4" fill="#222" />'
        )
    return "\n".join(parts)


# ---------- Scene assembly from latent z ----------
LATENT_DIM = 32  # 4 people × 7 dims + dolphin (2) + turtle (2)

# Fixed environment ---------------------------------------------------------
WATER_COLOR = "#5d8ac9"
GRASS_COLOR = "#7fb04a"
SAND_COLOR  = "#dcc480"
HOUSE_HALF   = 1.0   # x half-width
HOUSE_HALF_Z = 0.65  # z half-depth (shorter than width; houses are wider than deep)

# Island ellipse on the ground plane (y=0). Sized so the visible canvas shows
# the whole island with a margin of water on every side.
ISLAND_A = 5.2  # x semi-axis
ISLAND_B = 2.6  # z semi-axis

HOUSES = [
    {"x": -3.0, "z": 0.1, "wall": "#7ec9b8", "roof": "#d4a070", "door": "#5d3f2a"},  # mint walls / sandy thatch roof
    {"x":  0.0, "z": 1.8, "wall": "#f5d290", "roof": "#b87852", "door": "#3a2818"},  # cream walls / terracotta roof
    {"x":  3.0, "z": 0.1, "wall": "#f4a892", "roof": "#9c6841", "door": "#5d3a25"},  # coral walls / warm-brown roof
]

# Trees all live BEHIND or BESIDE the houses (z >= 0.6). This guarantees trees
# project to a region of the screen that never overlaps the foreground people
# lanes vertically, so people can never be occluded by a tree.
TREES = [
    {"x": -4.3, "z":  1.0, "size": 1.70, "leaf": "#3f8a3a"},  # back-left palm
    {"x": -1.5, "z":  1.0, "size": 1.30, "leaf": "#4ea14a"},  # between left house and back house
    {"x":  1.5, "z":  1.0, "size": 1.30, "leaf": "#3d8e3d"},  # between back house and right house
    {"x":  4.3, "z":  1.0, "size": 1.70, "leaf": "#3f8a3a"},  # back-right palm
]

# Per-person spawn rectangles. Lanes 0 (red) and 3 (gold) stay in the front
# strip; they can't walk back without overlapping the side houses. Lanes 1
# (blue) and 2 (green) live in the gap between left and right houses, so they
# can roam much further back, all the way up toward the yellow back-house
# (whose south wall sits at z=1.15).
PERSON_LANES = [
    {"x_min": -3.5, "x_max": -2.0, "z_min": -1.4, "z_max": -0.7},
    {"x_min": -1.5, "x_max":  0.0, "z_min": -1.4, "z_max":  0.85},
    {"x_min":  0.0, "x_max":  1.5, "z_min": -1.4, "z_max":  0.85},
    {"x_min":  2.0, "x_max":  3.5, "z_min": -1.4, "z_max": -0.7},
]

# One distinct shirt color per lane index; keeps each "person" identifiable
# across samples (the red-shirt person is always lane 0, etc.).
SHIRT_COLORS = ["#d24a3a", "#3a78c9", "#3aa84d", "#e0a32a"]


def render_island(cx=0.0, cz=0.0, a=ISLAND_A, b=ISLAND_B, n_points=80):
    """Organic island shape on the ground plane (y=0). A sandy ring rendered
    just outside a slightly-perturbed grass ellipse, both projected as polygons."""
    parts = []
    grass_pts = []
    sand_pts = []
    for i in range(n_points):
        t = 2 * np.pi * i / n_points
        # Subtle harmonic perturbation so the shoreline doesn't read as a clean ellipse.
        mod = 1.0 + 0.04 * np.sin(3 * t + 0.7) + 0.025 * np.cos(5 * t + 1.9)
        gx = cx + a * np.cos(t) * mod
        gz = cz + b * np.sin(t) * mod
        sxg, syg = project(gx, 0, gz)
        grass_pts.append((sxg, syg))
        # Sand ring is the same shape, scaled outward ~6%.
        sxs, sys = project(cx + a * np.cos(t) * mod * 1.06,
                           0,
                           cz + b * np.sin(t) * mod * 1.06)
        sand_pts.append((sxs, sys))

    sand_str  = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in sand_pts)
    grass_str = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in grass_pts)
    parts.append(
        f'<polygon points="{sand_str}" fill="{SAND_COLOR}" '
        f'stroke="{shade(SAND_COLOR, 0.78)}" stroke-width="1" stroke-linejoin="round" />'
    )
    parts.append(
        f'<polygon points="{grass_str}" fill="{GRASS_COLOR}" '
        f'stroke="{shade(GRASS_COLOR, 0.82)}" stroke-width="1" stroke-linejoin="round" />'
    )
    return "\n".join(parts)


def render_wave_marks(n=32, seed=13):
    """Decorative white tilde-marks scattered in the water region, around but
    not on the island. Fixed seed so the pattern stays constant across samples."""
    rng = np.random.default_rng(seed)
    parts = []
    placed = 0
    attempts = 0
    while placed < n and attempts < 400:
        attempts += 1
        x  = -7.5 + rng.uniform() * 15.0
        zc = -4.2 + rng.uniform() * 7.4
        # Skip if inside or too near the island (sand ring + a margin).
        if (x / ISLAND_A) ** 2 + (zc / ISLAND_B) ** 2 < 1.22:
            continue
        cx, cy = project(x, 0, zc)
        # Skip if off-canvas.
        if not (10 < cx < CANVAS_W - 10 and 8 < cy < CANVAS_H - 8):
            continue
        size = 4.0 + rng.uniform() * 3.0
        # Single-bump curve (a little arc), suggesting a wavelet seen from above.
        parts.append(
            f'<path d="M{cx - size:.1f},{cy:.1f} q{size:.1f},{-size * 0.55:.1f} {size * 2:.1f},0" '
            f'stroke="#eaf3fb" stroke-width="1.3" fill="none" '
            f'stroke-opacity="0.6" stroke-linecap="round" />'
        )
        placed += 1
    return "\n".join(parts)


def _splash_line(cx, cy, half_span):
    """Wavy white line at y=cy spanning ±half_span; suggests water surface
    around a partially-submerged animal."""
    n_segs = 4
    seg_w = (half_span * 2) / n_segs
    half = seg_w / 2
    d = f"M{cx - half_span:.1f},{cy:.1f}"
    sign = -1
    for _ in range(n_segs):
        d += f" q{half:.1f},{sign * 1.6:.1f} {seg_w:.1f},0"
        sign = -sign
    return (f'<path d="{d}" stroke="#eaf3fb" stroke-width="1.2" fill="none" '
            f'stroke-opacity="0.85" stroke-linecap="round" />')


def render_dolphin(x, z, clip_id="clip_dolphin"):
    """Stylized dolphin sprite seen from above, facing right. Drawn HALF-SUBMERGED:
    the lower half of the body (below y=cy in screen) is clipped away, and a
    small splash mark is drawn at the water line."""
    cx, cy = project(x, 0.05, z)
    parts = []
    # Clip-path: keep only what's above the water line at y=cy.
    parts.append(
        f'<defs><clipPath id="{clip_id}">'
        f'<rect x="0" y="0" width="{CANVAS_W}" height="{cy:.1f}" />'
        f'</clipPath></defs>'
    )
    parts.append(f'<g clip-path="url(#{clip_id})">')
    # Body: sleek elongated ellipse
    parts.append(
        f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="22" ry="7.5" '
        f'fill="#5a7a9d" stroke="#243a55" stroke-width="1.1" />'
    )
    # Tail flukes (left side)
    parts.append(
        f'<polygon points="'
        f'{cx - 20:.1f},{cy:.1f} '
        f'{cx - 30:.1f},{cy - 7:.1f} '
        f'{cx - 26:.1f},{cy:.1f} '
        f'{cx - 30:.1f},{cy + 7:.1f}'
        f'" fill="#5a7a9d" stroke="#243a55" stroke-width="1.0" />'
    )
    # Dorsal fin (top, slightly forward of center)
    parts.append(
        f'<polygon points="'
        f'{cx - 2:.1f},{cy - 6:.1f} '
        f'{cx + 4:.1f},{cy - 13:.1f} '
        f'{cx + 8:.1f},{cy - 6:.1f}'
        f'" fill="#3d5a7a" stroke="#243a55" stroke-width="1.0" />'
    )
    # Eye
    parts.append(f'<circle cx="{cx + 13:.1f}" cy="{cy - 2.5:.1f}" r="1.4" fill="#222" />')
    parts.append('</g>')
    # Splash line at the water cut
    parts.append(_splash_line(cx, cy, half_span=28))
    return "\n".join(parts)


def render_turtle(x, z, clip_id="clip_turtle"):
    """Stylized sea-turtle sprite seen from above, facing right. HALF-SUBMERGED:
    only the part above y=cy is drawn; a splash line marks the water surface."""
    cx, cy = project(x, 0.05, z)
    parts = []
    parts.append(
        f'<defs><clipPath id="{clip_id}">'
        f'<rect x="0" y="0" width="{CANVAS_W}" height="{cy:.1f}" />'
        f'</clipPath></defs>'
    )
    parts.append(f'<g clip-path="url(#{clip_id})">')
    # Front flippers (drawn first so the shell sits over them)
    for fx, fy in [(cx - 9, cy - 4), (cx + 9, cy - 4)]:
        parts.append(
            f'<ellipse cx="{fx:.1f}" cy="{fy:.1f}" rx="5" ry="2.6" '
            f'fill="#6e9a4e" stroke="#2c4a20" stroke-width="0.9" />'
        )
    # Shell
    parts.append(
        f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="13" ry="9" '
        f'fill="#5a8a4a" stroke="#2c4a20" stroke-width="1.1" />'
    )
    # Shell pattern (top half only is visible after clip)
    parts.append(
        f'<path d="M{cx - 7:.1f},{cy - 1:.1f} Q{cx:.1f},{cy - 5:.1f} {cx + 7:.1f},{cy - 1:.1f}" '
        f'stroke="#3a6a2a" stroke-width="1" fill="none" />'
    )
    parts.append(
        f'<line x1="{cx:.1f}" y1="{cy - 8:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
        f'stroke="#3a6a2a" stroke-width="0.9" />'
    )
    # Head (popping out the right)
    parts.append(
        f'<ellipse cx="{cx + 14:.1f}" cy="{cy - 2:.1f}" rx="4.5" ry="3.5" '
        f'fill="#7aa55a" stroke="#2c4a20" stroke-width="1.0" />'
    )
    # Eye
    parts.append(f'<circle cx="{cx + 16:.1f}" cy="{cy - 3:.1f}" r="1.1" fill="#222" />')
    parts.append('</g>')
    # Splash line at the water cut
    parts.append(_splash_line(cx, cy, half_span=18))
    return "\n".join(parts)


# Animal lanes: both occupy the front-water strip (z south of the island)
# Dolphin gets the LEFT half, turtle gets the RIGHT half so they don't overlap.
# DOLPHIN_LANE = {"x_min": -3.5, "x_max": -0.5, "z_min": -3.7, "z_max": -3.0}
# TURTLE_LANE  = {"x_min":  0.5, "x_max":  3.5, "z_min": -3.7, "z_max": -3.0}
DOLPHIN_LANE = {"x_min": -3.5, "x_max": -0.5, "z_min": -5.5, "z_max": -3.0}
TURTLE_LANE  = {"x_min":  0.5, "x_max":  2.5, "z_min": -5.5, "z_max": -3.0}

def on_island(x, zc, margin=0.92):
    """True if (x, 0, zc) is comfortably inside the island grass region."""
    return (x / ISLAND_A) ** 2 + (zc / ISLAND_B) ** 2 < margin ** 2


def push_out_of_houses(x, zc, buffer=0.25):
    """Kept as a utility but no longer called; lanes are guaranteed clear."""
    for h in HOUSES:
        x_lo = h["x"] - HOUSE_HALF - buffer
        x_hi = h["x"] + HOUSE_HALF + buffer
        z_lo = h["z"] - HOUSE_HALF_Z - buffer
        z_hi = h["z"] + HOUSE_HALF_Z + buffer
        if x_lo <= x <= x_hi and z_lo <= zc <= z_hi:
            d_left  = x - x_lo
            d_right = x_hi - x
            d_south = zc - z_lo
            d_north = z_hi - zc
            md = min(d_left, d_right, d_south, d_north)
            if md == d_left:    x = x_lo
            elif md == d_right: x = x_hi
            elif md == d_south: zc = z_lo
            else:               zc = z_hi
    return x, zc


def generate_scene_svg(z):
    z = np.asarray(z, dtype=float).clip(0, 1)
    assert z.shape[0] >= LATENT_DIM, f"need at least {LATENT_DIM} latents"

    parts = []

    # --- Water background ---
    parts.append(f'<rect width="100%" height="100%" fill="{WATER_COLOR}" />')

    # --- Decorative wave marks on the water (fixed seed, stays constant) ---
    parts.append(render_wave_marks())

    # --- Island (sand ring + grass) ---
    parts.append(render_island())

    # --- Decorative grass tufts on the island, fixed seed ---
    rng_dec = np.random.default_rng(42)
    tuft_color = shade(GRASS_COLOR, 0.65)
    placed = 0
    attempts = 0
    while placed < 22 and attempts < 200:
        attempts += 1
        gx = -ISLAND_A * 0.95 + rng_dec.uniform() * (ISLAND_A * 1.9)
        gz = -ISLAND_B * 0.85 + rng_dec.uniform() * (ISLAND_B * 1.7)
        if not on_island(gx, gz, margin=0.90):
            continue
        if any(abs(gx - h["x"]) < HOUSE_HALF + 0.1 and abs(gz - h["z"]) < HOUSE_HALF_Z + 0.1
               for h in HOUSES):
            continue
        parts.append(render_grass_tuft(gx, gz, tuft_color))
        placed += 1

    # --- Flowers on the island, fixed seed ---
    flower_palette = ["#e74c3c", "#f1c40f", "#9b59b6", "#ecf0f1", "#ff8a65"]
    rng_fl = np.random.default_rng(7)
    placed = 0
    attempts = 0
    while placed < 12 and attempts < 200:
        attempts += 1
        fx = -ISLAND_A * 0.9 + rng_fl.uniform() * (ISLAND_A * 1.8)
        fz = -ISLAND_B * 0.85 + rng_fl.uniform() * (ISLAND_B * 1.7)
        if not on_island(fx, fz, margin=0.88):
            continue
        if any(abs(fx - h["x"]) < HOUSE_HALF + 0.1 and abs(fz - h["z"]) < HOUSE_HALF_Z + 0.1
               for h in HOUSES):
            continue
        parts.append(render_flower(fx, fz, flower_palette[placed % len(flower_palette)]))
        placed += 1

    # --- People plan from latents ---
    facings = ["south", "east", "north", "west"]
    people_data = []
    for i in range(4):
        b = i * 7
        lane = PERSON_LANES[i]
        px = lane["x_min"] + z[b]     * (lane["x_max"] - lane["x_min"])
        pz = lane["z_min"] + z[b + 1] * (lane["z_max"] - lane["z_min"])
        facing_idx = int(z[b + 2] * 3.999)
        skin = hsv_hex(0.05 + z[b + 3] * 0.06, 0.40, 0.85)
        # Shirt color is FIXED per lane; lets us identify each person across
        # samples. Shirt latents (z[b+4..6]) are unused for now.
        shirt = SHIRT_COLORS[i]
        people_data.append({
            "x": px, "z": pz, "facing": facings[facing_idx],
            "skin": skin, "shirt": shirt, "hair": "#3e2c1a",
        })

    # --- Animals from latents (positions only) ---
    dolphin_x = DOLPHIN_LANE["x_min"] + z[28] * (DOLPHIN_LANE["x_max"] - DOLPHIN_LANE["x_min"])
    dolphin_z = DOLPHIN_LANE["z_min"] + z[29] * (DOLPHIN_LANE["z_max"] - DOLPHIN_LANE["z_min"])
    turtle_x  = TURTLE_LANE["x_min"]  + z[30] * (TURTLE_LANE["x_max"]  - TURTLE_LANE["x_min"])
    turtle_z  = TURTLE_LANE["z_min"]  + z[31] * (TURTLE_LANE["z_max"]  - TURTLE_LANE["z_min"])

    # --- 3D objects, painter's algorithm by south-front edge of footprint ---
    objs = []
    for h in HOUSES:
        objs.append((h["z"] - HOUSE_HALF_Z,
                     render_house(h["x"], h["z"],
                                  wx=HOUSE_HALF, wz=HOUSE_HALF_Z,
                                  wall=h["wall"], roof=h["roof"], door=h["door"])))
    for t in TREES:
        objs.append((t["z"], render_tree(t["x"], t["z"], height=t["size"],
                                         leaf_color=t["leaf"],
                                         leaf_radius=t["size"] * 0.38)))
    for p in people_data:
        objs.append((p["z"], render_person(p["x"], p["z"],
                                           skin_color=p["skin"],
                                           shirt_color=p["shirt"],
                                           hair_color=p["hair"],
                                           facing=p["facing"])))
    objs.append((dolphin_z, render_dolphin(dolphin_x, dolphin_z)))
    objs.append((turtle_z,  render_turtle(turtle_x,  turtle_z)))

    # Sort by depth: largest depth-key (furthest north) drawn first.
    objs.sort(key=lambda kv: -kv[0])
    for _, svg in objs:
        parts.append(svg)

    body = "\n".join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" '
            f'height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">\n'
            f'{body}\n</svg>')


# ---------- Render a small grid of examples ----------
def svg_to_png_array(svg_str, w=CANVAS_W, h=CANVAS_H):
    png = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"),
                           output_width=w, output_height=h)
    return np.array(Image.open(io.BytesIO(png)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_rows, n_cols = 3, 4
    n = n_rows * n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.0))
    for i, ax in enumerate(axes.flat):
        z = rng.uniform(0, 1, LATENT_DIM)
        svg_str = generate_scene_svg(z)
        ax.imshow(svg_to_png_array(svg_str))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"sample {i}", fontsize=9)

    plt.tight_layout()
    out = "svg_island_samples.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")

    # Also save one large SVG for inspection
    z = rng.uniform(0, 1, LATENT_DIM)
    with open("svg_island_sample.svg", "w") as f:
        f.write(generate_scene_svg(z))
    print("Saved: svg_island_sample.svg")
