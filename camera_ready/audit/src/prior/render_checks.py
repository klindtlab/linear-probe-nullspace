"""Render validation for the SVG rasterizer swap.

The submitted pipeline rasterized with cairosvg, which needs a system libcairo
the job image does not ship, so the run uses resvg. A pixel-identity comparison
against cairosvg is therefore impossible inside the job; what CAN be established
is that the replacement rasterizer reproduces the scene geometry the generators
encode. If it did not, every position-latent claim would be measuring a broken
renderer rather than an encoder.

Three checks, all on the rendered pixels alone:

  geometry     moving a position latent moves the corresponding object in the
               projected direction, monotonically
  determinism  the same latent renders to identical bytes on a repeat call
  sensitivity  scenes that differ in one latent differ in pixels, and the two
               styles differ from each other at the same latent

The projection is oblique (see svg_island.project): +lane_x moves an object
right on screen, and +lane_z moves it up and to the left.
"""

from __future__ import annotations

import numpy as np

import rebuttal_config as C
from svgworld import raster
from svgworld_data import STYLE_MODULES, square_pad_svg

# lane_x / lane_z of person 0 and of the two animals
PROBE_DIMS = {"person0_lane_x": 0, "person0_lane_z": 1,
              "animal0_lane_x": 28, "animal1_lane_x": 30}


def _render(z: np.ndarray, style: str, size: int = C.IMAGE_SIZE) -> np.ndarray:
    svg = square_pad_svg(STYLE_MODULES[style].generate_scene_svg(z))
    return raster.svg_to_rgb_array(svg, size=size)


def _spearman(vals) -> float:
    """Rank correlation of the sequence against its own index order."""
    v = np.asarray(vals, dtype=np.float64)
    r = np.argsort(np.argsort(v)).astype(np.float64)
    i = np.arange(len(v), dtype=np.float64)
    if v.std() == 0:
        return 0.0
    return float(np.corrcoef(r, i)[0, 1])


def _ink_centroid(img: np.ndarray, ref: np.ndarray):
    """Centroid of the pixels that changed relative to a reference render."""
    d = np.abs(img.astype(np.int16) - ref.astype(np.int16)).sum(axis=2)
    m = d > 24
    if m.sum() < 10:
        return None, int(m.sum())
    ys, xs = np.nonzero(m)
    w = d[m].astype(np.float64)
    return (float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())), int(m.sum())


def check_geometry(style: str, seed: int = 0,
                   levels=(0.05, 0.28, 0.50, 0.73, 0.95)) -> dict:
    """Sweep one latent at a time and test that the moved object tracks it.

    The reference render is the FIRST level, so every later level has a nonzero
    difference mask to take a centroid of. (Referencing the middle level makes
    that level's own mask empty, which is not a renderer failure.)
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.3, 0.7, 32).astype(np.float32)
    out = {}
    for name, dim in PROBE_DIMS.items():
        cxs, cys, npx = [], [], []
        renders = []
        for v in levels:
            z = base.copy()
            z[dim] = np.float32(v)
            renders.append(_render(z, style))
        ref = renders[0]
        for img in renders[1:]:
            c, n = _ink_centroid(img, ref)
            npx.append(n)
            if c is not None:
                cxs.append(c[0])
                cys.append(c[1])
        res = dict(n_changed_pixels=npx, n_levels=len(levels))
        if len(cxs) == len(levels) - 1:
            res.update(centroid_x=cxs, centroid_y=cys)
            if name.endswith("lane_x"):
                res["monotone_rightward"] = bool(
                    all(b > a for a, b in zip(cxs, cxs[1:])))
                res["x_span_px"] = float(cxs[-1] - cxs[0])
                res["spearman_x"] = _spearman(cxs)
            else:  # lane_z: up and to the left on screen
                res["monotone_upward"] = bool(
                    all(b < a for a, b in zip(cys, cys[1:])))
                res["y_span_px"] = float(cys[0] - cys[-1])
                res["spearman_y"] = _spearman([-v for v in cys])
        out[name] = res
    return out


def check_determinism(style: str, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, 1, 32).astype(np.float32)
    a, b = _render(z, style), _render(z, style)
    return dict(identical=bool(np.array_equal(a, b)),
                max_abs_diff=int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()))


def check_style_sensitivity(seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    z1 = rng.uniform(0, 1, 32).astype(np.float32)
    z2 = z1.copy()
    z2[0] = np.float32(1.0 - z1[0])
    isl1, wes1 = _render(z1, "island"), _render(z1, "western")
    isl2 = _render(z2, "island")
    def mad(a, b):
        return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
    return dict(mean_abs_diff_across_styles=mad(isl1, wes1),
                mean_abs_diff_within_style_latent_change=mad(isl1, isl2),
                styles_differ=bool(mad(isl1, wes1) > 1.0),
                latent_change_visible=bool(mad(isl1, isl2) > 0.05))


def run_all(styles=C.STYLES) -> dict:
    rep = dict(backend=raster.backend_info(),
               style_sensitivity=check_style_sensitivity())
    for s in styles:
        rep[s] = dict(geometry=check_geometry(s), determinism=check_determinism(s))
    ok = True
    for s in styles:
        g = rep[s]["geometry"]
        ok &= bool(rep[s]["determinism"]["identical"])
        ok &= bool(g["person0_lane_x"].get("monotone_rightward", False))
        ok &= bool(g["animal0_lane_x"].get("monotone_rightward", False))
        ok &= bool(g["person0_lane_z"].get("monotone_upward", False))
    rep["all_checks_pass"] = bool(ok and rep["style_sensitivity"]["styles_differ"]
                                 and rep["style_sensitivity"]["latent_change_visible"])
    return rep
