"""SVG-World: render both styles from one shared latent matrix, then extract
pooled features for every (encoder, variant) condition.

Row i of the AquaWorld (island) set and row i of the WestWorld (western) set are
rendered from the SAME latent vector. That is what makes the cross-style
transfer diagnostic (D3) well defined, and it is why splits must be taken over
scene ids so paired styles never straddle the train/test boundary.

Feature extraction for every encoder lives in `encoders.py`; this module only
produces the images and the shared latent matrix.
"""

from __future__ import annotations

import io
import os
import time

import numpy as np

import rebuttal_config as C
from svgworld import raster, svg_island, svg_western

STYLE_MODULES = {"island": svg_island, "western": svg_western}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def sample_latents(n: int, seed: int = C.DATA_SEED) -> np.ndarray:
    """The submitted latent sampler: Uniform[0,1] over all 32 stored dims."""
    assert svg_island.LATENT_DIM == svg_western.LATENT_DIM == 32
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (n, svg_island.LATENT_DIM)).astype(np.float32)


def square_pad_svg(svg_str: str, vb_w: int = 600, vb_h: int = 420) -> str:
    """Extend a 600x420 viewBox to a square 600x600 by padding top and bottom.

    Identical to the submitted `1_render_dataset.py`, so the framing of every
    scene matches the submission.
    """
    pad = (vb_w - vb_h) // 2
    return svg_str.replace(f'viewBox="0 0 {vb_w} {vb_h}"',
                           f'viewBox="0 {-pad} {vb_w} {vb_w}"')


def render_style(Z: np.ndarray, style: str, size: int = C.IMAGE_SIZE,
                 png_dir: str | None = None, n_png_keep: int = 0,
                 verbose_every: int = 1000) -> np.ndarray:
    """Render every latent row in one style. Returns uint8 (N, size, size, 3)."""
    mod = STYLE_MODULES[style]
    n = Z.shape[0]
    out = np.empty((n, size, size, 3), dtype=np.uint8)
    if png_dir and n_png_keep:
        os.makedirs(png_dir, exist_ok=True)
    t0 = time.time()
    for i in range(n):
        svg = square_pad_svg(mod.generate_scene_svg(Z[i]))
        if png_dir and i < n_png_keep:
            png = raster.svg_to_png_bytes(svg, size=size)
            with open(os.path.join(png_dir, f"{style}_{i:05d}.png"), "wb") as f:
                f.write(png)
            from PIL import Image
            img = Image.open(io.BytesIO(png)).convert("RGB")
            out[i] = np.asarray(img)
        else:
            out[i] = raster.svg_to_rgb_array(svg, size=size)
        if verbose_every and (i + 1) % verbose_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[render {style}] {i + 1}/{n} ({rate:.1f} scenes/s)", flush=True)
    print(f"[render {style}] done {n} scenes in {time.time() - t0:.1f}s "
          f"backend={raster.BACKEND}", flush=True)
    return out


def render_report(images: dict[str, np.ndarray]) -> dict:
    """Integrity facts about the rendered set, checkpointed before any probing."""
    rep = {}
    for style, arr in images.items():
        flat = arr.reshape(arr.shape[0], -1)
        per_image_mean = flat.mean(axis=1)
        rep[style] = dict(
            n=int(arr.shape[0]),
            shape=list(arr.shape[1:]),
            pixel_mean=float(arr.mean()),
            pixel_std=float(arr.std()),
            n_constant_images=int((flat.std(axis=1) < 1e-6).sum()),
            n_duplicate_images=_n_duplicate_rows(flat),
            per_image_mean_std=float(per_image_mean.std()),
        )
    if len(images) == 2:
        a, b = (images[s].reshape(images[s].shape[0], -1).astype(np.int16)
                for s in ("island", "western"))
        rep["cross_style_mean_abs_pixel_diff"] = float(np.abs(a - b).mean())
    return rep

def _n_duplicate_rows(flat: np.ndarray) -> int:
    """Byte-exact duplicate count. Hashing the FULL row matters: subsampling
    every 97th pixel reported 29 false duplicates out of 96 dSprites images,
    whose sprites are sparse and share most background pixels."""
    seen = {hash(r.tobytes()) for r in np.ascontiguousarray(flat)}
    return int(flat.shape[0] - len(seen))
