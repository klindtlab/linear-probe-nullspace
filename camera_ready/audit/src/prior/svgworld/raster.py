"""SVG rasterization backend for SVG-World.

The submitted pipeline used ``cairosvg``, which needs the system library
``libcairo.so.2``. The job container does not ship it, so this module picks a
backend at import time and records which one was used, so the choice is part of
the run provenance rather than an invisible detail.

Backends, in preference order:
  1. ``cairosvg``      identical to the submitted pipeline (needs libcairo)
  2. ``resvg_py``      self-contained Rust rasterizer, no system dependency

Both are driven the same way: the SVG's outer width/height are set to the
target pixel size while the viewBox is left alone, so the rasterizer does the
scaling and no aspect distortion is introduced.
"""

from __future__ import annotations

import io
import re

import numpy as np

BACKEND: str | None = None
_BACKEND_VERSION: str | None = None

try:  # pragma: no cover - environment dependent
    import cairosvg as _cairosvg

    # cairocffi only dlopens libcairo on first use, so force it here.
    _cairosvg.svg2png(
        bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4" '
        b'viewBox="0 0 4 4"><rect width="4" height="4" fill="#123456"/></svg>',
        output_width=4,
        output_height=4,
    )
    BACKEND = "cairosvg"
    _BACKEND_VERSION = getattr(_cairosvg, "__version__", "unknown")
except Exception:  # pragma: no cover
    _cairosvg = None

if BACKEND is None:
    try:  # pragma: no cover - environment dependent
        import resvg_py as _resvg

        BACKEND = "resvg_py"
        _BACKEND_VERSION = getattr(_resvg, "__version__", "unknown")
    except Exception:  # pragma: no cover
        _resvg = None
else:
    _resvg = None

_W_ATTR = re.compile(r'(<svg[^>]*?)\swidth="[^"]*"\s+height="[^"]*"')


def backend_info() -> dict:
    return {"backend": BACKEND, "version": _BACKEND_VERSION}


def _set_outer_size(svg_str: str, size: int) -> str:
    """Set the outer width/height to ``size`` px, leaving the viewBox alone."""
    new, n = _W_ATTR.subn(rf'\1 width="{size}" height="{size}"', svg_str, count=1)
    if n == 0:
        new = svg_str.replace("<svg", f'<svg width="{size}" height="{size}"', 1)
    return new


def svg_to_png_bytes(svg_str: str, size: int = 224) -> bytes:
    """Rasterize an SVG string to a ``size`` x ``size`` PNG."""
    if BACKEND is None:
        raise RuntimeError(
            "no SVG rasterization backend available (tried cairosvg, resvg_py)"
        )
    sized = _set_outer_size(svg_str, size)
    if BACKEND == "cairosvg":
        return _cairosvg.svg2png(
            bytestring=sized.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
    out = _resvg.svg_to_bytes(svg_string=sized)
    return bytes(out) if not isinstance(out, bytes) else out


def svg_to_rgb_array(svg_str: str, size: int = 224) -> np.ndarray:
    """Rasterize to an (size, size, 3) uint8 RGB array on a white background."""
    from PIL import Image

    png = svg_to_png_bytes(svg_str, size=size)
    img = Image.open(io.BytesIO(png))
    if img.size != (size, size):
        img = img.convert("RGBA").resize((size, size), Image.LANCZOS)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.asarray(img.convert("RGB"))
