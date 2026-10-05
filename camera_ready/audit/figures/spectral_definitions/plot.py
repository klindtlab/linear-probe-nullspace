"""Spectral concentration vs cutoff k: the printed statistic S(k) and the
monotone alternative C(k) for three simulation conditions (five seeds
averaged, width 1024). Loads co-located data.json; writes
spectral_definitions.svg. Points where a statistic is undefined (NaN in the
data) are omitted."""
import json
import math
from pathlib import Path

import plotly.graph_objects as go

HERE = Path(__file__).parent
raw = (HERE / "data.json").read_text().replace("NaN", "null")
rows = json.loads(raw)

# entity colors (figures/entity_colors.json): condition -> color
COND = {
    "random": ("random init", "#C4650D"),
    "trained_wd0": ("trained, weight decay 0", "#4E728A"),
    "trained_wd1": ("trained, weight decay 1", "#2E6E4E"),
}
LATENT_RANK = 2


def apply_theme(fig, *, height=460):
    fig.update_layout(
        height=height,
        margin=dict(t=16, r=24, b=24, l=24, autoexpand=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1D272A",
                  family="'Suisse Intl', -apple-system, BlinkMacSystemFont, Arial, sans-serif",
                  size=13),
        title=None,
        legend=dict(orientation="h", xref="container", x=0, xanchor="left",
                    yref="container", y=0, yanchor="bottom",
                    title=dict(side="top"), bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#1D272A", size=12)),
        modebar=dict(orientation="h", bgcolor="rgba(0,0,0,0)",
                     color="#1D272A", activecolor="#C4650D",
                     remove=["lasso2d", "select2d", "autoScale2d"]),
    )
    fig.update_xaxes(gridcolor="#B4B4B4", zerolinecolor="#B4B4B4", zerolinewidth=1,
                     automargin=True, ticks="outside",
                     tickfont=dict(size=12), title_font=dict(size=13),
                     minor=dict(ticks="outside", ticklen=3,
                                tickcolor="#B4B4B4", showgrid=False))
    fig.update_yaxes(gridcolor="#B4B4B4", zerolinecolor="#B4B4B4", zerolinewidth=1,
                     automargin=True, ticks="outside",
                     tickfont=dict(size=12), title_font=dict(size=13))
    for trace in fig.data:
        if hasattr(trace, "cliponaxis"):
            trace.cliponaxis = True
    return fig


def series(cond, key):
    pts = sorted((r for r in rows if r["condition"] == cond), key=lambda r: r["k"])
    pts = [r for r in pts if r[key] is not None and not (
        isinstance(r[key], float) and math.isnan(r[key]))]
    return [r["k"] for r in pts], [r[key] for r in pts]


def build():
    fig = go.Figure()
    for cond, (label, color) in COND.items():
        ks, s = series(cond, "S")
        fig.add_trace(go.Scatter(
            x=ks, y=s, mode="lines+markers",
            name=f"{label}, printed statistic S(k)",
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color, symbol="circle"),
        ))
    for cond, (label, color) in COND.items():
        ks, c = series(cond, "C")
        fig.add_trace(go.Scatter(
            x=ks, y=c, mode="lines+markers",
            name=f"{label}, monotone alternative C(k)",
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(size=6, color=color, symbol="circle-open"),
        ))
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=LATENT_RANK, x1=LATENT_RANK, y0=0, y1=1,
                  line=dict(width=1, dash="dot", color="#7B7B7B"))
    fig.add_annotation(x=math.log10(LATENT_RANK), y=1.04, xref="x", yref="paper",
                       text="latent rank", showarrow=False,
                       font=dict(size=12, color="#7B7B7B"))
    fig.update_xaxes(
        type="log",
        title_text="Cutoff k, top principal components retained (log)",
    )
    fig.update_yaxes(title_text="Spectral concentration", range=[-0.04, 1.08])
    return fig


if __name__ == "__main__":
    fig = build()
    apply_theme(fig, height=480)
    fig.write_image(str(HERE / "spectral_definitions.svg"))
    print("wrote", HERE / "spectral_definitions.svg")
