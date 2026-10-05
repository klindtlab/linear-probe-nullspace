"""Inversion mechanism on 3D Shapes: participation ratio of the feature
spectrum against reverse predictivity for the two transformer encoders,
pretrained vs random initialization (random points are means over 5
instances). Loads co-located data.json; writes inversion_mechanism.svg.
ResNet-50 rows in the data are excluded here: its untrained control is
construction-dependent and it is reported separately on the page."""
import json
from pathlib import Path

import plotly.graph_objects as go

HERE = Path(__file__).parent
data = json.loads((HERE / "data.json").read_text())

# entity colors (figures/entity_colors.json)
MODEL = {
    "clip_b16": ("CLIP-B/16", "#4E728A"),
    "dinov3_b16": ("DINOv3-B/16", "#2E6E4E"),
}


def apply_theme(fig, *, height=440):
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
                     tickfont=dict(size=12), title_font=dict(size=13))
    fig.update_yaxes(gridcolor="#B4B4B4", zerolinecolor="#B4B4B4", zerolinewidth=1,
                     automargin=True, ticks="outside",
                     tickfont=dict(size=12), title_font=dict(size=13))
    for trace in fig.data:
        if hasattr(trace, "cliponaxis"):
            trace.cliponaxis = True
    return fig


def build():
    fig = go.Figure()
    rows = [r for r in data["rows"]
            if r["model"] in MODEL and r["kind"] in ("pretrained", "random")]
    for r in rows:
        label, color = MODEL[r["model"]]
        pretrained = r["kind"] == "pretrained"
        fig.add_trace(go.Scatter(
            x=[r["participation_ratio"]], y=[r["d1"]],
            mode="markers+text",
            name=f"{label}, {'pretrained' if pretrained else 'random init (mean of 5)'}",
            marker=dict(
                size=14, color=color,
                symbol="circle" if pretrained else "circle-open",
                line=dict(width=2, color=color),
            ),
            text=[f"{label} {'pretrained' if pretrained else 'random'}"],
            textposition=("middle left" if r["model"] == "clip_b16" and pretrained
                          else "middle right" if pretrained else "top right"),
            textfont=dict(size=12, color=color),
            hovertemplate=(f"{label} {'pretrained' if pretrained else 'random'}"
                           "<br>participation ratio %{x:.2f}"
                           "<br>reverse predictivity %{y:.4f}"
                           f"<br>forward predictivity (linear decodability) {r['forward_r2']:.4f}"
                           "<extra></extra>"),
        ))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(
        title_text="Participation ratio of the feature spectrum (effective dimensionality)",
        range=[0, 19],
    )
    fig.update_yaxes(title_text="Reverse predictivity", range=[0, 1.0])
    return fig


if __name__ == "__main__":
    fig = build()
    apply_theme(fig, height=440)
    fig.write_image(str(HERE / "inversion_mechanism.svg"))
    print("wrote", HERE / "inversion_mechanism.svg")
