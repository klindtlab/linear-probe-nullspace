"""Per-factor recipe attribution: horizontal bars of the mean change in
reverse predictivity per recipe factor, with min/max across 8 conditions
as asymmetric range whiskers. Loads co-located data.json; writes
recipe_attribution.svg."""
import json
from pathlib import Path

import plotly.graph_objects as go

HERE = Path(__file__).parent
data = json.loads((HERE / "data.json").read_text())

SLATE = "#4E728A"

LABELS = {
    "alpha_mode=cv": "ridge penalty by cross-validation (alpha_mode=cv)",
    "dtype=f64": "float64 arithmetic (dtype=f64)",
    "formula=sse": "sum-of-squares R\u00b2 formula (formula=sse)",
    "split=unit0": "split assignment (split=unit0)",
    "targets=position12": "target columns (targets=position12)",
}


def apply_theme(fig, *, height=400):
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
        uniformtext=dict(minsize=10, mode="hide"),
    )
    fig.update_traces(textposition="none", selector=dict(type="bar"))
    fig.update_xaxes(gridcolor="#B4B4B4", zerolinecolor="#B4B4B4", zerolinewidth=1,
                     automargin=True, ticks="outside",
                     tickfont=dict(size=12), title_font=dict(size=13),
                     autotickangles=[0, 30])
    fig.update_yaxes(gridcolor="#B4B4B4", zerolinecolor="#B4B4B4", zerolinewidth=1,
                     automargin=True, ticks="outside",
                     tickfont=dict(size=12), title_font=dict(size=13))
    for trace in fig.data:
        if hasattr(trace, "cliponaxis"):
            trace.cliponaxis = True
    return fig


def build():
    rows = sorted(data["per_factor"], key=lambda r: abs(r["mean"]))
    ys = [LABELS[r["factor"]] for r in rows]
    means = [r["mean"] for r in rows]
    err_plus = [r["max"] - r["mean"] for r in rows]
    err_minus = [r["mean"] - r["min"] for r in rows]
    fig = go.Figure(go.Bar(
        x=means, y=ys, orientation="h",
        marker=dict(color=SLATE),
        error_x=dict(type="data", array=err_plus, arrayminus=err_minus,
                     color="#1D272A", thickness=1.2, width=4),
        showlegend=False,
    ))
    fig.update_xaxes(
        title_text="Change in reverse predictivity vs the submitted recipe",
        tickformat=".4f",
    )
    fig.update_yaxes(title_text="Recipe factor")
    return fig


if __name__ == "__main__":
    fig = build()
    apply_theme(fig, height=380)
    fig.write_image(str(HERE / "recipe_attribution.svg"))
    print("wrote", HERE / "recipe_attribution.svg")
