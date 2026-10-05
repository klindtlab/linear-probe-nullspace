"""Is the trained Section 4 network affine on the data?

An affine network has exactly one linear region over the dataset and a constant
input-output Jacobian. This figure plots both against hidden width for the two
trained regimes and random initialization, with the degeneracy threshold marked.
"""
import json
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _theme import PLOTLY_CONFIG, apply_theme  # noqa: E402

rows = json.loads((HERE / "data.json").read_text())
colors = json.loads((HERE.parent / "entity_colors.json").read_text())
LABEL = {"trained_wd1": "Trained, weight decay 1",
         "trained_wd0": "Trained, weight decay 0",
         "random_init": "Random init"}

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.11,
                    subplot_titles=("Distinct ReLU regions over the held-out set",
                                    "Jacobian variation (coefficient of variation)"))
for i, cond in enumerate(["trained_wd1", "trained_wd0", "random_init"]):
    sel = sorted([r for r in rows if r["condition"] == cond], key=lambda r: r["width"])
    if not sel:
        continue
    x = [r["width"] for r in sel]
    for row_i, key in enumerate(["n_regions_mean", "jacobian_cv_mean"], start=1):
        lo = [r[key] - (r[key] - r[key.replace("_mean", "_min")]) for r in sel]
        hi = [r[key.replace("_mean", "_max")] for r in sel]
        fig.add_trace(go.Scatter(
            x=x, y=[r[key] for r in sel], name=LABEL[cond], mode="lines+markers",
            line=dict(color=colors[cond]), marker=dict(color=colors[cond], size=7),
            showlegend=(row_i == 1),
            error_y=dict(type="data", symmetric=False,
                         array=[h - r[key] for h, r in zip(hi, sel)],
                         arrayminus=[r[key] - l for l, r in zip(lo, sel)],
                         color=colors[cond], thickness=1, width=3),
            hovertemplate=(f"{LABEL[cond]}<br>width %{{x}}<br>%{{y:.4g}}"
                           "<extra></extra>"),
        ), row=row_i, col=1)

# One region, or zero Jacobian variation, is what "affine on the data" looks like.
fig.add_hline(y=1, row=1, col=1, line=dict(color="#B9605B", dash="dash", width=1),
              annotation_text="1 region = affine on the data",
              annotation_position="bottom right",
              annotation_font=dict(size=11, color="#B9605B"))
fig.update_yaxes(title_text="Distinct regions (of 2000)", type="log", row=1, col=1)
fig.update_yaxes(title_text="Jacobian CV", row=2, col=1)
fig.update_xaxes(title_text="Hidden width h", type="log", row=2, col=1)
apply_theme(fig, height=620)
fig.write_image(str(HERE / "simulation_integrity.svg"))
(HERE / "simulation_integrity.html").write_text(
    fig.to_html(full_html=False, include_plotlyjs="cdn", config=PLOTLY_CONFIG))
print("wrote", HERE / "simulation_integrity.svg")
