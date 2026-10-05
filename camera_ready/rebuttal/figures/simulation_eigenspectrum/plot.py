"""The first 20 eigenvalues of the feature covariance, on a linear scale.

On a log-log plot the spectrum collapsing after k=2 can look like a bug. On a linear
scale it is the paper's positive result: a trained network whose features are an
affine image of a two-dimensional latent has a rank-2 feature covariance, so
everything past the second eigenvalue is floating-point noise.
"""
import json
import sys
from pathlib import Path

import plotly.graph_objects as go

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _theme import PLOTLY_CONFIG, apply_theme  # noqa: E402

rows = json.loads((HERE / "data.json").read_text())
colors = json.loads((HERE.parent / "entity_colors.json").read_text())
LABEL = {"trained_wd1": "Trained, weight decay 1",
         "trained_wd0": "Trained, weight decay 0",
         "random_init": "Random init"}

fig = go.Figure()
for cond in ["trained_wd1", "trained_wd0", "random_init"]:
    sel = sorted([r for r in rows if r["condition"] == cond],
                 key=lambda r: r["pc_index"])
    if not sel:
        continue
    fig.add_trace(go.Scatter(
        x=[r["pc_index"] for r in sel],
        y=[r["eigenvalue_share_mean"] for r in sel],
        name=LABEL[cond], mode="lines+markers",
        line=dict(color=colors[cond]), marker=dict(color=colors[cond], size=7),
        error_y=dict(type="data", array=[r.get("eigenvalue_share_sd", 0) for r in sel],
                     color=colors[cond], thickness=1, width=3),
        hovertemplate=(f"{LABEL[cond]}<br>PC %{{x}}<br>share %{{y:.4g}}"
                       "<extra></extra>"),
    ))
fig.add_vline(x=2.5, line=dict(color="#B4B4B4", dash="dot", width=1),
              annotation_text="latent dimension = 2", annotation_position="top right",
              annotation_font=dict(size=11, color="#7B7B7B"))
fig.update_xaxes(title_text="Principal component index", dtick=2)
fig.update_yaxes(title_text="Share of total feature variance")
apply_theme(fig, height=440)
fig.write_image(str(HERE / "simulation_eigenspectrum.svg"))
(HERE / "simulation_eigenspectrum.html").write_text(
    fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG))
print("wrote", HERE / "simulation_eigenspectrum.svg")
