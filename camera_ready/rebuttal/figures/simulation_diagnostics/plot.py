"""The four diagnostics across hidden width, with one D2 definition.

Replaces the submitted Figure 3, whose spectral-concentration panel computed a
different statistic from the one the text defined and from the one the SVG-World
figure used. All four panels here come from the definitions in src/diagnostics.py.
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
PANELS = [("fwd_r2", "Forward R²"), ("rev_r2", "Reverse R² (D1)"),
          ("d2_whitened_at_klat", "Spectral concentration S(k_lat) (D2)"),
          ("ood_r2", "Out-of-distribution R² (D3)")]

fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                    subplot_titles=[t for _, t in PANELS])
for pi, (key, title) in enumerate(PANELS, start=1):
    for cond in ["trained_wd1", "trained_wd0", "random_init"]:
        sel = sorted([r for r in rows if r["condition"] == cond],
                     key=lambda r: r["width"])
        if not sel:
            continue
        mk, sk = f"{key}_mean", f"{key}_sd"
        fig.add_trace(go.Scatter(
            x=[r["width"] for r in sel], y=[r[mk] for r in sel],
            name=LABEL[cond], mode="lines+markers", showlegend=(pi == 1),
            line=dict(color=colors[cond]), marker=dict(color=colors[cond], size=6),
            error_y=dict(type="data", array=[r.get(sk, 0) for r in sel],
                         color=colors[cond], thickness=1, width=3),
            hovertemplate=(f"{LABEL[cond]}<br>width %{{x}}<br>{title} %{{y:.4f}}"
                           "<extra></extra>"),
        ), row=pi, col=1)
    fig.update_yaxes(title_text=title.split(" (")[0], range=[-0.2, 1.1],
                     row=pi, col=1)
fig.update_xaxes(title_text="Hidden width h", type="log", row=4, col=1)
apply_theme(fig, height=900)
fig.write_image(str(HERE / "simulation_diagnostics.svg"))
(HERE / "simulation_diagnostics.html").write_text(
    fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG))
print("wrote", HERE / "simulation_diagnostics.svg")
