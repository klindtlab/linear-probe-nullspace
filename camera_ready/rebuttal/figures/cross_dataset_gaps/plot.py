"""The same three encoders across every dataset, one panel per diagnostic.

Latent spaces of different datasets are not comparable in absolute terms, so what
is plotted is each encoder's pretrained-minus-random gap within its own dataset,
grouped by dataset. Error bars are 95% paired bootstrap intervals over held-out
split units.
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
PANELS = [("forward", "Forward R\u00b2 \u0394"), ("D1_raw", "Reverse R\u00b2 \u0394 (D1)"),
          ("D3", "Out-of-distribution R\u00b2 \u0394 (D3)")]
datasets = list(dict.fromkeys(r["dataset_label"] for r in rows))
models = list(dict.fromkeys(r["model"] for r in rows))

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=[t for _, t in PANELS])
for pi, (diag, title) in enumerate(PANELS, start=1):
    for m in models:
        sel = {r["dataset_label"]: r for r in rows
               if r["diagnostic"] == diag and r["model"] == m}
        xs = [d for d in datasets if d in sel]
        if not xs:
            continue
        fig.add_trace(go.Bar(
            x=xs, y=[sel[d]["diff"] for d in xs],
            name=sel[xs[0]]["model_label"], marker_color=colors.get(m, "#31362E"),
            showlegend=(pi == 1),
            error_y=dict(type="data", symmetric=False,
                         array=[sel[d]["diff_ci_hi"] - sel[d]["diff"] for d in xs],
                         arrayminus=[sel[d]["diff"] - sel[d]["diff_ci_lo"] for d in xs],
                         color="#31362E", thickness=1.2, width=4),
            customdata=[[sel[d]["pretrained"], sel[d]["random"],
                         sel[d]["diff_ci_lo"], sel[d]["diff_ci_hi"]] for d in xs],
            hovertemplate=("<b>%{fullData.name}</b> on %{x}<br>gap %{y:.4f} "
                           "(95% CI [%{customdata[2]:.4f}, %{customdata[3]:.4f}])<br>"
                           "pretrained %{customdata[0]:.4f} / random "
                           "%{customdata[1]:.4f}<extra></extra>"),
        ), row=pi, col=1)
    fig.add_hline(y=0, row=pi, col=1, line=dict(color="#B4B4B4", width=1))
    fig.update_yaxes(title_text=title, row=pi, col=1)
fig.update_xaxes(title_text="Dataset", row=3, col=1)
apply_theme(fig, height=860)
fig.update_layout(barmode="group")
fig.write_image(str(HERE / "cross_dataset_gaps.svg"))
(HERE / "cross_dataset_gaps.html").write_text(
    fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG))
print("wrote", HERE / "cross_dataset_gaps.svg")
