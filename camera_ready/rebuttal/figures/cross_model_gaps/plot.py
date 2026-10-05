"""Pretrained minus matched-random gaps on SVG-World.

Forward and reverse gaps carry 95% paired bootstrap intervals over held-out
scenes. The cross-style transfer panel reports mean Pearson rho over the 12
position latents, because per-style scale mismatch makes the transfer R^2
uninterpretable; its band is the spread across five splits and both transfer
directions rather than a bootstrap.

Each encoder is compared only against a random initialization of its own
architecture, so a gap cannot come from a width or capacity difference. The three
diagnostics sit on different scales, so they are separate panels rather than
separate series in one panel.
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
          ("D3_rho", "Cross-style transfer \u03c1 \u0394 (D3)")]
order = list(dict.fromkeys(r["model"] for r in rows if r["diagnostic"] == "D1_raw"))

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=[t for _, t in PANELS])
for pi, (diag, title) in enumerate(PANELS, start=1):
    by = {r["model"]: r for r in rows if r["diagnostic"] == diag}
    xs = [m for m in order if m in by]
    if not xs:
        continue
    fig.add_trace(go.Bar(
        x=[by[m]["model_label"] for m in xs],
        y=[by[m]["diff"] for m in xs],
        marker_color=[colors.get(m, "#31362E") for m in xs],
        showlegend=False,
        error_y=dict(type="data", symmetric=False,
                     array=[by[m]["diff_ci_hi"] - by[m]["diff"] for m in xs],
                     arrayminus=[by[m]["diff"] - by[m]["diff_ci_lo"] for m in xs],
                     color="#31362E", thickness=1.2, width=4),
        customdata=[[by[m]["pretrained"], by[m]["random"], by[m]["diff_ci_lo"],
                     by[m]["diff_ci_hi"], by[m]["paradigm"]] for m in xs],
        hovertemplate=("<b>%{x}</b><br>gap %{y:.4f} "
                       "(95% CI [%{customdata[2]:.4f}, %{customdata[3]:.4f}])<br>"
                       "pretrained %{customdata[0]:.4f} / random %{customdata[1]:.4f}"
                       "<br><i>%{customdata[4]}</i><extra></extra>"),
    ), row=pi, col=1)
    fig.add_hline(y=0, row=pi, col=1, line=dict(color="#B4B4B4", width=1))
    fig.update_yaxes(title_text=title, row=pi, col=1)
fig.update_xaxes(title_text="Encoder", row=3, col=1)
apply_theme(fig, height=820)
fig.write_image(str(HERE / "cross_model_gaps.svg"))
(HERE / "cross_model_gaps.html").write_text(
    fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG))
print("wrote", HERE / "cross_model_gaps.svg")
