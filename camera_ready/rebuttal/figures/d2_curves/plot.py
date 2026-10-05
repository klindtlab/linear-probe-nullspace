"""S(k), the D2 statistic, against the number of retained principal directions.

One definition now: reverse R-squared after PCA and whitening fitted on the
training split only. Pretrained encoders on top, their matched random
initializations below, on a shared axis so the two panels are directly comparable.

Note on the shape. In the controlled simulation a trained linear world model gives
S(k) = 1 up to the latent dimension and then falls, because whitening rescales the
sub-leading directions (which hold only residue) to unit variance. No real encoder
here shows that plateau: every curve peaks in the single digits of k and decays,
which is the kernel-machine signature the paper predicts for a representation that
is not a linear world model.
"""
import json
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _theme import PLOTLY_CONFIG, apply_theme  # noqa: E402

payload = json.loads((HERE / "data.json").read_text())
colors = json.loads((HERE.parent / "entity_colors.json").read_text())
meta = payload.get("meta", {})
r_eff = meta.get("r_eff")

PANELS = [("pre", "Pretrained"), ("rand", "Matched random initialization")]
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                    subplot_titles=[t for _, t in PANELS])
seen = set()
for row, (variant, _) in enumerate(PANELS, start=1):
    for series in payload["series"]:
        if series["variant"] != variant:
            continue
        m = series["model"]
        show = m not in seen
        seen.add(m)
        fig.add_trace(go.Scatter(
            x=series["k"], y=series["S"], name=series["model_label"],
            mode="lines+markers", legendgroup=m, showlegend=show,
            line=dict(color=colors.get(m, "#31362E")),
            marker=dict(color=colors.get(m, "#31362E"), size=5),
            hovertemplate=(series["model_label"] + "<br>k %{x}<br>S(k) %{y:.4f}"
                           "<extra></extra>"),
        ), row=row, col=1)

# Plotly's add_vline takes DATA coordinates while an annotation on a log axis
# takes log10 units; both are needed to land the marker at k = r_eff.
if r_eff:
    import math
    for row in (1, 2):
        fig.add_vline(x=r_eff, row=row, col=1,
                      line=dict(color="#B4B4B4", dash="dot", width=1))
    fig.add_annotation(x=math.log10(r_eff), y=1.0, yref="y domain", row=1, col=1,
                       text="k = r_eff = %d" % r_eff, showarrow=False,
                       xanchor="left", yanchor="top",
                       font=dict(size=11, color="#7B7B7B"))

fig.update_xaxes(type="log", range=[0, 2.95], row=1, col=1)
fig.update_xaxes(type="log", range=[0, 2.95],
                 title_text="Retained principal directions k", row=2, col=1)
for row in (1, 2):
    fig.update_yaxes(title_text="S(k)", range=[-0.02, 0.68], row=row, col=1)
apply_theme(fig, height=680)
# apply_theme labels any untitled log axis "(log scale)"; on a shared-axis figure
# that would print between the panels, so clear the upper axis title after theming.
fig.update_xaxes(title_text=None, row=1, col=1)
fig.write_image(str(HERE / "d2_curves.svg"))
(HERE / "d2_curves.html").write_text(
    fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG))
print("wrote", HERE / "d2_curves.svg")
