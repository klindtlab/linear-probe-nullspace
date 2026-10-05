"""Does pixel-space factor variance explain the D1 inversion? Partly.

Top panel: D1 measured on raw images (area-averaged to 32x32x3) instead of encoder
features, which asks how much of the image variance a linear function of the target
factors explains. Bottom panel: the mean pretrained-minus-random D1 gap over the
encoders run on that dataset, on the same axis order, so the two can be read
against each other.

The proposed account was that the added benchmarks' factors dominate image variance
and SVG-World's do not. It holds for 3D Shapes and fails for dSprites, so it is
reported as a partial explanation rather than a mechanism.
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
rows = sorted(rows, key=lambda r: r["pixel_d1_raw"])
x = [r["dataset_label"] for r in rows]
EMBER, SLATE = "#C4650D", "#4E728A"

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
    subplot_titles=("D1 on raw pixels: how much image variance the factors explain",
                    "Mean encoder D1 gap, pretrained minus matched random"))
fig.add_trace(go.Bar(
    x=x, y=[r["pixel_d1_raw"] for r in rows], marker_color=EMBER, showlegend=False,
    customdata=[[r["pixel_d1_ceiling"], r["pixel_forward_r2"], r["n_targets"]]
                for r in rows],
    hovertemplate=("<b>%{x}</b><br>pixel D1 %{y:.4f}<br>"
                   "ceiling %{customdata[0]:.4f}<br>"
                   "pixel forward R\u00b2 %{customdata[1]:.4f}<br>"
                   "%{customdata[2]} targets<extra></extra>")), row=1, col=1)
fig.add_trace(go.Bar(
    x=x, y=[r["mean_encoder_d1_gap"] for r in rows], marker_color=SLATE,
    showlegend=False,
    hovertemplate="<b>%{x}</b><br>mean D1 gap %{y:+.4f}<extra></extra>"),
    row=2, col=1)
fig.add_hline(y=0, row=2, col=1, line=dict(color="#B4B4B4", width=1))
fig.update_yaxes(title_text="D1 on pixels", row=1, col=1)
fig.update_yaxes(title_text="Encoder D1 \u0394", row=2, col=1)
fig.update_xaxes(title_text="Dataset, ordered by pixel-space D1", row=2, col=1)
apply_theme(fig, height=620)
fig.write_image(str(HERE / "pixel_control.svg"))
(HERE / "pixel_control.html").write_text(
    fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG))
print("wrote", HERE / "pixel_control.svg")
