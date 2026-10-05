"""Encoding sensitivity heatmap: pretrained-minus-random gap in reverse
predictivity for 2 transformer encoders x 3 benchmarks x 3 target
encodings, averaged over 3 dataset samples. Diverging colorscale centered
at zero because the gap is signed. Loads co-located data.json; writes
encoding_sensitivity.svg."""
import json
from pathlib import Path

import plotly.graph_objects as go

HERE = Path(__file__).parent
rows = json.loads((HERE / "data.json").read_text())

DIVERGING = [[0.00, "#4E728A"], [0.25, "#8FBCD8"], [0.50, "#F6F5F0"],
             [0.75, "#DE9D50"], [1.00, "#C4650D"]]

MODEL_LABEL = {"clip_b16": "CLIP-B/16", "dinov3_b16": "DINOv3-B/16"}
DATASET_LABEL = {"shapes3d": "3D Shapes", "dsprites": "dSprites",
                 "mpi3d_realistic": "MPI3D-realistic"}
ENCODING_LABEL = {
    "canonical": "canonical",
    "ordinal": "plain ordinal",
    "with_onehot": "canonical + one-hot identity",
}
COL_ORDER = [("clip_b16", d) for d in ("shapes3d", "dsprites", "mpi3d_realistic")] + \
            [("dinov3_b16", d) for d in ("shapes3d", "dsprites", "mpi3d_realistic")]
ROW_ORDER = ["canonical", "ordinal", "with_onehot"]


def apply_theme(fig, *, height=400):
    fig.update_layout(
        height=height,
        margin=dict(t=16, r=92, b=24, l=24, autoexpand=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1D272A",
                  family="'Suisse Intl', -apple-system, BlinkMacSystemFont, Arial, sans-serif",
                  size=13),
        title=None,
        modebar=dict(orientation="h", bgcolor="rgba(0,0,0,0)",
                     color="#1D272A", activecolor="#C4650D",
                     remove=["lasso2d", "select2d", "autoScale2d"]),
    )
    fig.update_xaxes(automargin=True, ticks="outside",
                     tickfont=dict(size=12), title_font=dict(size=13))
    fig.update_yaxes(automargin=True, ticks="outside",
                     tickfont=dict(size=12), title_font=dict(size=13))
    return fig


def build():
    lookup = {(r["model"], r["dataset"], r["encoding"]): r["gap"] for r in rows}
    z, text = [], []
    for enc in ROW_ORDER:
        zrow, trow = [], []
        for model, ds in COL_ORDER:
            g = lookup[(model, ds, enc)]
            zrow.append(g)
            trow.append(f"{g:+.4f}")
        z.append(zrow)
        text.append(trow)
    xlabels = [f"{MODEL_LABEL[m]}<br>{DATASET_LABEL[d]}" for m, d in COL_ORDER]
    ylabels = [ENCODING_LABEL[e] for e in ROW_ORDER]
    zmax = max(abs(v) for zrow in z for v in zrow)
    fig = go.Figure(go.Heatmap(
        z=z, x=xlabels, y=ylabels,
        zmin=-zmax, zmax=zmax,
        colorscale=DIVERGING, zmid=0,
        text=text, texttemplate="%{text}",
        textfont=dict(size=12, color="#1D272A"),
        xgap=2, ygap=2,
        colorbar=dict(
            title=dict(text="reverse-<br>predictivity<br>gap", side="top"),
            thickness=14, len=1.0, x=1.0, xanchor="left", xpad=8,
            outlinewidth=0,
        ),
        hovertemplate="%{x}<br>%{y}<br>gap %{z:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Encoder and benchmark")
    fig.update_yaxes(title_text="Target encoding", autorange="reversed")
    return fig


if __name__ == "__main__":
    fig = build()
    apply_theme(fig, height=380)
    fig.write_image(str(HERE / "encoding_sensitivity.svg"))
    print("wrote", HERE / "encoding_sensitivity.svg")
