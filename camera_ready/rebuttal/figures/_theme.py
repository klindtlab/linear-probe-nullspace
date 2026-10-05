"""Shared Plotly theme for every figure bundle in this experiment.

Each `figures/<name>/plot.py` is self-contained apart from importing this file,
which carries only the house theme (no experiment-module imports, no data
recomputation), so any figure re-renders from its own committed data.
"""
from __future__ import annotations

EDITORIAL_8 = ['#C4650D', '#4E728A', '#2E6E4E', '#988453',
               '#B9605B', '#7495AB', '#84713A', '#31362E']
SEQUENTIAL = [[0.00, '#F0E2D0'], [0.35, '#EDD8C5'], [0.65, '#DE9D50'],
              [1.00, '#C4650D']]
DIVERGING = [[0.00, '#4E728A'], [0.25, '#8FBCD8'], [0.50, '#F6F5F0'],
             [0.75, '#DE9D50'], [1.00, '#C4650D']]
PLOTLY_CONFIG = {
    "responsive": True, "displayModeBar": "hover", "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "filename": "figure", "scale": 2},
}


def apply_theme(fig, *, height=400, legend='auto'):
    if legend == 'auto':
        legend = dict(orientation='h', xref='container', x=0, xanchor='left',
                      yref='container', y=0, yanchor='bottom',
                      title=dict(side='top'))
    fig.update_layout(
        height=height,
        margin=dict(t=40, r=24, b=24, l=24, autoexpand=True),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1D272A',
                  family="'Suisse Intl', -apple-system, BlinkMacSystemFont, Arial, sans-serif",
                  size=13),
        title=None,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#1D272A', size=12),
                    **legend),
        colorway=EDITORIAL_8,
        hoverlabel=dict(bgcolor='#FFFFFF', bordercolor='#B4B4B4',
                        font=dict(family='ui-monospace, Menlo, monospace', size=12,
                                  color='#1D272A')),
        modebar=dict(orientation='h', bgcolor='rgba(0,0,0,0)', color='#1D272A',
                     activecolor='#C4650D',
                     remove=['lasso2d', 'select2d', 'autoScale2d']),
        uniformtext=dict(minsize=10, mode='hide'),
    )
    fig.update_traces(textposition='none', selector=dict(type='bar'))
    for tr in fig.data:
        if getattr(tr, 'type', None) in ('heatmap', 'contour'):
            if not getattr(tr, 'colorscale', None) and not getattr(tr, 'coloraxis', None):
                tr.colorscale = SEQUENTIAL
        if hasattr(tr, 'cliponaxis'):
            tr.cliponaxis = True
    has_cb = False

    def _fit(cb):
        cb.title.side = 'top'; cb.thickness = 14; cb.len = 1.0
        cb.x = 1.0; cb.xanchor = 'left'; cb.xpad = 8; cb.outlinewidth = 0
    for cax in fig.select_coloraxes():
        if not getattr(cax, 'colorscale', None):
            cax.colorscale = SEQUENTIAL
        has_cb = True; _fit(cax.colorbar)
    for tr in fig.data:
        if getattr(tr, 'type', None) in ('heatmap', 'contour') and not getattr(tr, 'coloraxis', None):
            has_cb = True; _fit(tr.colorbar)
    if has_cb:
        fig.update_layout(margin_r=92)
    fig.update_xaxes(gridcolor='#B4B4B4', zerolinecolor='#B4B4B4', zerolinewidth=1,
                     automargin=True, ticks='outside', tickfont=dict(size=12),
                     title_font=dict(size=13), autotickangles=[0, 30])
    fig.update_yaxes(gridcolor='#B4B4B4', zerolinecolor='#B4B4B4', zerolinewidth=1,
                     automargin=True, ticks='outside', tickfont=dict(size=12),
                     title_font=dict(size=13))
    for axis in list(fig.select_xaxes()) + list(fig.select_yaxes()):
        if getattr(axis, 'type', None) != 'log':
            continue
        t = axis.title.text if (axis.title is not None and getattr(axis.title, 'text', None)) else ''
        if t and '(log' not in t.lower():
            axis.title.text = f'{t} (log)'
        elif not t:
            axis.title = dict(text='(log scale)')
        axis.minor = dict(ticks='outside', ticklen=3, tickcolor='#B4B4B4', showgrid=False)
    fig.update_layout(title=None, title_text=None)
    return fig
