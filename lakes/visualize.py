"""
lakes/visualize.py

"""

import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
from matplotlib.path import Path as MplPath
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html

DATA_FILE = Path("data") / "lakes" / "data_stream-moda.nc"

FRIENDLY_NAMES = {
    "lmlt": "Mix-Layer Temperature", "lmld": "Mix-Layer Depth",
    "lblt": "Bottom Temperature", "ltlt": "Total Layer Temperature",
    "lict": "Ice Temperature", "licd": "Ice Depth",
}
TEMP_VARS = {"lmlt", "lblt", "ltlt", "lict"}
DEPTH_VARS = {"lmld", "licd"}
COLORS = {"lmlt": "#0891b2", "lblt": "#1e3a8a", "ltlt": "#06b6d4", "lict": "#7dd3fc"}
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _standardize_time(ds):
    if "valid_time" in ds.dims:
        return ds.rename({"valid_time": "time"})
    return ds


def get_data_mask(ds, geometry):
    lons, lats = np.meshgrid(ds["longitude"].values, ds["latitude"].values)
    points = np.vstack((lons.flatten(), lats.flatten())).T
    mask = np.zeros(points.shape[0], dtype=bool)
    if geometry and geometry.get("type") in ["Polygon", "MultiPolygon"]:
        paths = []
        if geometry["type"] == "Polygon":
            paths.append(MplPath(geometry["coordinates"][0]))
        elif geometry["type"] == "MultiPolygon":
            for poly in geometry["coordinates"]:
                paths.append(MplPath(poly[0]))
        for path in paths:
            extents = path.get_extents()
            bbox = (
                (points[:, 0] >= extents.xmin) & (points[:, 0] <= extents.xmax) &
                (points[:, 1] >= extents.ymin) & (points[:, 1] <= extents.ymax)
            )
            sub_points = points[bbox]
            if len(sub_points) > 0:
                sub_mask = path.contains_points(sub_points)
                mask[bbox] |= sub_mask
    if not mask.any():
        mask = np.ones(points.shape[0], dtype=bool)
    return mask.reshape(lons.shape)


def get_masked_mean(ds, mask):
    df = pd.DataFrame({"time": ds["time"].values}).set_index("time")
    valid = np.argwhere(mask)
    if len(valid) == 0:
        return df
    min_lat, min_lon = valid.min(axis=0)
    max_lat, max_lon = valid.max(axis=0)
    sub_mask = mask[min_lat:max_lat + 1, min_lon:max_lon + 1]
    for var in ds.data_vars:
        if len(ds[var].dims) >= 3 and "time" in ds[var].dims:
            data = ds[var][:, min_lat:max_lat + 1, min_lon:max_lon + 1].values
            masked = np.where(sub_mask, data, np.nan)
            df[var] = np.nanmean(masked, axis=(1, 2))
    return df


# ---------------------------------------------------------------------------
# Small reusable UI pieces
# ---------------------------------------------------------------------------

def _empty_layout(message):
    return html.Div(dbc.Alert(message, color="warning", className="text-center"), className="p-4")


def _metric_card(label, value, color="#0f172a"):
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Div(label, className="text-muted small text-uppercase", style={"letterSpacing": "0.5px"}),
            html.Div(value, style={"fontSize": "1.5rem", "fontWeight": 700, "color": color}),
        ]), className="shadow-sm text-center h-100"),
        md=3, xs=6, className="mb-3",
    )


def _chart_card(fig):
    return dbc.Card(dbc.CardBody(dcc.Graph(figure=fig, config={"displayModeBar": False})),
                     className="shadow-sm mb-4")


def _base_layout(height=380, legend_bottom=True):
    layout = dict(
        template="plotly_white", height=height,
        margin=dict(l=55, r=25, t=45, b=45),
        font=dict(family="Inter, sans-serif", size=12, color="#1e293b"),
    )
    if legend_bottom:
        layout["legend"] = dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5)
    return layout


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_state_view(state_name, geojson_geometry=None):
    if not DATA_FILE.exists():
        return _empty_layout(
            f"No lake data found. Expected file: {DATA_FILE}. "
            f"(Note: many inland states have no lake grid cells at all.)"
        )

    try:
        ds = _standardize_time(xr.open_dataset(DATA_FILE))
    except Exception as e:
        return _empty_layout(f"Error loading lake dataset: {e}")

    mask = get_data_mask(ds, geojson_geometry)
    df = get_masked_mean(ds, mask)

    if df.empty or len(df.columns) == 0 or df.isna().all().all():
        return _empty_layout(
            f"No lake grid cells found inside {state_name}. "
            f"ERA5-Land lake variables only exist over open water bodies."
        )

    df_c = df.copy()
    for col in df_c.columns:
        if col in TEMP_VARS and df_c[col].mean(skipna=True) > 100:
            df_c[col] = df_c[col] - 273.15

    temp_cols = [c for c in df_c.columns if c in TEMP_VARS]
    depth_cols = [c for c in df_c.columns if c in DEPTH_VARS]
    df_c["month_num"] = df_c.index.month
    primary_temp = "lmlt" if "lmlt" in df_c.columns else (temp_cols[0] if temp_cols else None)

    # ---- Quick-stat cards ----
    stats_children = []
    if primary_temp:
        stats_children.append(_metric_card("Latest Surface Temp", f"{df_c[primary_temp].iloc[-1]:.1f} °C", "#0891b2"))
        stats_children.append(_metric_card("Peak Summer Temp", f"{df_c[primary_temp].max():.1f} °C", "#dc2626"))
        stats_children.append(_metric_card("Coldest Month", f"{df_c[primary_temp].min():.1f} °C", "#1e40af"))
    if "licd" in df_c.columns:
        ice_months = int((df_c["licd"] > 0.01).sum())
        stats_children.append(_metric_card("Months w/ Ice Cover", f"{ice_months} / {len(df_c)}", "#7dd3fc"))
    stats_row = dbc.Row(stats_children, className="g-3 mb-2") if stats_children else html.Div()

    # ---- Chart 1: Lake temperature time series ----
    fig1 = go.Figure()
    for col in temp_cols:
        fig1.add_trace(go.Scatter(x=df_c.index, y=df_c[col], mode="lines",
                                   name=FRIENDLY_NAMES.get(col, col),
                                   line=dict(width=2, color=COLORS.get(col, "#0891b2"))))
    if not temp_cols:
        fig1.add_annotation(text="No lake temperature variables found", x=0.5, y=0.5,
                             xref="paper", yref="paper", showarrow=False)
    fig1.update_layout(title="Lake Temperature Profile Over Time", yaxis_title="°C", **_base_layout())

    # ---- Chart 2: Depth variables (mix-layer depth / ice depth) ----
    fig2 = go.Figure()
    for col in depth_cols:
        fig2.add_trace(go.Scatter(x=df_c.index, y=df_c[col], mode="lines", fill="tozeroy",
                                   name=FRIENDLY_NAMES.get(col, col),
                                   line=dict(width=1.5, color="#1e40af" if col == "lmld" else "#93c5fd")))
    if not depth_cols:
        fig2.add_annotation(text="No lake depth variables found", x=0.5, y=0.5,
                             xref="paper", yref="paper", showarrow=False)
    fig2.update_layout(title="Mix-Layer Depth / Ice Depth Over Time", yaxis_title="m", **_base_layout())

    # ---- Chart 3: Surface temp vs mix-layer depth scatter ----
    fig3 = go.Figure()
    if "lmlt" in df_c.columns and "lmld" in df_c.columns:
        fig3.add_trace(go.Scatter(
            x=df_c["lmld"], y=df_c["lmlt"], mode="markers",
            marker=dict(size=9, color=df_c["month_num"], colorscale="Viridis",
                        colorbar=dict(title="Month")),
            text=[t.strftime("%Y-%m") for t in df_c.index],
        ))
        fig3.update_xaxes(title_text="Mix-Layer Depth (m)")
        fig3.update_yaxes(title_text="Mix-Layer Temp (°C)")
    else:
        fig3.add_annotation(text="Need lmlt & lmld for this scatter", x=0.5, y=0.5,
                             xref="paper", yref="paper", showarrow=False)
    fig3.update_layout(title="Temperature vs Mix-Layer Depth", **_base_layout(legend_bottom=False))

    # ---- Chart 4: Seasonal climatology bar ----
    fig4 = go.Figure()
    if primary_temp:
        clim = df_c.groupby("month_num")[primary_temp].mean().reindex(range(1, 13))
        fig4.add_trace(go.Bar(x=MONTH_LABELS, y=clim.values, marker_color=clim.values,
                               marker_colorscale="RdBu_r"))
        fig4.update_yaxes(title_text="°C")
    fig4.update_layout(title="Surface Temperature Seasonal Climatology",
                        **_base_layout(legend_bottom=False))

    # ---- Chart 5 (NEW): Ice-cover calendar (year x month heatmap) ----
    # A quick "at a glance" freeze calendar - much easier to read trends
    # across multiple years than the raw ice-depth time series alone.
    fig5 = go.Figure()
    if "licd" in df_c.columns:
        df_c["year"] = df_c.index.year
        pivot = df_c.pivot_table(index="year", columns="month_num", values="licd", aggfunc="mean")
        pivot = pivot.reindex(columns=range(1, 13))
        fig5.add_trace(go.Heatmap(
            z=pivot.values, x=MONTH_LABELS, y=pivot.index.astype(str),
            colorscale="Blues", colorbar=dict(title="Ice Depth (m)"),
        ))
    else:
        fig5.add_annotation(text="No ice-depth variable (licd) found", x=0.5, y=0.5,
                             xref="paper", yref="paper", showarrow=False)
    fig5.update_layout(title="Ice Cover Calendar (Year × Month)", **_base_layout(legend_bottom=False))

    grid = dbc.Row([
        dbc.Col(_chart_card(fig1), md=6), dbc.Col(_chart_card(fig2), md=6),
        dbc.Col(_chart_card(fig3), md=6), dbc.Col(_chart_card(fig4), md=6),
        dbc.Col(_chart_card(fig5), md=12),
    ], className="g-4")

    return html.Div([
        html.H4(f"Lake Analysis — {state_name}", className="mb-3"),
        stats_row,
        grid,
    ])