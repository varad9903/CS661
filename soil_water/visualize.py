"""
soil_water/visualize.py

"""

import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.path import Path as MplPath
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html

DATA_FILE = Path("data") / "soil_water" / "data_stream-moda.nc"

LAYER_LABELS = {
    "swvl1": "Layer 1 (0-7cm)",
    "swvl2": "Layer 2 (7-28cm)",
    "swvl3": "Layer 3 (28-100cm)",
    "swvl4": "Layer 4 (100-289cm)",
}
# approximate layer thickness in metres, used to weight layers into a
# single "total column water storage" figure (Panel 5)
LAYER_THICKNESS_M = {"swvl1": 0.07, "swvl2": 0.21, "swvl3": 0.72, "swvl4": 1.89}
LAYER_COLORS = {"swvl1": "#92400e", "swvl2": "#b45309", "swvl3": "#0369a1", "swvl4": "#1e3a8a"}
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
    return dbc.Card(
        dbc.CardBody(dcc.Graph(figure=fig, config={"displayModeBar": False})),
        className="shadow-sm mb-4",
    )


def _base_layout(height=380, legend_bottom=True):
    layout = dict(
        template="plotly_white",
        height=height,
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
            f"No soil water data found. Expected file: {DATA_FILE}. "
            f"Download the ERA5-Land 'soil_water' NetCDF and place it there."
        )

    try:
        ds = _standardize_time(xr.open_dataset(DATA_FILE))
    except Exception as e:
        return _empty_layout(f"Error loading soil water dataset: {e}")

    mask = get_data_mask(ds, geojson_geometry)
    df = get_masked_mean(ds, mask)

    if df.empty or len(df.columns) == 0:
        return _empty_layout(f"No soil water variables found for {state_name}.")

    layer_cols = [c for c in ["swvl1", "swvl2", "swvl3", "swvl4"] if c in df.columns] or list(df.columns)
    df["month_num"] = df.index.month
    top_layer = layer_cols[0]

    # ---- Quick-stat cards (the "extra info" beyond the charts) ----
    latest = df[top_layer].iloc[-1]
    avg = df[top_layer].mean()
    vmin, vmax = df[top_layer].min(), df[top_layer].max()
    stats_row = dbc.Row([
        _metric_card("Latest (Top Layer)", f"{latest:.3f} m³/m³", "#92400e"),
        _metric_card("2yr Average", f"{avg:.3f} m³/m³", "#0f172a"),
        _metric_card("Driest Month", f"{vmin:.3f} m³/m³", "#dc2626"),
        _metric_card("Wettest Month", f"{vmax:.3f} m³/m³", "#0369a1"),
    ], className="g-3 mb-2")

    # ---- Chart 1: Multi-layer time series ----
    fig1 = go.Figure()
    for col in layer_cols:
        fig1.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines", name=LAYER_LABELS.get(col, col),
            line=dict(width=2.2, color=LAYER_COLORS.get(col, "#334155")),
        ))
    fig1.update_layout(title="Soil Moisture by Depth Layer Over Time",
                        yaxis_title="Volumetric Water (m³/m³)", **_base_layout())

    # ---- Chart 2: Depth-profile heatmap (layers on y, months*years on x) ----
    if len(layer_cols) > 1:
        z = df[layer_cols].to_numpy().T
        fig2 = go.Figure(go.Heatmap(
            z=z, x=df.index, y=[LAYER_LABELS.get(c, c) for c in layer_cols],
            colorscale="YlGnBu", colorbar=dict(title="m³/m³"),
        ))
        fig2.update_yaxes(autorange="reversed")  # shallow layer on top, like a soil profile
        fig2.update_layout(title="Depth Profile Evolution", **_base_layout(legend_bottom=False))
    else:
        fig2 = go.Figure()
        fig2.add_annotation(text="Need 2+ layers for a depth profile", x=0.5, y=0.5,
                             xref="paper", yref="paper", showarrow=False)
        fig2.update_layout(**_base_layout(legend_bottom=False))

    # ---- Chart 3: Layer correlation matrix ----
    if len(layer_cols) > 1:
        corr = df[layer_cols].corr()
        labels = [LAYER_LABELS.get(c, c) for c in layer_cols]
        fig3 = px.imshow(
            corr.values, x=labels, y=labels, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
        )
        fig3.update_layout(title="Layer Correlation Matrix", coloraxis_colorbar=dict(title="r"),
                            **_base_layout(legend_bottom=False))
    else:
        fig3 = go.Figure()
        fig3.add_annotation(text="Need 2+ layers for correlation", x=0.5, y=0.5,
                             xref="paper", yref="paper", showarrow=False)
        fig3.update_layout(**_base_layout(legend_bottom=False))

    # ---- Chart 4: Seasonal box plot of the top (most reactive) layer ----
    fig4 = go.Figure()
    for m in range(1, 13):
        vals = df.loc[df["month_num"] == m, top_layer]
        if len(vals) > 0:
            fig4.add_trace(go.Box(y=vals, name=MONTH_LABELS[m - 1], marker_color="#0369a1",
                                   showlegend=False))
    fig4.update_layout(title=f"Seasonal Distribution — {LAYER_LABELS.get(top_layer, top_layer)}",
                        yaxis_title="m³/m³", **_base_layout(legend_bottom=False))

    # ---- Chart 5 (NEW): Total column water storage ----
    # Depth-weighted sum of all layers approximates total soil water column
    # storage (in mm of water equivalent) - a single number that summarises
    # "how much water is the soil holding this month", useful for drought /
    # monsoon-recovery tracking that no single layer alone shows.
    weighted = sum(df[c] * LAYER_THICKNESS_M.get(c, 0) for c in layer_cols) * 1000  # m -> mm
    fig5 = go.Figure(go.Scatter(
        x=df.index, y=weighted, mode="lines", fill="tozeroy",
        line=dict(width=2, color="#065f46"), name="Total Column Storage",
    ))
    fig5.update_layout(title="Total Soil Column Water Storage (0-289cm, depth-weighted)",
                        yaxis_title="mm water equivalent", **_base_layout(legend_bottom=False))

    grid = dbc.Row([
        dbc.Col(_chart_card(fig1), md=6), dbc.Col(_chart_card(fig2), md=6),
        dbc.Col(_chart_card(fig3), md=6), dbc.Col(_chart_card(fig4), md=6),
        dbc.Col(_chart_card(fig5), md=12),
    ], className="g-4")

    return html.Div([
        html.H4(f"Soil Water Analysis — {state_name}", className="mb-3"),
        stats_row,
        grid,
    ])