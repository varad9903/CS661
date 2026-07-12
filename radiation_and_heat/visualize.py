"""
radiation_and_heat/visualize.py

"""

import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
from matplotlib.path import Path as MplPath
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html

DATA_FILE = Path("data") / "heat_radiation" / "data_stream-moda.nc"

FRIENDLY_NAMES = {
    "ssr": "Net Solar Radiation", "str": "Net Thermal Radiation", "fal": "Forecast Albedo",
    "ssrd": "Solar Radiation (Downward)", "strd": "Thermal Radiation (Downward)",
    "slhf": "Latent Heat Flux", "sshf": "Sensible Heat Flux",
}
COLORS = {
    "ssr": "#f59e0b", "str": "#ef4444", "fal": "#a855f7", "ssrd": "#fbbf24",
    "strd": "#fb923c", "slhf": "#3b82f6", "sshf": "#10b981",
}
FLUX_VARS = {"ssr", "str", "ssrd", "strd", "slhf", "sshf"}
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
            f"No radiation/heat data found. Expected file: {DATA_FILE}. "
            f"Download the ERA5-Land 'radiation' NetCDF and place it there."
        )

    try:
        ds = _standardize_time(xr.open_dataset(DATA_FILE))
    except Exception as e:
        return _empty_layout(f"Error loading radiation dataset: {e}")

    mask = get_data_mask(ds, geojson_geometry)
    df = get_masked_mean(ds, mask)

    if df.empty or len(df.columns) == 0:
        return _empty_layout(f"No radiation/heat variables found for {state_name}.")

    plot_vars = [c for c in df.columns if c in FRIENDLY_NAMES] or list(df.columns)
    flux_vars = [c for c in plot_vars if c in FLUX_VARS]
    energy_vars = [c for c in ["ssr", "str", "slhf", "sshf"] if c in df.columns]
    has_albedo = "fal" in df.columns
    df["month_num"] = df.index.month

    # ---- Quick-stat cards ----
    net_solar = df["ssr"].iloc[-1] if "ssr" in df.columns else None
    albedo_latest = df["fal"].iloc[-1] if has_albedo else None
    stats_children = []
    if net_solar is not None:
        stats_children.append(_metric_card("Latest Net Solar", f"{net_solar/1e6:.2f} MJ/m²", "#f59e0b"))
    if albedo_latest is not None:
        stats_children.append(_metric_card("Latest Albedo", f"{albedo_latest:.2f}", "#a855f7"))
    if "slhf" in df.columns:
        stats_children.append(_metric_card("Avg Latent Heat", f"{df['slhf'].mean()/1e6:.2f} MJ/m²", "#3b82f6"))
    if "sshf" in df.columns:
        stats_children.append(_metric_card("Avg Sensible Heat", f"{df['sshf'].mean()/1e6:.2f} MJ/m²", "#10b981"))
    stats_row = dbc.Row(stats_children, className="g-3 mb-2") if stats_children else html.Div()

    # ---- Chart 1: Flux time series + albedo on secondary axis ----
    fig1 = go.Figure()
    for col in flux_vars[:5]:
        fig1.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines",
                                   name=FRIENDLY_NAMES.get(col, col),
                                   line=dict(width=2, color=COLORS.get(col, "#6366f1"))))
    if has_albedo:
        fig1.add_trace(go.Scatter(x=df.index, y=df["fal"], mode="lines", name="Forecast Albedo",
                                   line=dict(width=2, color=COLORS["fal"], dash="dash"),
                                   yaxis="y2"))
        fig1.update_layout(yaxis2=dict(title="Albedo (0-1)", overlaying="y", side="right", range=[0, 1]))
    fig1.update_layout(title="Radiation & Heat Fluxes Over Time", yaxis_title="J/m²",
                        **_base_layout())

    # ---- Chart 2: Stacked energy balance ----
    fig2 = go.Figure()
    for col in energy_vars:
        fig2.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines",
                                   name=FRIENDLY_NAMES.get(col, col),
                                   line=dict(width=0.5, color=COLORS.get(col, "#6366f1")),
                                   stackgroup="energy"))
    if not energy_vars:
        fig2.add_annotation(text="ssr/str/slhf/sshf not found in dataset", x=0.5, y=0.5,
                             xref="paper", yref="paper", showarrow=False)
    fig2.update_layout(title="Surface Energy Balance (Stacked)", yaxis_title="J/m²", **_base_layout())

    # ---- Chart 3: Monthly climatology heatmap (flux vars only) ----
    if flux_vars:
        clim = df[flux_vars].groupby(df["month_num"]).mean().reindex(range(1, 13))
        fig3 = go.Figure(go.Heatmap(
            z=clim[flux_vars].to_numpy().T, x=MONTH_LABELS,
            y=[FRIENDLY_NAMES.get(c, c) for c in flux_vars],
            colorscale="Turbo", colorbar=dict(title="J/m²"),
        ))
    else:
        fig3 = go.Figure()
        fig3.add_annotation(text="No flux variables found", x=0.5, y=0.5,
                             xref="paper", yref="paper", showarrow=False)
    fig3.update_layout(title="Monthly Climatology Heatmap", **_base_layout(legend_bottom=False))

    # ---- Chart 4: Solar vs turbulent flux scatter ----
    x_var = "ssr" if "ssr" in df.columns else (plot_vars[0] if plot_vars else None)
    y_var = "slhf" if "slhf" in df.columns else ("sshf" if "sshf" in df.columns else None)
    fig4 = go.Figure()
    if x_var and y_var:
        fig4.add_trace(go.Scatter(
            x=df[x_var], y=df[y_var], mode="markers",
            marker=dict(size=9, color=df["month_num"], colorscale="Viridis",
                        colorbar=dict(title="Month")),
            text=[t.strftime("%Y-%m") for t in df.index],
        ))
        fig4.update_xaxes(title_text=FRIENDLY_NAMES.get(x_var, x_var))
        fig4.update_yaxes(title_text=FRIENDLY_NAMES.get(y_var, y_var))
    else:
        fig4.add_annotation(text="Insufficient variables for scatter", x=0.5, y=0.5,
                             xref="paper", yref="paper", showarrow=False)
    fig4.update_layout(title="Solar Radiation vs Turbulent Heat Fluxes",
                        **_base_layout(legend_bottom=False))

    # ---- Chart 5 (NEW): Monthly Bowen ratio ----
    # Bowen ratio = sensible heat flux / latent heat flux. High ratio -> dry
    # surface (energy goes into heating air); low/near-zero -> wet surface
    # (energy goes into evaporation). A natural complement to the scatter
    # above and directly readable as "how dry is the surface this month".
    fig5 = go.Figure()
    if "sshf" in df.columns and "slhf" in df.columns:
        bowen = (df["sshf"] / df["slhf"].replace(0, np.nan)).groupby(df["month_num"]).mean().reindex(range(1, 13))
        fig5.add_trace(go.Bar(x=MONTH_LABELS, y=bowen.values, marker_color="#f97316"))
        fig5.update_layout(yaxis_title="Bowen Ratio (SSHF / SLHF)")
    else:
        fig5.add_annotation(text="Need sshf & slhf for Bowen ratio", x=0.5, y=0.5,
                             xref="paper", yref="paper", showarrow=False)
    fig5.update_layout(title="Monthly Bowen Ratio (Surface Dryness Indicator)",
                        **_base_layout(legend_bottom=False))

    grid = dbc.Row([
        dbc.Col(_chart_card(fig1), md=6), dbc.Col(_chart_card(fig2), md=6),
        dbc.Col(_chart_card(fig3), md=6), dbc.Col(_chart_card(fig4), md=6),
        dbc.Col(_chart_card(fig5), md=12),
    ], className="g-4")

    return html.Div([
        html.H4(f"Radiation & Heat Flux Analysis — {state_name}", className="mb-3"),
        stats_row,
        grid,
    ])