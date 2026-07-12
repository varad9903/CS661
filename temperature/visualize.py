"""
temperature/visualize.py

"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.path import Path as MplPath

import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, no_update

DATA_FILE = Path("data") / "temperature" / "data_stream-moda.nc"


# ---------------------------------------------------------------------------
# Data loading / masking helpers
# ---------------------------------------------------------------------------

def _standardize_time(ds):
    if "valid_time" in ds.dims:
        return ds.rename({"valid_time": "time"})
    return ds


@lru_cache(maxsize=1)
def _load_dataset():
    """Open (and cache) the temperature dataset. xarray keeps data on disk
    until a slice is actually read, so this is cheap even for the full
    global grid."""
    if not DATA_FILE.exists():
        return None
    return _standardize_time(xr.open_dataset(DATA_FILE))


def get_data_mask(ds, geometry):
    """Boolean (lat, lon) mask of grid cells that fall inside `geometry`."""
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
    """DataFrame of the spatial mean over time for every 3D (time, lat, lon) variable."""
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


def _get_bbox(mask):
    valid = np.argwhere(mask)
    if len(valid) == 0:
        return None
    min_lat, min_lon = valid.min(axis=0)
    max_lat, max_lon = valid.max(axis=0)
    return int(min_lat), int(min_lon), int(max_lat), int(max_lon)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _empty_layout(message):
    return html.Div(dbc.Alert(message, color="warning", className="text-center"),
                     className="p-4")


def _build_timeseries_figure(df_c, region_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_c.index, y=df_c["t2m"], mode="lines+markers", name="2m Temperature (t2m)",
        line=dict(width=2, color="#ef4444"), marker=dict(size=7),
    ))
    if "d2m" in df_c.columns:
        fig.add_trace(go.Scatter(
            x=df_c.index, y=df_c["d2m"], mode="lines", name="2m Dewpoint (d2m)",
            line=dict(width=1.5, color="#3b82f6", dash="dot"),
        ))
    fig.update_layout(
        title=f"Monthly Temperature — {region_name}",
        template="plotly_white", height=420,
        margin=dict(l=40, r=20, t=50, b=60),
        xaxis_title="Month", yaxis_title="°C",
        clickmode="event+select",
        # anchored BELOW the plot so it never competes with the title above it
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
    )
    return fig


def _build_climatology_figure(df_c, region_name):
    """Average seasonal cycle: mean t2m (and d2m, if present) per calendar
    month, collapsing both years into one 12-month climatology."""
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_num = df_c.index.month
    clim_t2m = df_c["t2m"].groupby(month_num).mean().reindex(range(1, 13))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=month_labels, y=clim_t2m.values, name="2m Temperature (t2m)",
        marker_color=clim_t2m.values, marker_colorscale="RdBu_r",
    ))
    if "d2m" in df_c.columns:
        clim_d2m = df_c["d2m"].groupby(month_num).mean().reindex(range(1, 13))
        fig.add_trace(go.Scatter(
            x=month_labels, y=clim_d2m.values, mode="lines+markers", name="2m Dewpoint (d2m)",
            line=dict(width=2, color="#3b82f6", dash="dot"),
        ))
    fig.update_layout(
        title=f"Average Seasonal Cycle — {region_name}",
        template="plotly_white", height=380,
        margin=dict(l=40, r=20, t=50, b=60),
        yaxis_title="°C",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
    )
    return fig


def _build_anomaly_figure(df_c, region_name):
    """Monthly t2m anomaly relative to the full-record mean — a quick visual
    for spotting unusually hot/cold months (heatwave screening), directly
    tied to the proposal's interest in rising heatwave frequency."""
    mean_t2m = df_c["t2m"].mean()
    anomaly = df_c["t2m"] - mean_t2m
    colors = ["#dc2626" if v >= 0 else "#2563eb" for v in anomaly]

    fig = go.Figure(go.Bar(
        x=df_c.index, y=anomaly, marker_color=colors, name="Anomaly",
    ))
    fig.add_hline(y=0, line_width=1, line_color="#64748b")
    fig.update_layout(
        title=f"Monthly Temperature Anomaly (vs. {mean_t2m:.1f}°C record mean) — {region_name}",
        template="plotly_white", height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis_title="Anomaly (°C)", xaxis_title="Month",
        showlegend=False,
    )
    return fig


def _build_native_map(ds, mask, var, time_value, region_name):
    """Native ERA5-resolution point map of `var` at `time_value`, restricted
    to grid cells inside `mask`."""
    bbox = _get_bbox(mask)
    if bbox is None:
        fig = go.Figure()
        fig.add_annotation(text="No grid cells in this region", xref="paper",
                            yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=420, template="plotly_white")
        return fig

    min_lat, min_lon, max_lat, max_lon = bbox
    da = ds[var].sel(time=time_value, method="nearest")[
        min_lat:max_lat + 1, min_lon:max_lon + 1
    ]
    sub_mask = mask[min_lat:max_lat + 1, min_lon:max_lon + 1]

    lat_vals = ds["latitude"].values[min_lat:max_lat + 1]
    lon_vals = ds["longitude"].values[min_lon:max_lon + 1]
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    values = da.values.astype(float)
    if np.nanmean(values) > 100:  # looks like Kelvin
        values = values - 273.15

    flat_lat = lat_grid[sub_mask]
    flat_lon = lon_grid[sub_mask]
    flat_val = values[sub_mask]

    fig = px.scatter_mapbox(
        lat=flat_lat, lon=flat_lon, color=flat_val,
        color_continuous_scale="RdBu_r",
        zoom=5, height=420,
        labels={"color": "°C"},
    )
    fig.update_traces(marker=dict(size=11, opacity=0.85))
    month_label = pd.Timestamp(time_value).strftime("%b %Y")
    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=45, b=0),
        title=f"t2m (native 0.1° grid) — {month_label} — {region_name}",
    )
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _metric_card(label, value, color="#0f172a"):
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Div(label, className="text-muted small text-uppercase", style={"letterSpacing": "0.5px"}),
            html.Div(value, style={"fontSize": "1.5rem", "fontWeight": 700, "color": color}),
        ]), className="shadow-sm text-center h-100"),
        md=3, xs=6, className="mb-3",
    )


def build_layout(region_name, geometry=None):
    """Builds the linked time-series + native-resolution-map layout.
    Call `register_callbacks(app)` once at startup for the click interaction
    to work."""
    ds = _load_dataset()
    if ds is None:
        return _empty_layout(
            f"No temperature data found. Expected file: {DATA_FILE}"
        )

    mask = get_data_mask(ds, geometry)
    df = get_masked_mean(ds, mask)

    if df.empty or "t2m" not in df.columns:
        return _empty_layout(f"No t2m variable found for {region_name}.")

    df_c = df.copy()
    for col in ("t2m", "d2m"):
        if col in df_c.columns:
            df_c[col] = df_c[col] - 273.15

    # ---- Quick-stat cards ----
    stats_children = [
        _metric_card("Latest Month", f"{df_c['t2m'].iloc[-1]:.1f} °C", "#ef4444"),
        _metric_card("2yr Average", f"{df_c['t2m'].mean():.1f} °C", "#0f172a"),
        _metric_card("Coolest Month", f"{df_c['t2m'].min():.1f} °C", "#1e40af"),
        _metric_card("Warmest Month", f"{df_c['t2m'].max():.1f} °C", "#dc2626"),
    ]
    stats_row = dbc.Row(stats_children, className="g-3 mb-2")

    fig_ts = _build_timeseries_figure(df_c, region_name)
    fig_map = _build_native_map(ds, mask, "t2m", df_c.index[-1], region_name)
    fig_clim = _build_climatology_figure(df_c, region_name)
    fig_anom = _build_anomaly_figure(df_c, region_name)

    store_data = {"region_name": region_name, "geometry": geometry}

    return html.Div([
        html.H4(f"Temperature Analysis — {region_name}", className="mb-3"),
        stats_row,
        dcc.Store(id="temp-panel-store", data=store_data),
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(id="temp-timeseries-graph", figure=fig_ts,
                              config={"displayModeBar": False})
                ), className="shadow-sm"),
                md=6,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(id="temp-map-graph", figure=fig_map,
                              config={"displayModeBar": False})
                ), className="shadow-sm"),
                md=6,
            ),
        ], className="g-4 mb-4"),
        html.P(
            "Click any point on the time series to update the map to that month.",
            className="text-muted text-center mb-4",
        ),
        dbc.Row([
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=fig_clim, config={"displayModeBar": False})
                ), className="shadow-sm"),
                md=6,
            ),
            dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=fig_anom, config={"displayModeBar": False})
                ), className="shadow-sm"),
                md=6,
            ),
        ], className="g-4"),
    ])


_callbacks_registered = False


def register_callbacks(app):
    """Wires up the click-on-timeseries -> update-map interaction.
    Call this exactly once, right after the Dash `app` object is created."""
    global _callbacks_registered
    if _callbacks_registered:
        return
    _callbacks_registered = True

    @app.callback(
        Output("temp-map-graph", "figure"),
        Input("temp-timeseries-graph", "clickData"),
        State("temp-panel-store", "data"),
        prevent_initial_call=True,
    )
    def _update_map_on_click(click_data, store_data):
        if not click_data or not store_data:
            return no_update
        ds = _load_dataset()
        if ds is None:
            return no_update
        geometry = store_data.get("geometry")
        region_name = store_data.get("region_name", "")
        mask = get_data_mask(ds, geometry)
        clicked_x = click_data["points"][0]["x"]
        time_value = pd.Timestamp(clicked_x)
        return _build_native_map(ds, mask, "t2m", time_value, region_name)