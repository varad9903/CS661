import plotly.graph_objects as go
import plotly.express as px
import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MplPath
import dash_bootstrap_components as dbc
from dash import dcc, html
import json
import time as time_module


# ── Shared helpers (same algorithm as evaporation module) ──

def get_data_mask(ds, geometry):
    """Returns a boolean mask for the dataset grid based on geometry."""
    lons, lats = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
    points = np.vstack((lons.flatten(), lats.flatten())).T

    mask = np.zeros(points.shape[0], dtype=bool)
    if geometry and geometry.get('type') in ['Polygon', 'MultiPolygon']:
        paths = []
        if geometry['type'] == 'Polygon':
            paths.append(MplPath(geometry['coordinates'][0]))
        elif geometry['type'] == 'MultiPolygon':
            for poly in geometry['coordinates']:
                paths.append(MplPath(poly[0]))

        for path in paths:
            extents = path.get_extents()
            path_mask = (
                (points[:, 0] >= extents.xmin) & (points[:, 0] <= extents.xmax) &
                (points[:, 1] >= extents.ymin) & (points[:, 1] <= extents.ymax)
            )
            sub_points = points[path_mask]
            if len(sub_points) > 0:
                sub_mask = path.contains_points(sub_points)
                mask[path_mask] |= sub_mask

    if not mask.any():
        mask = np.ones(points.shape[0], dtype=bool)

    return mask.reshape(lons.shape)


def get_masked_mean(ds, mask):
    """Returns a DataFrame of the spatial mean over time for the pre-computed mask."""
    df = pd.DataFrame({'time': ds['time'].values})
    df.set_index('time', inplace=True)

    valid_indices = np.argwhere(mask)
    if len(valid_indices) == 0:
        return df

    min_lat_idx, min_lon_idx = valid_indices.min(axis=0)
    max_lat_idx, max_lon_idx = valid_indices.max(axis=0)
    sub_mask = mask[min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]

    for var in ds.data_vars:
        if len(ds[var].dims) >= 3 and 'time' in ds[var].dims:
            data = ds[var][:, min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1].values
            masked_data = np.where(sub_mask, data, np.nan)
            df[var] = np.nanmean(masked_data, axis=(1, 2))

    return df


# ── Wind Vector Animation ──

def create_wind_animation(ds, mask):
    """Create an animated wind vector field using line-segment arrows."""
    valid = np.argwhere(mask)
    r0, c0 = valid.min(axis=0)
    r1, c1 = valid.max(axis=0)

    sub_mask = mask[r0:r1+1, c0:c1+1]
    lats = ds['latitude'].values[r0:r1+1]
    lons = ds['longitude'].values[c0:c1+1]

    # Adaptive subsampling: target ~500-700 arrow points per frame
    height, width = sub_mask.shape
    total_bb = height * width
    subsample = max(2, int(np.sqrt(total_bb / 600)))

    ss_mask = sub_mask[::subsample, ::subsample]
    ss_lats = lats[::subsample]
    ss_lons = lons[::subsample]
    lon_grid, lat_grid = np.meshgrid(ss_lons, ss_lats)
    valid_cells = ss_mask.flatten()

    n_arrows = int(valid_cells.sum())

    # Pre-scan for global max speed (for consistent arrow scaling)
    global_max = 0.0
    for t in range(len(ds['time'])):
        u = ds['u10'][t, r0:r1+1, c0:c1+1].values[::subsample, ::subsample]
        v = ds['v10'][t, r0:r1+1, c0:c1+1].values[::subsample, ::subsample]
        sp = np.sqrt(u**2 + v**2)
        mx = float(np.nanmax(sp.flatten()[valid_cells]))
        if mx > global_max:
            global_max = mx

    scale = 0.8  # visual length of the longest arrow in degrees

    frames = []
    for t in range(len(ds['time'])):
        u = ds['u10'][t, r0:r1+1, c0:c1+1].values[::subsample, ::subsample]
        v = ds['v10'][t, r0:r1+1, c0:c1+1].values[::subsample, ::subsample]
        speed = np.sqrt(u**2 + v**2)

        lon_f = lon_grid.flatten()[valid_cells]
        lat_f = lat_grid.flatten()[valid_cells]
        u_f = np.nan_to_num(u.flatten()[valid_cells], nan=0.0)
        v_f = np.nan_to_num(v.flatten()[valid_cells], nan=0.0)
        speed_f = np.nan_to_num(speed.flatten()[valid_cells], nan=0.0)

        # Normalize arrows so longest = scale degrees
        u_norm = u_f / (global_max + 1e-10) * scale
        v_norm = v_f / (global_max + 1e-10) * scale

        # Build line segments: [base, tip, None] per arrow
        x_lines, y_lines = [], []
        for i in range(len(lon_f)):
            x_lines.extend([float(lon_f[i]), float(lon_f[i] + u_norm[i]), None])
            y_lines.extend([float(lat_f[i]), float(lat_f[i] + v_norm[i]), None])

        sizes = (speed_f / (global_max + 1e-10) * 6 + 3).tolist()
        month_str = str(ds['time'].values[t])[:7]

        frames.append(go.Frame(
            data=[
                # Arrow shafts
                go.Scatter(
                    x=x_lines, y=y_lines, mode='lines',
                    line=dict(color='rgba(59,130,246,0.5)', width=1.5),
                    showlegend=False, hoverinfo='skip'
                ),
                # Arrow base dots (colored by speed)
                go.Scatter(
                    x=lon_f.tolist(), y=lat_f.tolist(), mode='markers',
                    marker=dict(
                        size=sizes,
                        color=speed_f.tolist(),
                        colorscale='Turbo',
                        cmin=0, cmax=float(global_max),
                        showscale=True,
                        colorbar=dict(title='m/s', len=0.6),
                    ),
                    text=[f'Speed: {s:.1f} m/s' for s in speed_f],
                    hoverinfo='text', showlegend=False
                )
            ],
            name=month_str
        ))

    fig = go.Figure(data=frames[0].data, frames=frames)

    fig.update_layout(
        title=dict(text="Wind Direction & Speed", font=dict(size=16)),
        xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)",
        template="plotly_white",
        xaxis=dict(scaleanchor="y", scaleratio=1, constrain="domain"),
        yaxis=dict(constrain="domain"),
        margin=dict(t=50, b=60, l=50, r=20),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=-0.12, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            active=0,
            steps=[dict(
                args=[[f.name], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
                label=f.name, method="animate"
            ) for f in frames],
            x=0.05, len=0.9, y=-0.18,
            currentvalue=dict(prefix="Month: ", visible=True),
        )]
    )

    return fig, n_arrows


# ── SLIC Superpixel Visualization ──

def create_slic_visualization(geometry, slic_json_path):
    """Create SLIC superpixel wind visualization filtered to the selected region."""
    with open(slic_json_path, 'r') as f:
        slic_data = json.load(f)

    clusters = slic_data['clusters']
    time_values = slic_data['time_values']

    # Filter clusters whose centroids fall inside the selected geometry
    if geometry:
        paths = []
        if geometry['type'] == 'Polygon':
            paths.append(MplPath(geometry['coordinates'][0]))
        elif geometry['type'] == 'MultiPolygon':
            for poly in geometry['coordinates']:
                paths.append(MplPath(poly[0]))

        filtered_ids = []
        for cid, cdata in clusters.items():
            pt = [cdata['centroid_lon'], cdata['centroid_lat']]
            for path in paths:
                if path.contains_point(pt):
                    filtered_ids.append(cid)
                    break
    else:
        filtered_ids = list(clusters.keys())

    if not filtered_ids:
        fig = go.Figure()
        fig.add_annotation(text="No SLIC clusters found in this region.", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
        metrics = {
            'n_clusters': 0,
            'total_cells': 0,
            'gaussian_fits': 0,
            'gmm_fits': 0,
        }
        return fig, metrics

    n_months = len(time_values)
    scale = 0.3

    # Global max speed across all filtered clusters for consistent scaling
    all_speeds = [clusters[cid]['monthly'][t]['mean_ws']
                  for cid in filtered_ids for t in range(n_months)
                  if clusters[cid]['monthly'][t] is not None]
    g_max = max(all_speeds) if all_speeds else 1.0

    frames = []
    for t in range(n_months):
        lons_arr, lats_arr, speeds_arr = [], [], []
        u_arr, v_arr, hover_arr = [], [], []

        for cid in filtered_ids:
            c = clusters[cid]
            m = c['monthly'][t]
            if m is None:
                continue

            # Skip clusters whose data was entirely NaN for this month
            # (happens for clusters near coastlines where ERA5 has missing values)
            mean_ws = m['mean_ws']
            mean_u  = m['mean_u10']
            mean_v  = m['mean_v10']
            if mean_ws is None or mean_ws != mean_ws:   # NaN check (NaN != NaN is True)
                continue

            lons_arr.append(c['centroid_lon'])
            lats_arr.append(c['centroid_lat'])
            speeds_arr.append(mean_ws)
            u_arr.append(mean_u if (mean_u == mean_u) else 0.0)
            v_arr.append(mean_v if (mean_v == mean_v) else 0.0)

            if m['dist_type'] == 'gaussian':
                p = m['dist_params']
                p_mean = p['mean']
                p_std  = p['std']
                if p_mean != p_mean or p_std != p_std:  # NaN check
                    hover = (f"Cluster {cid} ({c['n_cells']} cells)<br>"
                             f"Speed: {mean_ws:.2f} m/s<br>"
                             f"Distribution: Gaussian<br>"
                             f"(insufficient data for fit this month)")
                else:
                    hover = (f"Cluster {cid} ({c['n_cells']} cells)<br>"
                             f"Speed: {mean_ws:.2f} m/s<br>"
                             f"Distribution: Gaussian<br>"
                             f"μ = {p_mean:.2f}, σ = {p_std:.2f}")
            else:
                p = m['dist_params']
                hover = (f"Cluster {cid} ({c['n_cells']} cells)<br>"
                         f"Speed: {mean_ws:.2f} m/s<br>"
                         f"Distribution: GMM (2 components)<br>"
                         f"Component 1: {p['weights'][0]:.0%} @ {p['means'][0]:.2f} m/s<br>"
                         f"Component 2: {p['weights'][1]:.0%} @ {p['means'][1]:.2f} m/s")
            hover_arr.append(hover)


        lons_np = np.array(lons_arr)
        lats_np = np.array(lats_arr)
        speeds_np = np.nan_to_num(np.array(speeds_arr), nan=0.0)
        u_np = np.nan_to_num(np.array(u_arr), nan=0.0)
        v_np = np.nan_to_num(np.array(v_arr), nan=0.0)

        u_norm = u_np / (g_max + 1e-10) * scale
        v_norm = v_np / (g_max + 1e-10) * scale

        x_lines, y_lines = [], []
        for i in range(len(lons_np)):
            x_lines.extend([float(lons_np[i]), float(lons_np[i] + u_norm[i]), None])
            y_lines.extend([float(lats_np[i]), float(lats_np[i] + v_norm[i]), None])

        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=x_lines, y=y_lines, mode='lines',
                    line=dict(color='rgba(16,185,129,0.7)', width=2),
                    showlegend=False, hoverinfo='skip'
                ),
                go.Scatter(
                    x=lons_np.tolist(), y=lats_np.tolist(), mode='markers',
                    marker=dict(
                        size=(speeds_np / (g_max + 1e-10) * 14 + 6).tolist(),
                        color=speeds_np.tolist(),
                        colorscale='Viridis',
                        cmin=0, cmax=float(g_max),
                        showscale=True,
                        colorbar=dict(title='m/s', len=0.6),
                        line=dict(width=1, color='white')
                    ),
                    text=hover_arr, hoverinfo='text', showlegend=False
                )
            ],
            name=time_values[t]
        ))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=dict(text="SLIC Superpixel Wind Summary (Hover for Distribution)", font=dict(size=16)),
        xaxis_title="Longitude (°E)", yaxis_title="Latitude (°N)",
        template="plotly_white",
        xaxis=dict(scaleanchor="y", scaleratio=1, constrain="domain"),
        yaxis=dict(constrain="domain"),
        margin=dict(t=50, b=60, l=50, r=20),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=-0.12, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            active=0,
            steps=[dict(
                args=[[f.name], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
                label=f.name, method="animate"
            ) for f in frames],
            x=0.05, len=0.9, y=-0.18,
            currentvalue=dict(prefix="Month: ", visible=True),
        )]
    )

    # Metrics
    total_gauss = sum(1 for cid in filtered_ids for m in clusters[cid]['monthly'] if m and m['dist_type'] == 'gaussian')
    total_gmm = sum(1 for cid in filtered_ids for m in clusters[cid]['monthly'] if m and m['dist_type'] == 'gmm')
    total_cells = sum(clusters[cid]['n_cells'] for cid in filtered_ids)

    metrics = {
        'n_clusters': len(filtered_ids),
        'total_cells': total_cells,
        'gaussian_fits': total_gauss,
        'gmm_fits': total_gmm,
    }

    return fig, metrics


# ── Main entry point ──

def render_state_view(region_name, geometry=None):
    """
    Renders the Wind & Precipitation dashboard.
    Returns an html.Div containing:
      - Animated wind vector field
      - Wind speed + precipitation time series
      - SLIC comparison (if pre-computed)
    """
    data_dir = Path("data")
    wind_file = data_dir / "wind_precipitation" / "data_stream-moda.nc"
    slic_npz = Path("wind_pressure_and_precipitation/slic_summary.npz")
    slic_json = Path("wind_pressure_and_precipitation/slic_clusters.json")
    slic_mnth_json = Path("wind_pressure_and_precipitation/slic_clusters_mnth.json")

    if not wind_file.exists():
        return html.Div(
            dbc.Alert("Wind data file not found. Ensure data/wind_precipitation/data_stream-moda.nc exists.", color="danger")
        )

    # Step 1: Load the NetCDF dataset
    try:
        ds = xr.open_dataset(wind_file)
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})
    except Exception as e:
        return html.Div(f"Error loading wind data: {e}")

    # Step 2: Process raw data and generate time series and animation plots
    t_raw_start = time_module.time()

    mask = get_data_mask(ds, geometry)

    # Time series data
    df = get_masked_mean(ds, mask)
    if 'u10' in df.columns and 'v10' in df.columns:
        df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
    if 'tp' in df.columns:
        df['tp_mm'] = df['tp'] * 1000  # m to mm

    # Wind speed line chart
    fig_speed = go.Figure()
    if 'wind_speed' in df.columns:
        fig_speed.add_trace(go.Scatter(
            x=df.index, y=df['wind_speed'], mode='lines+markers',
            name='Wind Speed', line=dict(color='#3b82f6', width=2.5),
            marker=dict(size=5)
        ))
    fig_speed.update_layout(
        title="Monthly Mean Wind Speed",
        xaxis_title="Time", yaxis_title="Wind Speed (m/s)",
        template="plotly_white", margin=dict(t=50, b=20, l=40, r=20)
    )

    # Precipitation bar chart
    fig_precip = go.Figure()
    if 'tp_mm' in df.columns:
        fig_precip.add_trace(go.Bar(
            x=df.index, y=df['tp_mm'],
            name='Precipitation', marker_color='#06b6d4'
        ))
    fig_precip.update_layout(
        title="Monthly Mean Precipitation",
        xaxis_title="Time", yaxis_title="Precipitation (mm)",
        template="plotly_white", margin=dict(t=50, b=20, l=40, r=20)
    )

    # Surface pressure line chart
    fig_sp = go.Figure()
    if 'sp' in df.columns:
        fig_sp.add_trace(go.Scatter(
            x=df.index, y=df['sp'] / 100,  # Pa to hPa
            mode='lines+markers', name='Surface Pressure',
            line=dict(color='#f59e0b', width=2.5), marker=dict(size=5)
        ))
    fig_sp.update_layout(
        title="Monthly Mean Surface Pressure",
        xaxis_title="Time", yaxis_title="Pressure (hPa)",
        template="plotly_white", margin=dict(t=50, b=20, l=40, r=20)
    )

    # Animated wind vector field
    fig_anim, n_arrows = create_wind_animation(ds, mask)

    t_raw = time_module.time() - t_raw_start

    # Step 3: Render SLIC section for Monthly Averaged Data (395 MB)
    slic_section = []
    if slic_npz.exists() and slic_json.exists():
        t_slic_start = time_module.time()
        fig_slic, metrics = create_slic_visualization(geometry, slic_json)
        t_slic = time_module.time() - t_slic_start

        json_kb = slic_json.stat().st_size / 1024
        speedup = t_raw / max(t_slic, 0.001)

        comparison_card = dbc.Card([
            dbc.CardHeader(html.H5("Raw vs SLIC Comparison (Monthly Avg)", className="mb-0")),
            dbc.CardBody([
                html.Table([
                    html.Thead(html.Tr([html.Th(""), html.Th("Raw Data"), html.Th("SLIC")])),
                    html.Tbody([
                        html.Tr([html.Td("Source"), html.Td("395 MB NetCDF"), html.Td(f"{json_kb:.0f} KB JSON")]),
                        html.Tr([html.Td("Load Time"), html.Td(f"{t_raw:.1f}s"), html.Td(f"{t_slic:.2f}s")]),
                        html.Tr([html.Td("Data Points"), html.Td(f"{n_arrows:,} arrows/mo"), html.Td(f"{metrics['n_clusters']} clusters")]),
                        html.Tr([html.Td("Gaussian Fits (24mo Total)"), html.Td("—"), html.Td(f"{metrics['gaussian_fits']}")]),
                        html.Tr([html.Td("GMM Fits (24mo Total)"), html.Td("—"), html.Td(f"{metrics['gmm_fits']}")]),
                        html.Tr([html.Td("Total Cells"), html.Td(f"{int(mask.sum()):,}"), html.Td(f"{metrics['total_cells']:,}")]),
                    ])
                ], className="table table-sm table-bordered", style={'width': '100%'}),
                html.Hr(),
                html.P(
                    f"SLIC is {speedup:.0f}x faster",
                    style={'fontWeight': 'bold', 'color': '#10b981', 'fontSize': '1.3em', 'textAlign': 'center'}
                )
            ])
        ], className="shadow-sm h-100")

        for f in [fig_slic]:
            f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter", color="#1e293b"))

        slic_section = [
            html.Hr(className="my-4"),
            html.H4("SLIC Data Summarization — Monthly Averaged (395 MB)", className="mb-3"),
            html.P("Each dot is a SLIC superpixel from the monthly-averaged dataset, hover to see its Gaussian or GMM distribution model.",
                   className="text-muted mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_slic, config={'displayModeBar': False}, style={'height': '55vh'}), md=8),
                dbc.Col(comparison_card, md=4, className="d-flex"),
            ], className="mb-4"),
        ]
    else:
        slic_section = [
            html.Hr(className="my-4"),
            dbc.Alert([
                html.H5("SLIC Pre-Computation (Monthly Averaged)"),
                html.P("To enable the SLIC comparison visualization, run the following command once:"),
                html.Code("python wind_pressure_and_precipitation/precompute_slic.py", className="d-block p-2 bg-light mb-2"),
                html.P("This processes the monthly-averaged wind data (~2–5 min) and saves a compact summary.",
                       className="mb-0")
            ], color="info", className="mt-4")
        ]

    # Step 4: Render SLIC section for Hourly Data (9.9 GB)
    slic_mnth_section = []
    if slic_mnth_json.exists():
        t_slic_mnth_start = time_module.time()
        fig_slic_mnth, metrics_mnth = create_slic_visualization(geometry, slic_mnth_json)
        t_slic_mnth = time_module.time() - t_slic_mnth_start

        mnth_json_kb = slic_mnth_json.stat().st_size / 1024
        mnth_source_mb = 9900  # data_stream-mnth.nc is ~9.9 GB
        mnth_compression = mnth_source_mb / (mnth_json_kb / 1024) if mnth_json_kb > 0 else 0

        comparison_card_mnth = dbc.Card([
            dbc.CardHeader(html.H5("Hourly Data Compression", className="mb-0")),
            dbc.CardBody([
                html.Table([
                    html.Thead(html.Tr([html.Th(""), html.Th("Raw Hourly"), html.Th("SLIC")])),
                    html.Tbody([
                        html.Tr([html.Td("Source"), html.Td(f"{mnth_source_mb:,} MB NetCDF"), html.Td(f"{mnth_json_kb:.0f} KB JSON")]),
                        html.Tr([html.Td("Time Steps"), html.Td("576 (24h × 24mo)"), html.Td("24 monthly summaries")]),
                        html.Tr([html.Td("Load Time"), html.Td("N/A (too large)"), html.Td(f"{t_slic_mnth:.2f}s")]),
                        html.Tr([html.Td("Data Points"), html.Td("6.5M cells × 576"), html.Td(f"{metrics_mnth['n_clusters']} clusters")]),
                        html.Tr([html.Td("Gaussian Fits (24mo Total)"), html.Td("—"), html.Td(f"{metrics_mnth['gaussian_fits']}")]),
                        html.Tr([html.Td("GMM Fits (24mo Total)"), html.Td("—"), html.Td(f"{metrics_mnth['gmm_fits']}")]),
                        html.Tr([html.Td("Total Cells"), html.Td("—"), html.Td(f"{metrics_mnth['total_cells']:,}")]),
                    ])
                ], className="table table-sm table-bordered", style={'width': '100%'}),
                html.Hr(),
                html.P(
                    f"{mnth_compression:,.0f}x Compression Ratio",
                    style={'fontWeight': 'bold', 'color': '#8b5cf6', 'fontSize': '1.3em', 'textAlign': 'center'}
                )
            ])
        ], className="shadow-sm h-100", style={'borderColor': '#8b5cf6'})

        fig_slic_mnth.update_layout(
            title=dict(text="SLIC Superpixel Wind Summary with Hourly Source (Hover for Distribution)", font=dict(size=16)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#1e293b")
        )

        slic_mnth_section = [
            html.Hr(className="my-4"),
            html.H4("SLIC Data Summarization — Hourly Data (9.9 GB)", className="mb-3"),
            html.P([
                "Same SLIC technique applied to the full hourly dataset (576 time steps, 9.9 GB). ",
                "Each cluster's distribution is fitted using all 24 hourly wind speed readings per month, ",
                "giving statistically richer Gaussian/GMM fits."
            ], className="text-muted mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_slic_mnth, config={'displayModeBar': False}, style={'height': '55vh'}), md=8),
                dbc.Col(comparison_card_mnth, md=4, className="d-flex"),
            ], className="mb-4"),
        ]
    else:
        slic_mnth_section = [
            html.Hr(className="my-4"),
            dbc.Alert([
                html.H5("SLIC Pre-Computation (Hourly Data — 9.9 GB)"),
                html.P("To enable the hourly SLIC visualization, run the following command once:"),
                html.Code("python wind_pressure_and_precipitation/precompute_slic_mnth.py", className="d-block p-2 bg-light mb-2"),
                html.P("This processes 9.9 GB of hourly wind data (~10–15 min) and saves a compact summary (~3 MB).",
                       className="mb-0")
            ], color="warning", className="mt-4")
        ]

    # Step 5: Apply common layout styling to all generated figures
    for f in [fig_speed, fig_precip, fig_sp, fig_anim]:
        f.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#1e293b")
        )

    # Step 6: Assemble the final dashboard layout
    layout = html.Div([
        # Row 1: Animated wind vector field
        html.H4("Animated Wind Direction and Speed", className="mb-2"),
        html.P("Each arrow shows the wind direction; color and size indicate speed. Use the slider or play button to animate across months.",
               className="text-muted mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_anim, config={'displayModeBar': False}, style={'height': '58vh'}), md=12),
        ], className="mb-4"),

        # Row 2: Time series charts
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_speed, config={'displayModeBar': False}), md=4),
            dbc.Col(dcc.Graph(figure=fig_precip, config={'displayModeBar': False}), md=4),
            dbc.Col(dcc.Graph(figure=fig_sp, config={'displayModeBar': False}), md=4),
        ], className="mb-4"),

        # Row 3: SLIC comparison — Monthly Averaged (conditional)
        *slic_section,

        # Row 4: SLIC comparison — Hourly Data (conditional)
        *slic_mnth_section
    ])

    return layout

