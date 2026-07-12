import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.path import Path as MplPath
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, clientside_callback, ClientsideFunction

ROOT_DIR = Path(__file__).resolve().parent.parent

# Friendly names for the two vegetation layers the dashboard can toggle between.
VEG_LABELS = {'lai_hv': 'High Vegetation', 'lai_lv': 'Low Vegetation'}

# Calendar-month labels in chronological order (ERA5-Land LAI is a year-invariant climatology,
# so the module presents a single 12-month cycle rather than the raw 24-month record).
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def get_data_mask(ds, geometry):
    """Returns a boolean numpy array mask for the dataset grid based on geometry."""
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
            path_mask = (points[:,0] >= extents.xmin) & (points[:,0] <= extents.xmax) & (points[:,1] >= extents.ymin) & (points[:,1] <= extents.ymax)
            sub_points = points[path_mask]
            if len(sub_points) > 0:
                sub_mask = path.contains_points(sub_points)
                mask[path_mask] |= sub_mask

    # if no geometry or parsing failed, just use all true
    if not mask.any():
        mask = np.ones(points.shape[0], dtype=bool)

    return mask.reshape(lons.shape)


def get_masked_mean(ds, mask):
    """Returns a pandas DataFrame of the spatial mean over time for the pre-computed mask."""
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
            # Slice the dataset spatially BEFORE calling .values to prevent loading the entire globe into RAM
            data = ds[var][:, min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1].values
            masked_data = np.where(sub_mask, data, np.nan)
            df[var] = np.nanmean(masked_data, axis=(1, 2))

    return df


def get_cropped_3d(ds, mask, var_name):
    """Returns a cropped, mask-applied DataArray (out-of-region cells as NaN) for spatial/animated plots."""
    valid_indices = np.argwhere(mask)
    if len(valid_indices) == 0:
        return None

    min_lat_idx, min_lon_idx = valid_indices.min(axis=0)
    max_lat_idx, max_lon_idx = valid_indices.max(axis=0)

    da = ds[var_name][:, min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
    sub_mask = mask[min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]

    return da.where(sub_mask)


def _standardize_time(ds):
    if 'valid_time' in ds.dims:
        return ds.rename({'valid_time': 'time'})
    return ds


def _build_veg_figs(veg_col, ds_veg, ds_soil, mask, df_all, da_t_c, da_s_c):
    """
    Builds the four vegetation-layer-dependent figures for a single LAI column
    ('lai_hv' or 'lai_lv'): the animated canopy-density map, the greening
    hysteresis loop, the root-zone response heatmap, and the climate niche.

    da_t_c / da_s_c are the pre-cropped, coarsened temperature and soil-moisture
    arrays shared across both layers (so the expensive climate crop happens once).
    Returns dict: {'anim', 'hyst', 'rootzone', 'niche'}.
    """
    label = VEG_LABELS.get(veg_col, veg_col)

    # Crop + coarsen the selected LAI layer once; the 24-month array feeds the niche, and a
    # calendar-month climatology derived from it feeds the animation.
    da_v = get_cropped_3d(ds_veg, mask, veg_col)
    if da_v is not None:
        da_v = da_v.coarsen(latitude=2, longitude=2, boundary='trim').mean()

    # --- Animated Spatial Heatmap of Canopy Density (12-month climatology) ---
    # LAI is year-invariant, so a 24-month animation would just repeat the same cycle twice.
    # Average each pixel over calendar month to a single 12-frame climatological cycle.
    fig_anim = go.Figure()
    if da_v is not None:
        da_clim = da_v.groupby('time.month').mean()
        df_anim = da_clim.to_dataframe(name='lai').reset_index().dropna()
        df_anim['month_name'] = df_anim['month'].map(lambda m: MONTHS[int(m) - 1])
        # Fix the color scale to the min/max across all 12 months so frames are
        # visually comparable (otherwise each frame would auto-scale independently).
        lai_min, lai_max = df_anim['lai'].min(), df_anim['lai'].max()
        fig_anim = px.density_mapbox(
            df_anim, lat='latitude', lon='longitude', z='lai', radius=15,
            animation_frame='month_name', category_orders={'month_name': MONTHS},
            center=dict(lat=da_v.latitude.mean().item(), lon=da_v.longitude.mean().item()),
            zoom=4.0, mapbox_style="carto-positron",
            color_continuous_scale="Greens", range_color=[lai_min, lai_max],
            title=f"Spatial Canopy Density (Monthly Climatology, {label} LAI)"
        )
        # range_color sets the coloraxis, but each frame's own trace defaults to
        # zauto - pin zmin/zmax explicitly on every frame + the base trace too.
        fig_anim.update_traces(zmin=lai_min, zmax=lai_max)
        for frame in fig_anim.frames:
            frame.data[0].update(zmin=lai_min, zmax=lai_max)
    fig_anim.update_layout(margin=dict(t=50, b=20, l=20, r=20))

    # --- Greening Hysteresis Loop (LAI vs 2m Temperature) ---
    # Points connected in chronological order reveal the seasonal loop: canopy
    # green-up and brown-down trace different paths against temperature (thermal lag).
    fig_hyst = go.Figure()
    if all(c in df_all.columns for c in [veg_col, 't2m']):
        df_h = df_all.reset_index()[['time', veg_col, 't2m']].dropna().sort_values('time')
        df_h['t2m_c'] = df_h['t2m'] - 273.15  # Kelvin -> Celsius for legibility
        df_h['month_num'] = df_h['time'].dt.month
        df_h['label'] = df_h['time'].dt.strftime('%b %Y')
        fig_hyst.add_trace(go.Scatter(
            x=df_h['t2m_c'], y=df_h[veg_col],
            mode='lines+markers',
            line=dict(color='rgba(100,116,139,0.45)', width=1),
            marker=dict(
                size=11, color=df_h['month_num'], colorscale='Turbo',
                showscale=True, cmin=1, cmax=12,
                colorbar=dict(title='Month', tickvals=[1, 4, 7, 10], ticktext=['Jan', 'Apr', 'Jul', 'Oct'])
            ),
            text=df_h['label'],
            hovertemplate='%{text}<br>Temp: %{x:.1f}°C<br>LAI: %{y:.2f}<extra></extra>'
        ))
    fig_hyst.update_layout(
        title=f"Greening Hysteresis: {label} LAI vs Temperature",
        xaxis_title="2m Temperature (°C)", yaxis_title=f"{label} LAI",
        template="plotly_white", margin=dict(t=50, b=20, l=40, r=20)
    )

    # --- Root-Zone Response Heatmap (LAI vs soil moisture by depth & lag) ---
    # For each soil depth layer and each lead time, correlate canopy LAI(t) with
    # soil moisture(t - lag). Reveals which root-zone depth and response delay the
    # vegetation actually tracks (uptake depth + vegetation memory).
    fig_rootzone = go.Figure()
    soil_layers = [
        ('swvl1', 'Layer 1 (0-7 cm)'), ('swvl2', 'Layer 2 (7-28 cm)'),
        ('swvl3', 'Layer 3 (28-100 cm)'), ('swvl4', 'Layer 4 (100-289 cm)')
    ]
    avail_layers = [(c, lbl) for c, lbl in soil_layers if c in df_all.columns]
    if veg_col in df_all.columns and avail_layers:
        lags = [0, 1, 2, 3]
        corr_grid = [
            [df_all[veg_col].corr(df_all[col].shift(lag)) for lag in lags]
            for col, _ in avail_layers
        ]
        fig_rootzone = px.imshow(
            np.array(corr_grid, dtype=float),
            x=[f"{l} mo" for l in lags],
            y=[lbl for _, lbl in avail_layers],
            text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title=f"Root-Zone Water Uptake: {label} LAI vs Soil Moisture (depth × lag)"
        )
        fig_rootzone.update_layout(xaxis_title="Soil moisture lead time", yaxis_title="Soil depth")
    fig_rootzone.update_layout(template="plotly_white", margin=dict(t=50, b=20, l=40, r=20))

    # --- Greenness Climate Niche (mean LAI over Temperature x Soil Moisture) ---
    # Uses the full per-pixel monthly grid (not just regional means) to bin the
    # temperature x soil-moisture plane and colour each cell by mean canopy LAI,
    # exposing the bivariate climate envelope in which vegetation is greenest.
    fig_niche = go.Figure()
    if da_v is not None and da_t_c is not None and da_s_c is not None:
        df_niche = pd.DataFrame({
            't2m': da_t_c.values.flatten() - 273.15,  # Kelvin -> Celsius
            'swvl1': da_s_c.values.flatten(),
            'lai': da_v.values.flatten(),
        }).dropna()
        if len(df_niche) > 0:
            fig_niche = px.density_heatmap(
                df_niche, x='t2m', y='swvl1', z='lai', histfunc='avg',
                nbinsx=25, nbinsy=25, color_continuous_scale='YlGn',
                title=f"Greenness Climate Niche: Mean {label} LAI by Temperature & Soil Moisture",
                labels={'t2m': '2m Temperature (°C)', 'swvl1': 'Soil Moisture L1 (m³/m³)', 'lai': 'Mean LAI'}
            )
    fig_niche.update_layout(template="plotly_white", margin=dict(t=50, b=20, l=40, r=20))

    for f in [fig_hyst, fig_rootzone, fig_niche]:
        f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#1e293b"))

    return {'anim': fig_anim, 'hyst': fig_hyst, 'rootzone': fig_rootzone, 'niche': fig_niche}


# Clientside toggle: swap the four vegetation-layer figures instantly from the
# pre-built store (no server round-trip / no data reload on toggle). The JS body
# lives in assets/veg_toggle.js (namespace 'vegToggle') so it is reliably served;
# registered once at import and binds whenever the tab's components appear.
clientside_callback(
    ClientsideFunction(namespace='vegToggle', function_name='swap'),
    Output('veg-fig-anim', 'figure'),
    Output('veg-fig-hyst', 'figure'),
    Output('veg-fig-rootzone', 'figure'),
    Output('veg-fig-niche', 'figure'),
    Input('veg-type-toggle', 'value'),
    State('veg-fig-store', 'data'),
    prevent_initial_call=True,
)


def render_state_view(state_name, geojson_geometry=None):
    """
    Renders a multi-chart analytical dashboard for the Vegetation tab, with a
    High/Low vegetation-layer toggle driving the layer-dependent charts.
    """
    try:
        ds_veg = _standardize_time(xr.open_dataset(ROOT_DIR / "data" / "vegetation" / "data_stream-moda.nc"))
        ds_temp = _standardize_time(xr.open_dataset(ROOT_DIR / "data" / "temperature" / "data_stream-moda.nc"))
        ds_soil = _standardize_time(xr.open_dataset(ROOT_DIR / "data" / "soil_water" / "data_stream-moda.nc"))
    except Exception as e:
        return html.Div(f"Error loading datasets: {e}")

    # Evaporation is used only by the canopy-evaporation chart; treat it as optional so the rest
    # of the tab still renders if the file is absent.
    try:
        ds_evap = _standardize_time(xr.open_dataset(ROOT_DIR / "data" / "evaporation" / "data_stream-moda.nc"))
    except Exception:
        ds_evap = None

    # Single mask (all three datasets share the same 0.1 deg global grid)
    mask = get_data_mask(ds_veg, geojson_geometry)

    df_veg = get_masked_mean(ds_veg, mask)
    df_temp = get_masked_mean(ds_temp, mask)
    df_soil = get_masked_mean(ds_soil, mask)

    df_all = pd.concat([df_veg, df_temp, df_soil], axis=1)

    # --- STATIC CHART A: Canopy Density vs Evaporation Pathways (climatology) ---
    # Links canopy density (total LAI) to the two evaporative fluxes the vegetation actually
    # drives -- transpiration (evavt) and canopy interception (evatc) -- as a 12-month climatology.
    # These two pathways have OPPOSITE seasonal timing: pure transpiration is energy/demand-limited
    # and peaks pre-monsoon, while interception is rainfall-limited and peaks in the monsoon. Only
    # their sum (the canopy's total evaporative output) tracks canopy density, which is why a naive
    # LAI-vs-transpiration plot is misleading. LAI is year-invariant, so we show a single 12-month
    # cycle; evaporation is averaged over calendar month for consistency.
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    if ds_evap is not None and any(c in df_all.columns for c in ['lai_hv', 'lai_lv']):
        veg_cols = [c for c in ['lai_hv', 'lai_lv'] if c in df_all.columns]
        df_e = get_masked_mean(ds_evap, mask)
        dfc = pd.concat([df_all[veg_cols], df_e], axis=1).reset_index()
        dfc['total_lai'] = dfc[veg_cols].sum(axis=1)
        dfc['month_num'] = dfc['time'].dt.month
        for col in ['evavt', 'evatc']:
            if col in dfc.columns:
                dfc[col] = dfc[col] * -1000.0  # m of water equivalent -> positive mm/day
        clim = dfc.groupby('month_num').mean(numeric_only=True).reset_index().sort_values('month_num')
        clim['month_name'] = clim['month_num'].map(lambda m: MONTHS[m - 1])

        # Secondary axis: stacked canopy-evaporation composition (transpiration + interception)
        if 'evavt' in clim.columns:
            fig_ts.add_trace(go.Scatter(
                x=clim['month_name'], y=clim['evavt'], name='Transpiration (evavt)',
                mode='lines', line=dict(width=0.5, color='#0891b2'),
                stackgroup='canopy', fillcolor='rgba(8,145,178,0.4)'
            ), secondary_y=True)
        if 'evatc' in clim.columns:
            fig_ts.add_trace(go.Scatter(
                x=clim['month_name'], y=clim['evatc'], name='Canopy interception (evatc)',
                mode='lines', line=dict(width=0.5, color='#38bdf8'),
                stackgroup='canopy', fillcolor='rgba(56,189,248,0.4)'
            ), secondary_y=True)
        # Primary axis: canopy density
        fig_ts.add_trace(go.Scatter(
            x=clim['month_name'], y=clim['total_lai'], name='Total LAI (canopy density)',
            mode='lines+markers', line=dict(color='#166534', width=3)
        ), secondary_y=False)

        if 'evavt' in clim.columns and 'evatc' in clim.columns:
            r = clim['total_lai'].corr(clim['evavt'] + clim['evatc'])
            fig_ts.add_annotation(
                xref='paper', yref='paper', x=0.02, y=0.98, showarrow=False,
                text=f'r(LAI, total canopy evap) = {r:.2f}',
                font=dict(size=12, color='#166534'), bgcolor='rgba(255,255,255,0.6)'
            )
        fig_ts.update_yaxes(title_text="Total Leaf Area Index (m² / m²)", secondary_y=False)
        fig_ts.update_yaxes(title_text="Canopy evaporation (mm / day)", secondary_y=True)
    else:
        fig_ts.add_annotation(
            text="Evaporation data unavailable for this region.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    fig_ts.update_layout(
        title=f"Canopy Density vs Evaporation Pathways - {state_name}",
        xaxis_title="Month",
        xaxis=dict(categoryorder='array', categoryarray=MONTHS),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- STATIC CHART B: Climatological Seasonal LAI Cycle (always shows both layers) ---
    # NOTE: ERA5-Land LAI is a prescribed monthly climatology (CHTESSEL reads the same 12 monthly
    # maps every year), so it carries essentially no interannual variability (2024 == 2025 to
    # within ~1e-5). We therefore show the mean seasonal cycle with whiskers spanning the
    # (negligible) year-to-year range, rather than a misleading 2024-vs-2025 comparison.
    fig_season = go.Figure()
    if all(c in df_all.columns for c in ['lai_hv', 'lai_lv']):
        dfc = df_all.reset_index()[['time', 'lai_hv', 'lai_lv']].copy()
        dfc['month_num'] = dfc['time'].dt.month
        dfc['month_name'] = dfc['time'].dt.strftime('%b')
        agg = dfc.groupby(['month_num', 'month_name']).agg(
            hv_mean=('lai_hv', 'mean'), hv_min=('lai_hv', 'min'), hv_max=('lai_hv', 'max'),
            lv_mean=('lai_lv', 'mean'), lv_min=('lai_lv', 'min'), lv_max=('lai_lv', 'max'),
        ).reset_index().sort_values('month_num')

        fig_season.add_trace(go.Bar(
            x=agg['month_name'], y=agg['hv_mean'], name='High Vegetation',
            marker_color='#166534',
            error_y=dict(type='data', symmetric=False,
                         array=(agg['hv_max'] - agg['hv_mean']).tolist(),
                         arrayminus=(agg['hv_mean'] - agg['hv_min']).tolist())
        ))
        fig_season.add_trace(go.Bar(
            x=agg['month_name'], y=agg['lv_mean'], name='Low Vegetation',
            marker_color='#84cc16',
            error_y=dict(type='data', symmetric=False,
                         array=(agg['lv_max'] - agg['lv_mean']).tolist(),
                         arrayminus=(agg['lv_mean'] - agg['lv_min']).tolist())
        ))
        fig_season.update_layout(barmode='group')
    fig_season.update_layout(
        title="Climatological Seasonal LAI Cycle (2024-2025 mean; whiskers = interannual range)",
        xaxis_title="Month", yaxis_title="Leaf Area Index (m² / m²)",
        template="plotly_white", margin=dict(t=50, b=20, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    for f in [fig_ts, fig_season]:
        f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#1e293b"))

    # Shared cropped + coarsened climate arrays for the niche (identical for both layers)
    da_t_c = get_cropped_3d(ds_temp, mask, 't2m')
    da_s_c = get_cropped_3d(ds_soil, mask, 'swvl1')
    if da_t_c is not None:
        da_t_c = da_t_c.coarsen(latitude=2, longitude=2, boundary='trim').mean()
    if da_s_c is not None:
        da_s_c = da_s_c.coarsen(latitude=2, longitude=2, boundary='trim').mean()

    # Build both layer-dependent figure sets once; the toggle swaps between them.
    figs_hv = _build_veg_figs('lai_hv', ds_veg, ds_soil, mask, df_all, da_t_c, da_s_c)
    figs_lv = _build_veg_figs('lai_lv', ds_veg, ds_soil, mask, df_all, da_t_c, da_s_c)

    store_data = {
        veg: {k: json.loads(fig.to_json()) for k, fig in figset.items()}
        for veg, figset in [('lai_hv', figs_hv), ('lai_lv', figs_lv)]
    }

    toggle = html.Div([
        html.Span("Vegetation layer:", style={'fontWeight': '600', 'marginRight': '12px'}),
        dcc.RadioItems(
            id='veg-type-toggle',
            options=[
                {'label': ' High vegetation', 'value': 'lai_hv'},
                {'label': ' Low vegetation', 'value': 'lai_lv'},
            ],
            value='lai_hv', inline=True,
            labelStyle={'marginRight': '20px', 'cursor': 'pointer'},
        ),
    ], className="mb-3", style={'display': 'flex', 'alignItems': 'center'})

    layout = html.Div([
        dcc.Store(id='veg-fig-store', data=store_data),
        toggle,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_ts, config={'displayModeBar': False}), md=6),
            dbc.Col(dcc.Graph(id='veg-fig-hyst', figure=figs_hv['hyst'], config={'displayModeBar': False}), md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='veg-fig-anim', figure=figs_hv['anim'], config={'displayModeBar': False}, style={'height': '60vh'}), md=12),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_season, config={'displayModeBar': False}), md=12),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id='veg-fig-niche', figure=figs_hv['niche'], config={'displayModeBar': False}), md=7),
            dbc.Col(dcc.Graph(id='veg-fig-rootzone', figure=figs_hv['rootzone'], config={'displayModeBar': False}), md=5),
        ], className="mb-4"),
    ])

    return layout
