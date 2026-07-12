import plotly.graph_objects as go
import plotly.express as px
import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MplPath
import dash_bootstrap_components as dbc
from dash import dcc, html

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
    """Returns a cropped DataArray for the animated heatmap."""
    # Get bounding box of mask
    valid_indices = np.argwhere(mask)
    if len(valid_indices) == 0:
        return None
        
    min_lat_idx, min_lon_idx = valid_indices.min(axis=0)
    max_lat_idx, max_lon_idx = valid_indices.max(axis=0)
    
    # slice the dataset
    da = ds[var_name][:, min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
    sub_mask = mask[min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
    
    # apply NaN to outside mask
    da = da.where(sub_mask)
    return da

def render_state_view(region_name, geometry=None):
    """
    Renders a multi-chart analytical dashboard for the Evaporation tab.
    """
    data_dir = Path("data")
    
    # Load required datasets
    try:
        def standardize_time(d):
            if 'valid_time' in d.dims:
                return d.rename({'valid_time': 'time'})
            return d
            
        ds_evap = standardize_time(xr.open_dataset(data_dir / "evaporation" / "data_stream-moda.nc"))
        ds_temp = standardize_time(xr.open_dataset(data_dir / "temperature" / "data_stream-moda.nc"))
        ds_soil = standardize_time(xr.open_dataset(data_dir / "soil_water" / "data_stream-moda.nc"))
        ds_rad = standardize_time(xr.open_dataset(data_dir / "heat_radiation" / "data_stream-moda.nc"))
    except Exception as e:
        return html.Div(f"Error loading datasets: {e}")

    # Process temporal means using a single pre-computed mask
    # We use ds_evap to compute the mask as all datasets share the same grid
    mask = get_data_mask(ds_evap, geometry)
    
    df_evap = get_masked_mean(ds_evap, mask)
    df_temp = get_masked_mean(ds_temp, mask)
    df_soil = get_masked_mean(ds_soil, mask)
    df_rad = get_masked_mean(ds_rad, mask)
    
    # Merge into one dataframe for cross-variable analysis
    df_all = pd.concat([df_evap, df_temp, df_soil, df_rad], axis=1)
    # Convert m to mm for readability
    for var in ['evabs', 'evatc', 'evavt', 'evaow', 'e']:
        if var in df_all:
            df_all[var] = df_all[var] * -1000  # Usually negative in ERA5, convert to positive mm
    
    # --- CHART 1: Stacked Area (Composition of Evaporation) ---
    fig_stack = go.Figure()
    colors = ['#f59e0b', '#10b981', '#3b82f6', '#06b6d4']
    names = ['Bare Soil (evabs)', 'Canopy (evatc)', 'Vegetation Transpiration (evavt)', 'Open Water (evaow)']
    cols = ['evabs', 'evatc', 'evavt', 'evaow']
    
    for i, col in enumerate(cols):
        if col in df_all.columns:
            fig_stack.add_trace(go.Scatter(
                x=df_all.index, y=df_all[col],
                mode='lines',
                line=dict(width=0.5, color=colors[i]),
                stackgroup='one',
                name=names[i]
            ))
    fig_stack.update_layout(
        title="Where does evaporated water come from? (Composition)",
        xaxis_title="Time", yaxis_title="Evaporation (mm/day)",
        template="plotly_white", margin=dict(t=50, b=20, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # --- CHART 2: Evap vs Soil Moisture Scatter (Feedback Loop) ---
    fig_scatter = go.Figure()
    if 'swvl1' in df_all.columns and 'evabs' in df_all.columns:
        fig_scatter = px.scatter(
            df_all.reset_index(), x='swvl1', y='evabs', color='t2m', 
            hover_data=['time'], color_continuous_scale='Inferno',
            title="Evaporation vs Soil Moisture (Feedback Loop)",
            labels={'swvl1': 'Volumetric Soil Water Layer 1 (m³/m³)', 'evabs': 'Bare Soil Evaporation (mm/day)', 't2m': 'Temp (K)'}
        )
    fig_scatter.update_layout(template="plotly_white", margin=dict(t=50, b=20, l=40, r=20))

    # --- CHART 3: Parallel Coordinates Plot ---
    fig_parallel = go.Figure()
    pc_cols = ['t2m', 'ssr', 'swvl1', 'e']
    if all(c in df_all.columns for c in pc_cols):
        fig_parallel = go.Figure(data=
            go.Parcoords(
                line=dict(color=df_all['e'], colorscale='Viridis', showscale=True, cmin=df_all['e'].min(), cmax=df_all['e'].max()),
                dimensions=[
                    dict(range=[df_all['t2m'].min(), df_all['t2m'].max()], label='Temperature (t2m)', values=df_all['t2m']),
                    dict(range=[df_all['ssr'].min(), df_all['ssr'].max()], label='Surface Net Solar Rad (ssr)', values=df_all['ssr']),
                    dict(range=[df_all['swvl1'].min(), df_all['swvl1'].max()], label='Soil Water (swvl1)', values=df_all['swvl1']),
                    dict(range=[df_all['e'].min(), df_all['e'].max()], label='Total Evap (e)', values=df_all['e']),
                ]
            )
        )
    fig_parallel.update_layout(
        title="Water-Energy Balance (Parallel Coordinates)",
        template="plotly_white", margin=dict(t=50, b=20, l=40, r=20)
    )

    # --- CHART 4: Correlation Heatmap ---
    corr_cols = ['t2m', 'ssr', 'swvl1', 'evabs', 'evaow', 'evatc', 'evavt', 'e']
    df_corr = df_all[[c for c in corr_cols if c in df_all.columns]].corr()
    fig_corr = px.imshow(
        df_corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Variable Correlation Matrix"
    )
    fig_corr.update_layout(template="plotly_white", margin=dict(t=50, b=20, l=40, r=20))

    # --- CHART 5: Animated Heatmap (Evaporation Intensity) ---
    fig_anim = go.Figure()
    da_e = get_cropped_3d(ds_evap, mask, 'e')
    if da_e is not None:
        # Downsample spatial resolution to keep browser animation lightweight
        da_e = da_e.coarsen(latitude=2, longitude=2, boundary='trim').mean()
        da_e_mm = da_e * -1000 # convert to positive mm
        
        # Convert DataArray to DataFrame for Plotly express
        df_anim = da_e_mm.to_dataframe(name='evap').reset_index()
        df_anim = df_anim.dropna() # remove out-of-mask nan pixels
        df_anim['month'] = df_anim['time'].dt.strftime('%Y-%m')
        
        fig_anim = px.density_mapbox(
            df_anim, lat='latitude', lon='longitude', z='evap', radius=15,
            animation_frame='month', center=dict(lat=da_e.latitude.mean().item(), lon=da_e.longitude.mean().item()),
            zoom=4.0, mapbox_style="carto-positron", color_continuous_scale="Blues",
            title="Spatial Evaporation Intensity (24-Month Animation, 0.2° grid)"
        )
    fig_anim.update_layout(margin=dict(t=50, b=20, l=20, r=20))

    # Common styling for all plots
    for f in [fig_stack, fig_scatter, fig_parallel, fig_corr]:
        f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#1e293b"))
    
    # Layout assembly
    layout = html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_stack, config={'displayModeBar': False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_scatter, config={'displayModeBar': False}), md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_anim, config={'displayModeBar': False, 'scrollZoom': True}, style={'height': '60vh'}), md=12),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_parallel, config={'displayModeBar': False}), md=12),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_corr, config={'displayModeBar': False}), md=6, className="mx-auto"),
        ], className="mb-4"),
        

    ])
    
    return layout
