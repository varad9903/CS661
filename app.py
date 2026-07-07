import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_leaflet as dl
import json
from pathlib import Path

# Import modules
import temperature.visualize as temp_viz
import lakes.visualize as lakes_viz
import snow.visualize as snow_viz
import soil_water.visualize as soil_viz
import radiation_and_heat.visualize as rad_viz
import evaporation_and_runoff.visualize as evap_viz
import vegetation.visualize as veg_viz
import wind_pressure_and_precipitation.visualize as wind_viz

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "India's Changing Climate"

# Load GeoJSON
geojson_path = Path("assets/india_states.geojson")
if geojson_path.exists():
    with open(geojson_path, "r", encoding="utf-8") as f:
        india_geojson = json.load(f)
else:
    india_geojson = {"type": "FeatureCollection", "features": []}

def create_isolated_map(region_name, geometry):
    single_region_geojson = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": geometry, "properties": {"NAME_1": region_name}}]
    }
    
    coords = geometry.get('coordinates', [])
    def flatten(c):
        if isinstance(c[0], (float, int)): return [c]
        out = []
        for sub in c: out.extend(flatten(sub))
        return out
    
    flat = flatten(coords)
    if flat:
        lons = [p[0] for p in flat]
        lats = [p[1] for p in flat]
        center_lon = sum(lons)/len(lons)
        center_lat = sum(lats)/len(lats)
        lon_range = max(lons) - min(lons)
        lat_range = max(lats) - min(lats)
        max_range = max(lon_range, lat_range)
        zoom = 8 - (max_range * 0.15)
        zoom = max(4.5, min(zoom, 9))
    else:
        center_lat, center_lon, zoom = 22.0, 78.0, 5
        
    fig = go.Figure(go.Choroplethmap(
        geojson=single_region_geojson,
        locations=[region_name],
        featureidkey="properties.NAME_1",
        z=[1],
        colorscale=[[0, '#2563eb'], [1, '#2563eb']],
        showscale=False,
        marker_opacity=0.6,
        marker_line_width=2,
        marker_line_color='#ffffff',
        hoverinfo="none"
    ))
    
    fig.update_layout(
        map_style="carto-positron",
        map_zoom=zoom,
        map_center={"lat": center_lat, "lon": center_lon},
        margin={"r":0,"t":0,"l":0,"b":0},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# --- APP LAYOUT ---
app.layout = html.Div([
    
    # === MODAL ===
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Confirm Analysis")),
            dbc.ModalBody(id="state-confirm-body"),
            dbc.ModalFooter([
                dbc.Button("Yes, Analyze", id="btn-confirm-state", color="primary"),
                dbc.Button("Cancel", id="btn-cancel-state", className="ms-auto", color="secondary")
            ])
        ],
        id="modal-state-confirm",
        is_open=False,
        centered=True
    ),
    
    # === NATIONAL VIEW (Landing Page) ===
    html.Div(id="national-view", className="fade-in", children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("India's Changing Climate", className="text-center mt-5 text-neon"),
                    html.P("A Visual Analytics System (2024-2025)", className="text-center text-muted mb-4", style={"fontSize": "1.2rem"}),
                ], width=12)
            ]),
            
            # Mode Selection
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dcc.RadioItems(
                            id='analysis-mode',
                            options=[
                                {'label': ' Entire India', 'value': 'india'},
                                {'label': ' State-Wise (Click State)', 'value': 'state'},
                                {'label': ' Custom Region (Draw Polygon)', 'value': 'custom'}
                            ],
                            value='state',
                            inline=True,
                            labelStyle={'marginRight': '30px', 'fontWeight': '600', 'cursor': 'pointer'}
                        ),
                        className="glass-panel text-center mb-4 p-3"
                    )
                ], width=12, md=8, className="mx-auto")
            ]),
            
            # Dash Leaflet Maps Container
            dbc.Row([
                dbc.Col([
                    html.Div(
                        id="main-map-container",
                        className="glass-panel hide-draw-tools",
                        children=[
                            html.Div(id="custom-instruction", className="text-center text-muted mb-2 hidden", children="Use the polygon tool (pentagon icon) on the left of the map to drop pins and draw a region."),
                            html.Button("Analyze Custom Region", id="btn-analyze-custom", className="btn-glow mb-3 hidden", style={"display": "block", "margin": "0 auto"}),
                            html.Button("Analyze Entire India", id="btn-analyze-india", className="btn-glow mb-3 hidden", style={"display": "block", "margin": "0 auto"}),
                            
                            html.Div(
                                children=[
                                    dl.Map(
                                        id='leaflet-map',
                                        children=[
                                            dl.TileLayer(url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"),
                                            dl.GeoJSON(
                                                data=india_geojson,
                                                id="india-geojson",
                                                options=dict(style=dict(color="#2563eb", weight=1, opacity=0.8, fillOpacity=0.1)),
                                                hoverStyle=dict(weight=3, color="#1e293b", fillOpacity=0.3)
                                            ),
                                            dl.FeatureGroup([
                                                dl.EditControl(
                                                    id="edit-control", 
                                                    draw=dict(polygon=True, polyline=False, rectangle=False, circle=False, marker=False, circlemarker=False)
                                                )
                                            ])
                                        ],
                                        center=[22.0, 82.0],
                                        zoom=4.5,
                                        style={'width': '100%', 'height': '100%'}
                                    )
                                ],
                                style={'height': '60vh', 'borderRadius': '12px', 'overflow': 'hidden', 'position': 'relative', 'zIndex': 0}
                            )
                        ]
                    )
                ], width=12, md=10, className="mx-auto")
            ])
        ], fluid=True, className="pb-5")
    ]),

    # === DETAIL VIEW (SPA Routing) ===
    html.Div(id="detail-view", className="hidden", children=[
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Button("← Back to Landing Page", id="btn-back", className="btn-glow mt-4 mb-3", n_clicks=0)
                ], width=12)
            ]),
            
            # Top Section: Elongated Map (Plotly)
            dbc.Row([
                dbc.Col([
                    html.Div(className="glass-panel fade-in mb-4", children=[
                        html.H3(id="detail-title", className="text-neon text-center mb-3"),
                        dcc.Graph(id="isolated-detail-map", style={'height': '50vh', 'borderRadius': '12px', 'overflow': 'hidden'}, config={'displayModeBar': False})
                    ])
                ], width=12, md=8, className="mx-auto")
            ]),
            
            # Bottom Section: Full Width Tabs
            dbc.Row([
                dbc.Col([
                    html.Div(className="glass-panel fade-in mb-5", children=[
                        html.Div(id="detail-tabs-container")
                    ])
                ], width=12)
            ])
            
        ], fluid=True)
    ])
    
], style={"minHeight": "100vh"})


@callback(
    [Output("main-map-container", "className"),
     Output("custom-instruction", "className"),
     Output("btn-analyze-custom", "className"),
     Output("btn-analyze-india", "className")],
    Input("analysis-mode", "value")
)
def update_map_mode(mode):
    if mode == "custom":
        return "glass-panel", "text-center text-muted mb-2", "btn-glow mb-3", "hidden"
    elif mode == "india":
        return "glass-panel hide-draw-tools", "hidden", "hidden", "btn-glow mb-3"
    else:
        return "glass-panel hide-draw-tools", "hidden", "hidden", "hidden"


@callback(
    [Output("modal-state-confirm", "is_open"),
     Output("state-confirm-body", "children")],
    [Input("india-geojson", "clickData"),
     Input("btn-cancel-state", "n_clicks"),
     Input("btn-confirm-state", "n_clicks")],
    [State("modal-state-confirm", "is_open"),
     State("analysis-mode", "value")],
    prevent_initial_call=True
)
def toggle_modal(clickData, btn_cancel, btn_confirm, is_open, mode):
    triggered_id = ctx.triggered_id
    if triggered_id == "india-geojson" and mode == "state" and clickData:
        state_name = clickData['properties'].get('NAME_1', 'Selected State')
        return True, f"Do you want to continue with {state_name}?"
    if triggered_id in ["btn-cancel-state", "btn-confirm-state"]:
        return False, dash.no_update
    return dash.no_update, dash.no_update


@callback(
    [Output("national-view", "className"),
     Output("detail-view", "className"),
     Output("detail-title", "children"),
     Output("isolated-detail-map", "figure"),
     Output("detail-tabs-container", "children")],
    [Input("btn-confirm-state", "n_clicks"),
     Input("btn-analyze-custom", "n_clicks"),
     Input("btn-analyze-india", "n_clicks"),
     Input("btn-back", "n_clicks")],
    [State("india-geojson", "clickData"),
     State("edit-control", "geojson"),
     State("analysis-mode", "value")],
    prevent_initial_call=True
)
def handle_routing(btn_state, btn_custom, btn_india, btn_back, clickData, drawn_geojson, mode):
    triggered_id = ctx.triggered_id
    print("DEBUG handle_routing -> triggered_id:", triggered_id, "mode:", mode)
    
    # BACK BUTTON
    if triggered_id == "btn-back":
        return "fade-in", "hidden", dash.no_update, dash.no_update, dash.no_update
        
    region_name = ""
    geometry = None
    
    # INDIA MODE
    if triggered_id == "btn-analyze-india" and mode == "india":
        region_name = "Entire India"
        all_polys = []
        for feature in india_geojson.get('features', []):
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                all_polys.append(geom['coordinates'])
            elif geom['type'] == 'MultiPolygon':
                for poly in geom['coordinates']:
                    all_polys.append(poly)
                    
        geometry = {
            "type": "MultiPolygon",
            "coordinates": all_polys
        }
        
    # STATE MODE (from confirmation modal)
    elif triggered_id == "btn-confirm-state" and mode == "state" and clickData:
        region_name = clickData['properties'].get('NAME_1', 'Selected State')
        geometry = clickData['geometry']
                
    # CUSTOM MODE (Dash Leaflet EditControl geojson)
    elif triggered_id == "btn-analyze-custom" and mode == "custom":
        region_name = "Custom Region"
        if drawn_geojson and 'features' in drawn_geojson and len(drawn_geojson['features']) > 0:
            feature = drawn_geojson['features'][-1]
            geometry = feature['geometry']
                
        if geometry is None:
            # Nothing drawn, exit
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if not geometry:
        print("DEBUG handle_routing -> No geometry matched! returning no_update")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    print("DEBUG handle_routing -> proceeding with region_name:", region_name)
    # Generate isolated map
    detail_map_fig = create_isolated_map(region_name, geometry)
    
    # Generate visual modules
    fig_temp = temp_viz.render_state_view(region_name, geometry)
    evap_content = evap_viz.render_state_view(region_name, geometry)
    fig_soil = soil_viz.render_state_view(region_name, geometry)
    fig_rad = rad_viz.render_state_view(region_name, geometry)
    fig_lakes = lakes_viz.render_state_view(region_name, geometry)
    fig_snow = snow_viz.render_state_view(region_name, geometry)
    fig_veg = veg_viz.render_state_view(region_name, geometry)
    fig_wind = wind_viz.render_state_view(region_name, geometry)
    
    for f in [fig_temp, fig_soil, fig_rad, fig_lakes, fig_snow, fig_veg, fig_wind]:
        f.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color="#1e293b"))
    
    tabs = dbc.Tabs([
        dbc.Tab(dcc.Graph(figure=fig_temp, config={'displayModeBar': False}, style={'height': '50vh'}), label="Temperature"),
        dbc.Tab(evap_content, label="Evap & Runoff"),
        dbc.Tab(dcc.Graph(figure=fig_soil, config={'displayModeBar': False}, style={'height': '50vh'}), label="Soil Water"),
        dbc.Tab(dcc.Graph(figure=fig_rad, config={'displayModeBar': False}, style={'height': '50vh'}), label="Radiation"),
        dbc.Tab(dcc.Graph(figure=fig_lakes, config={'displayModeBar': False}, style={'height': '50vh'}), label="Lakes"),
        dbc.Tab(dcc.Graph(figure=fig_snow, config={'displayModeBar': False}, style={'height': '50vh'}), label="Snow"),
        dbc.Tab(dcc.Graph(figure=fig_veg, config={'displayModeBar': False}, style={'height': '50vh'}), label="Vegetation"),
        dbc.Tab(dcc.Graph(figure=fig_wind, config={'displayModeBar': False}, style={'height': '50vh'}), label="Wind"),
    ])
    
    return "hidden", "fade-in", region_name, detail_map_fig, tabs
    

if __name__ == '__main__':
    app.run(debug=True, port=8501)
