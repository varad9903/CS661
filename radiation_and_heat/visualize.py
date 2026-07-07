import plotly.graph_objects as go
import xarray as xr
from pathlib import Path

def render_state_view(state_name, geojson_geometry=None):
    """
    Template for teammates.
    Returns a Plotly Figure for radiation_and_heat.
    """
    # Create a placeholder figure
    fig = go.Figure()
    fig.add_annotation(
        text=f"Visualization for radiation_and_heat in {state_name} will go here.",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        title=f"Dashboard Panel: Radiation And Heat - {state_name}",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig
