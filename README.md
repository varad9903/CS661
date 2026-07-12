# India's Changing Climate: Analytical Dashboard

An interactive geospatial web dashboard built with Plotly Dash to analyze and visualize climate patterns across India using ERA5-Land NetCDF datasets. The app covers everything from monsoon wind directions to soil moisture feedback loops.

## Features

- **8 Specialized Modules:** The dashboard breaks down the climate into specific areas:
  - Temperature
  - Lakes
  - Snow
  - Soil Water
  - Heat & Radiation
  - Evaporation & Runoff
  - Vegetation
  - Wind, Pressure & Precipitation
- **Multi-Level Spatial Analysis:** You can analyze the data at three different levels:
  - **Entire India:** Macro-level view.
  - **State-Wise:** Click on any state on the map to drill down.
  - **Custom Region:** Use the Leaflet drawing tools to draw your own polygon on the map and analyze that specific area.
- **Advanced Visualizations:** 
  - SLIC Superpixel compression for rendering thousands of wind vectors without lagging the browser.
  - Animated Mapbox density heatmaps for spatial data over time.
  - Parallel coordinates to show water-energy balance.
  - Scatter plots and correlation matrices to prove climate feedback loops.
- **Fast Spatial Masking:** We use a custom Ray Casting algorithm to crop the global NetCDF grid to just the selected area in less than a second.

## Tech Stack

- **Frontend/Backend:** Plotly Dash (`dash`, `dash-bootstrap-components`)
- **Mapping:** `dash-leaflet` for drawing tools, `plotly.express` and `plotly.graph_objects` for charts
- **Data Processing:** `xarray` (for lazy loading NetCDF files), `pandas`, `numpy`, `scikit-image` (for SLIC clustering)

## Setup & Installation

1. **Clone the repo:**
   ```bash
   git clone <your-repo-url>
   cd <repository-name>
   ```

2. **Create a virtual environment (optional but good idea):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Data Setup

The app needs the ERA5-Land NetCDF (`.nc`) files inside a `data/` folder at the root of the project. We didn't include them in the repo because they are way too big. 

Make sure your `data/` directory looks exactly like this:

```text
data/
├── evaporation_and_runoff/
├── heat_radiation/
├── lakes/
├── snow/
├── soil_water/
├── temperature/
├── vegetation/
└── wind_pressure_and_precipitation/
```
(Just drop your `.nc` files into their respective folders)

## Running the App

Just run the main app file:

```bash
python app.py
```

Then open your browser and go to `http://127.0.0.1:8050/` to see the dashboard.
