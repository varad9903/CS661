# Climate Data Analytical Dashboard
An advanced, interactive geospatial web dashboard built with Plotly Dash to analyze and visualize the water-energy balance and climate feedback loops across India using ERA5 NetCDF datasets.
## Features
- **Multi-Level Spatial Analysis:** Analyze climate data at three distinct granularities:
  - **Entire India:** Macro-level analysis of the entire country.
  - **State-Wise:** Interactive choropleth selection to drill down into specific states.
  - **Custom Region (GIS Drawing):** Use the integrated Leaflet drawing tools to drop pins and analyze any custom polygon geometry on the fly.
- **Advanced Visualizations:** 
  - Composition of Evaporation (Stacked Area)
  - Evaporation vs Soil Moisture Feedback Loops (Scatter)
  - Variable Correlation Matrices (Heatmaps)
  - Water-Energy Balance Multi-dimensional Brushing (Parallel Coordinates)
  - Spatial Evaporation Intensity (Animated Mapbox Density)
- **High-Performance Spatial Masking:** Utilizes a custom Ray Casting (Winding Number) algorithm with bounding-box optimization and NetCDF lazy-loading to perform sub-second spatial cropping on millions of global grid cells without crashing browser memory.
## Tech Stack
- **Frontend/Backend:** Plotly Dash (`dash`, `dash-bootstrap-components`)
- **GIS/Mapping:** `dash-leaflet` for true spatial drawing, `plotly.express` mapping
- **Data Processing:** `xarray` (lazy NetCDF loading), `pandas`, `numpy`, `matplotlib.path`
## Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <repository-name>
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
## Data Directory Structure
The application expects ERA5 NetCDF (.nc) files to be located in a `data/` directory at the root of the project. Due to file sizes, these are not included in the repository and must be downloaded separately.
Ensure your `data/` directory looks like this:
```text
data/
├── evaporation/
│   ├── data_stream-moda.nc
│   └── data_stream-mnth.nc
├── heat_radiation/
├── lakes/
├── snow/
├── soil_water/
├── temperature/
├── vegetation/
└── wind_precipitation/
```
## Running the Application
Start the Dash development server:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:8501/` to use the dashboard!
