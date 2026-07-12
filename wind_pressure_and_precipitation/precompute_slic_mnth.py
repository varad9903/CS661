"""
SLIC Pre-Computation for Hourly Wind Data (data_stream-mnth.nc)
================================================================
Run this script ONCE before launching the app:
    python wind_pressure_and_precipitation/precompute_slic_mnth.py

It will:
1. Load the hourly wind data (data_stream-mnth.nc, 9.9 GB on disk)
   → 576 time steps: 24 hours × 24 months on a (1801 × 3600) global grid
2. Build an India mask using the exact GeoJSON state boundaries
3. Run SLIC clustering on the time-averaged wind speed field
4. Fit Gaussian or GMM distributions to each cluster (per month,
   using all 24 hourly values within that month for richer statistics)
5. Save results to slic_summary_mnth.npz + slic_clusters_mnth.json

After running this, the Wind & Precipitation tab will show an
additional "SLIC on Hourly Data" section comparing 9.9 GB → ~3 MB.
"""

import numpy as np
import xarray as xr
import json
from pathlib import Path
from matplotlib.path import Path as MplPath
from scipy import stats
from sklearn.mixture import GaussianMixture
from skimage.segmentation import slic
import time as time_module
import warnings
warnings.filterwarnings('ignore')


def build_india_mask(lats, lons, geojson_path):
    """
    Build a boolean mask for all of India using the GeoJSON state boundaries.
    Uses the exact same winding-number algorithm as the app's get_data_mask().
    """
    with open(geojson_path, 'r') as f:
        india_geojson = json.load(f)

    all_polys = []
    for feature in india_geojson.get('features', []):
        geom = feature['geometry']
        if geom['type'] == 'Polygon':
            all_polys.append(geom['coordinates'])
        elif geom['type'] == 'MultiPolygon':
            for poly in geom['coordinates']:
                all_polys.append(poly)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
    mask = np.zeros(points.shape[0], dtype=bool)

    print(f"  Testing {len(all_polys)} sub-polygons against {len(points):,} grid points...")
    for i, poly in enumerate(all_polys):
        path = MplPath(poly[0])
        ext = path.get_extents()
        box = (
            (points[:, 0] >= ext.xmin) & (points[:, 0] <= ext.xmax) &
            (points[:, 1] >= ext.ymin) & (points[:, 1] <= ext.ymax)
        )
        sub = points[box]
        if len(sub) > 0:
            mask[box] |= path.contains_points(sub)
        if (i + 1) % 100 == 0:
            print(f"    ... {i+1}/{len(all_polys)} sub-polygons done")

    return mask.reshape(lon_grid.shape)


def main():
    print("=" * 60)
    print("  SLIC Pre-Computation for HOURLY Wind Data (mnth)")
    print("=" * 60)
    t0 = time_module.time()

    data_path = Path("data/wind_precipitation/data_stream-mnth.nc")
    geojson_path = Path("assets/india_states.geojson")
    out_npz = Path("wind_pressure_and_precipitation/slic_summary_mnth.npz")
    out_json = Path("wind_pressure_and_precipitation/slic_clusters_mnth.json")

    if not data_path.exists():
        print(f"ERROR: {data_path} not found!")
        return
    if not geojson_path.exists():
        print(f"ERROR: {geojson_path} not found!")
        return

    # Step 1: Load dataset
    print("\n[1/5] Loading dataset...")
    ds = xr.open_dataset(data_path)
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})

    n_total_steps = len(ds['time'])
    all_lats = ds['latitude'].values
    all_lons = ds['longitude'].values
    print(f"  Grid: {len(all_lats)} x {len(all_lons)}, {n_total_steps} time steps")
    print(f"  Variables: {list(ds.data_vars)}")

    # Determine monthly grouping: 24 hourly snapshots per month, 24 months
    # Group time steps by their YYYY-MM string
    time_strs_all = [str(ds['time'].values[t])[:7] for t in range(n_total_steps)]
    unique_months = sorted(set(time_strs_all))
    n_months = len(unique_months)
    print(f"  Unique months: {n_months}")
    print(f"  Time steps per month: {n_total_steps // n_months}")

    # Build month→index mapping
    month_indices = {}
    for t, m in enumerate(time_strs_all):
        month_indices.setdefault(m, []).append(t)

    # Step 2: Build India mask
    print("\n[2/5] Building India mask (this takes ~20-30 seconds)...")
    t_mask = time_module.time()
    full_mask = build_india_mask(all_lats, all_lons, geojson_path)
    print(f"  Mask built in {time_module.time() - t_mask:.1f}s")

    # Crop to bounding box of India
    valid_idx = np.argwhere(full_mask)
    r0, c0 = valid_idx.min(axis=0)
    r1, c1 = valid_idx.max(axis=0)

    india_mask = full_mask[r0:r1+1, c0:c1+1]
    india_lats = all_lats[r0:r1+1]
    india_lons = all_lons[c0:c1+1]

    print(f"  Lat range: [{india_lats[-1]:.1f}, {india_lats[0]:.1f}]")
    print(f"  Lon range: [{india_lons[0]:.1f}, {india_lons[-1]:.1f}]")
    print(f"  Cropped grid: {india_mask.shape[0]} x {india_mask.shape[1]}")
    print(f"  Cells inside India: {india_mask.sum():,} out of {india_mask.size:,}")

    # Step 3: Compute time-averaged wind speed for SLIC input
    # Average across ALL 576 time steps for the most stable spatial structure
    print(f"\n[3/5] Computing time-averaged wind speed across {n_total_steps} hourly steps...")
    t_avg = time_module.time()
    speed_sum = np.zeros(india_mask.shape, dtype=np.float64)

    # Process in monthly chunks to keep RAM manageable
    for mi, month_key in enumerate(unique_months):
        t_indices = month_indices[month_key]
        for t in t_indices:
            u = ds['u10'][t, r0:r1+1, c0:c1+1].values
            v = ds['v10'][t, r0:r1+1, c0:c1+1].values
            speed_sum += np.sqrt(u**2 + v**2)
        if (mi + 1) % 6 == 0:
            print(f"    ... averaged {mi+1}/{n_months} months ({(mi+1)*len(t_indices)} time steps)")

    mean_speed = (speed_sum / n_total_steps).astype(np.float32)
    print(f"  Averaging done in {time_module.time() - t_avg:.1f}s")

    # Normalize to [0, 1] for SLIC
    ms_india = np.where(india_mask, mean_speed, np.nan)
    vmin, vmax = float(np.nanmin(ms_india)), float(np.nanmax(ms_india))
    ms_norm = (ms_india - vmin) / (vmax - vmin + 1e-10)
    ms_norm = np.nan_to_num(ms_norm, nan=0.0).astype(np.float64)

    print(f"  Mean wind speed range: {vmin:.2f} - {vmax:.2f} m/s")

    # Step 4: Run SLIC
    n_segments = 400
    print(f"\n[4/5] Running SLIC (n_segments={n_segments}, compactness=0.1)...")
    t_slic = time_module.time()
    label_map = slic(
        ms_norm,
        n_segments=n_segments,
        compactness=0.1,
        enforce_connectivity=True,
        mask=india_mask,
        start_label=0,
        channel_axis=None
    )
    print(f"  SLIC completed in {time_module.time() - t_slic:.1f}s")

    unique_labels = sorted([int(l) for l in np.unique(label_map) if l >= 0])
    print(f"  Clusters produced: {len(unique_labels)}")

    # Step 5: Fit distributions per cluster per month
    # Key difference from moda: each cluster gets ALL hourly values within
    # that month (24 hours × N cells), giving much richer distributions.
    print(f"\n[5/5] Fitting distributions ({len(unique_labels)} clusters x {n_months} months)...")
    print(f"       Each cluster gets ~24 hourly readings per cell per month")
    t_dist = time_module.time()

    # Pre-compute cluster cell masks
    cluster_masks = {}
    for label_id in unique_labels:
        cmask = (label_map == label_id)
        if cmask.sum() < 3:
            continue
        cluster_masks[label_id] = cmask

    # Initialize cluster metadata
    clusters = {}
    for label_id, cmask in cluster_masks.items():
        cell_idx = np.argwhere(cmask)
        clusters[str(label_id)] = {
            'centroid_lat': float(india_lats[cell_idx[:, 0]].mean()),
            'centroid_lon': float(india_lons[cell_idx[:, 1]].mean()),
            'n_cells': int(cmask.sum()),
            'monthly': [None] * n_months
        }

    # Process month by month
    for mi, month_key in enumerate(unique_months):
        t_indices = month_indices[month_key]
        n_hours = len(t_indices)

        # Load all hourly data for this month's India crop
        # Stack all hourly readings: shape (n_hours, height, width)
        u_stack = np.stack([ds['u10'][t, r0:r1+1, c0:c1+1].values for t in t_indices])
        v_stack = np.stack([ds['v10'][t, r0:r1+1, c0:c1+1].values for t in t_indices])
        tp_stack = np.stack([ds['tp'][t, r0:r1+1, c0:c1+1].values for t in t_indices])
        sp_stack = np.stack([ds['sp'][t, r0:r1+1, c0:c1+1].values for t in t_indices])
        ws_stack = np.sqrt(u_stack**2 + v_stack**2)

        for label_id, cmask in cluster_masks.items():
            key = str(label_id)

            # Extract all hourly values for all cells in this cluster
            # cmask shape: (height, width), broadcast across time
            # Result: 1D array of (n_hours × n_cells_in_cluster) values
            u_c = u_stack[:, cmask].flatten()
            v_c = v_stack[:, cmask].flatten()
            ws_c = ws_stack[:, cmask].flatten()
            tp_c = tp_stack[:, cmask].flatten()
            sp_c = sp_stack[:, cmask].flatten()

            # Remove NaN values
            valid = np.isfinite(ws_c)
            ws_valid = ws_c[valid]

            # Normality test (Shapiro-Wilk) — sample up to 5000
            if len(ws_valid) >= 8:
                sample = ws_valid[:5000] if len(ws_valid) > 5000 else ws_valid
                _, p_value = stats.shapiro(sample)
                is_normal = p_value > 0.05
            else:
                is_normal = True

            if is_normal:
                dist_type = 'gaussian'
                dist_params = {
                    'mean': float(np.nanmean(ws_c)),
                    'std': float(np.nanstd(ws_c))
                }
            else:
                try:
                    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=50)
                    gmm.fit(ws_valid.reshape(-1, 1))
                    dist_type = 'gmm'
                    dist_params = {
                        'weights': [float(w) for w in gmm.weights_],
                        'means': [float(m) for m in gmm.means_.flatten()],
                        'covs': [float(c) for c in gmm.covariances_.flatten()]
                    }
                except Exception:
                    dist_type = 'gaussian'
                    dist_params = {
                        'mean': float(np.nanmean(ws_c)),
                        'std': float(np.nanstd(ws_c))
                    }

            clusters[key]['monthly'][mi] = {
                'mean_u10': float(np.nanmean(u_c)),
                'mean_v10': float(np.nanmean(v_c)),
                'mean_ws': float(np.nanmean(ws_c)),
                'mean_tp': float(np.nanmean(tp_c)),
                'mean_sp': float(np.nanmean(sp_c)),
                'n_hourly_samples': int(len(ws_valid)),
                'dist_type': dist_type,
                'dist_params': dist_params
            }

        if (mi + 1) % 6 == 0:
            print(f"  Month {mi+1}/{n_months} ({month_key}) done")

    n_gauss = sum(1 for c in clusters.values() for m in c['monthly'] if m and m['dist_type'] == 'gaussian')
    n_gmm = sum(1 for c in clusters.values() for m in c['monthly'] if m and m['dist_type'] == 'gmm')
    print(f"  Fitting done in {time_module.time() - t_dist:.1f}s")
    print(f"  Total fits: {n_gauss} Gaussian, {n_gmm} GMM")

    # Step 6: Save results
    print("\nSaving results...")
    np.savez_compressed(
        out_npz,
        label_map=label_map.astype(np.int16),
        india_mask=india_mask,
        india_lats=india_lats,
        india_lons=india_lons,
        bbox=np.array([r0, r1, c0, c1])
    )

    with open(out_json, 'w') as f:
        json.dump({
            'time_values': unique_months,
            'n_segments_requested': n_segments,
            'n_clusters_actual': len(clusters),
            'source_file': 'data_stream-mnth.nc',
            'source_size_mb': round(data_path.stat().st_size / (1024 * 1024), 1),
            'total_time_steps': n_total_steps,
            'hours_per_month': n_total_steps // n_months,
            'clusters': clusters
        }, f)

    npz_mb = out_npz.stat().st_size / (1024 * 1024)
    json_mb = out_json.stat().st_size / (1024 * 1024)
    total_mb = npz_mb + json_mb
    source_mb = data_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"  DONE in {time_module.time() - t0:.1f}s")
    print(f"  {out_npz.name}: {npz_mb:.1f} MB")
    print(f"  {out_json.name}: {json_mb:.1f} MB")
    print(f"  Total output: {total_mb:.1f} MB")
    print(f"  Source: {source_mb:.0f} MB → {total_mb:.1f} MB = {source_mb/total_mb:.0f}x compression")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
