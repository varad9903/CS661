# How Data Filtering Actually Works — Accurate, Step-by-Step

Let's walk through the entire pipeline, starting from "user clicked Maharashtra" all the way to "charts are rendered."
Everything here is verified directly against the source code.

---

## The Setup: What Do We Have?

**The NetCDF File (your data):** A file on your hard drive that stores climate values for the entire globe.
It is organized as a 3D cube:
```
Axis 1 → Time     (24 slices: Jan 2024, Feb 2024, ..., Dec 2025)
Axis 2 → Latitude (1801 rows: from +90° North to -90° South, spaced 0.1° apart)
Axis 3 → Longitude(3600 cols: from 0° to 359.9° East, spaced 0.1° apart)

Total grid cells = 1801 × 3600 = 6,483,600 cells per time step.
Total data = 24 × 6.5 million = ~156 million numbers per variable (e.g., evaporation).
```

**The GeoJSON File (your map):** A text file (`india_states.geojson`) that describes the borders of every Indian state as a list of (longitude, latitude) coordinate pairs. Maharashtra's entry in this file looks roughly like:
```json
{
  "type": "MultiPolygon",
  "coordinates": [
    [[[72.6, 20.7], [72.7, 20.8], [73.0, 21.2], ... , [72.6, 20.7]]],   ← main landmass
    [[[72.8, 19.1], [72.9, 19.2], ... , [72.8, 19.1]]],                  ← small island 1
    ...100+ more sub-polygons for tiny islands and enclaves...
  ]
}
```

> [!IMPORTANT]
> A `MultiPolygon` is just a collection of multiple separate `Polygon` shapes treated as one unit. Maharashtra isn't one solid blob — it has tiny offshore islands, and the GeoJSON has a separate polygon for each of them.

---

## PHASE 1: Getting the Polygon

### For State Mode (e.g., Maharashtra):
When you click on the map, `dash-leaflet` looks at its loaded GeoJSON and finds which state your cursor was inside. It hands our Python callback the full `geometry` dictionary from the GeoJSON file — the real surveyed border coordinates with hundreds of vertices defining Maharashtra's exact shape.

### For Custom Region (Draw Your Own):
When you drop pins on the map and click "Analyze", the Leaflet drawing tool gives us a GeoJSON `Polygon` containing only the coordinates of the pins you dropped (e.g., 6 points). The coordinates are in real-world Lat/Lon, anchored to Earth — not screen pixels.

### For Entire India:
We loop through every state in the GeoJSON and merge all their polygon coordinates into one giant `MultiPolygon`. This represents the full outline of India.

In all three cases, we end up with a `geometry` object. After this point, the algorithm is **identical**.

---

## PHASE 2: Building the Mask (`get_data_mask`)

*Code: lines 11–37 of `evaporation_and_runoff/visualize.py`*

We need to figure out which of the 6.5 million global grid cells are "inside" our selected region. The result of this phase is called a **mask** — a 2D boolean grid (True/False) of shape (1801 × 3600) that matches the NetCDF layout:

```
Mask (1801 × 3600):

F F F F F F F F F F F F F F F F F F F F F F F F F F F  ← Pacific Ocean (all False)
F F F F F F F F F F F F F F F F F F F F F F F F F F F
F F F F F F F F F F F F F T T T T F F F F F F F F F F  ← T = inside Maharashtra
F F F F F F F F F F F F T T T T T T F F F F F F F F F
F F F F F F F F F F F T T T T T T T T F F F F F F F F
F F F F F F F F F F F F F F F F F F F F F F F F F F F
```

### Step 2a: Create a flat list of all grid cell centers

```python
lons, lats = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
points = np.vstack((lons.flatten(), lats.flatten())).T
# points.shape = (6,483,600 × 2)
# Each row: [longitude_of_center, latitude_of_center] of one grid cell
```

Imagine this as printing the GPS coordinate of the center of every single square on the graph paper, in one giant Excel table with 6.5 million rows and 2 columns.

### Step 2b: Loop over each sub-polygon (this is the key step the previous doc got wrong)

Maharashtra is a `MultiPolygon` with ~100 sub-polygons (the main state + islands). The algorithm does **not** do one single global chop. It processes them **one sub-polygon at a time, in a loop:**

```python
for path in paths:  # ← loops ~100 times for Maharashtra
    ...
```

For each sub-polygon, two things happen:

**2b-i: Bounding Box Pre-Filter**

```python
extents = path.get_extents()
# extents: xmin=72.6, xmax=80.9, ymin=15.6, ymax=22.0  (for main Maharashtra landmass)

path_mask = (
    (points[:,0] >= extents.xmin) &   # lon >= 72.6
    (points[:,0] <= extents.xmax) &   # lon <= 80.9
    (points[:,1] >= extents.ymin) &   # lat >= 15.6
    (points[:,1] <= extents.ymax)     # lat <= 22.0
)
sub_points = points[path_mask]  # Only ~50,000 points survive
```

```
Global grid (6.5M points):          After bbox filter (~50K points):

. . . . . . . . . . . . . . .       . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . .       . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . .       . . . . . ┌─────────────┐ . .
. . . . . . . . . . . . . . .  →    . . . . . │ . . . . . . │ . .
. . . . . . . . . . . . . . .       . . . . . │ . . . . . . │ . .
. . . . . . . . . . . . . . .       . . . . . └─────────────┘ . .
. . . . . . . . . . . . . . .       . . . . . . . . . . . . . . .
```

This check is pure arithmetic — no fancy math. Insantaneous for a computer.

**2b-ii: Winding Number Test (`contains_points`)**

> [!NOTE]
> The previous docs called this "Ray Casting with Even/Odd rule." That was **slightly inaccurate**. Matplotlib's `contains_points` actually uses the **Winding Number algorithm**. For normal, non-self-intersecting polygon shapes (like state borders), both algorithms give the exact same answer, so practically it makes no difference — but this is what the code is actually running.

**What is the Winding Number?**

Imagine you are standing at the grid cell point you want to test. You are looking at the polygon (Maharashtra's border). Someone walks the entire perimeter of Maharashtra — they walk the full border and come back to the start. The question is: **how many times did they "wind" around you?**

- If the walker wound around you **1 time** (or any non-zero number), you are **inside**.
- If the walker's path cancelled itself out and the net winding is **0**, you are **outside**.

```
INSIDE (Nagpur):                    OUTSIDE (Arabian Sea):

       ←←←←←←←←←←←                         ←←←←←←←←←←←
       ↓             ↑                        ↓             ↑
       ↓    [YOU]    ↑     →                  ↓             ↑
       ↓             ↑   (winding=1)          ↓             ↑     →
       →→→→→→→→→→→                  [YOU]   →→→→→→→→→→→   (winding=0, 
                                                             arrows cancel)
```

For every one of the ~50K candidate points, this computation runs. The result is a boolean array of which points are truly inside this sub-polygon.

```python
sub_mask = path.contains_points(sub_points)  # True/False for each of the ~50K
mask[path_mask] |= sub_mask                  # Merge back into the global mask
```

The `|=` means "OR" — if a grid cell was already marked True by a previous sub-polygon (a previous island), it stays True.

After the loop finishes all sub-polygons, the final `mask` is a (1801 × 3600) boolean grid where every `True` cell is confirmed to be inside Maharashtra.

---

## PHASE 3: Extracting the Data (`get_masked_mean`)

*Code: lines 39–59 of `evaporation_and_runoff/visualize.py`*

Now we have the mask. The goal is to produce a pandas DataFrame with 24 rows (one per month) and one column per climate variable, containing the **spatial average over Maharashtra** for each month.

### Step 3a: Find the bounding box of the mask (second time — different purpose)

```python
valid_indices = np.argwhere(mask)               # All (row, col) where mask is True
min_lat_idx, min_lon_idx = valid_indices.min(axis=0)  # top-left corner
max_lat_idx, max_lon_idx = valid_indices.max(axis=0)  # bottom-right corner
sub_mask = mask[min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
```

This gives us a small rectangular window into the mask — the minimal bounding rectangle that contains all `True` cells. For Maharashtra, this might be rows 780–844, columns 726–809.

### Step 3b: Lazy-load only the bounding box rectangle from disk

> [!IMPORTANT]
> This is where the previous doc was also inaccurate. It said "the computer loads only the individual Keep squares." That's not how NetCDF works. NetCDF supports efficient loading of **rectangular slices**, not arbitrary scattered individual cells.
>
> So what we actually do: load the full small rectangle (bounding box) from disk — not the individual scattered True cells. This is still dramatically faster than loading the whole globe.

```python
data = ds[var][:, min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1].values
# Shape: (24 months, ~64 rows, ~83 cols) — a tiny 3D cube just for Maharashtra's bounding box
# RAM used: ~1 MB instead of ~600 MB
```

Think of it as: instead of loading the entire world's newspaper, you cut out the Maharashtra-shaped rectangle and only scan that.

### Step 3c: NaN the cells outside the actual polygon boundary

The loaded rectangle still includes cells from outside Maharashtra (like Goa or the Arabian Sea that happen to be in the rectangle). We now use our precise mask to zero those out:

```python
masked_data = np.where(sub_mask, data, np.nan)
#  ↑ For every cell: if sub_mask is True (inside Maharashtra) → keep the value
#                    if sub_mask is False (outside Maharashtra)→ replace with NaN
```

```
Loaded rectangular data:        After applying mask (NaN = outside):

  22  21  20  19  18  17          22  21  NaN NaN NaN NaN
  23  22  21  20  19  18    →     23  22  21  NaN NaN NaN
  24  23  22  21  20  19          24  23  22  21  NaN NaN
  25  24  23  22  21  20          NaN NaN 23  22  21  NaN
```

### Step 3d: Compute the spatial mean

```python
df[var] = np.nanmean(masked_data, axis=(1, 2))
# axis=(1,2) → average over all lat rows and lon cols
# NaN cells are automatically excluded from the average
# Result: a 1D array of 24 values — one average per month
```

This gives us: the average evaporation (or temperature, or soil water...) across all grid cells inside Maharashtra, for each of the 24 months. This is what gets plotted on every chart!

### Step 3e: The Silent Variable Filter (Good to Know!)

Before Step 3d runs, there is a gate that silently decides which variables even get processed. From line 53 of the code:

```python
if len(ds[var].dims) >= 3 and 'time' in ds[var].dims:
```

This means a variable is only processed if **both** of these are true:
1. It has **3 or more dimensions** (e.g., time × latitude × longitude)
2. One of those dimensions is specifically named **`time`**

**Why does this matter?** Some NetCDF files contain static, time-independent variables — for example, a land-sea mask (which grid cells are land vs. ocean) or terrain elevation. These are stored as 2D arrays (just latitude × longitude, no time axis). The condition above silently skips them — they never appear in the output DataFrame and therefore never show up in any chart.

> [!NOTE]
> This is not a bug — it is intentional. We only want time-varying climate signals for our charts, not static geographic constants. But if you ever add a new variable to the dataset and wonder why it is not showing up in the dashboard, check whether it has at least 3 dimensions with a `time` axis. If it is a 2D static field, it will be silently ignored here.

---

## Complete Verified Flow Diagram

```
User selects Maharashtra / draws polygon / picks Entire India
                    ↓
     Polygon geometry extracted (GeoJSON coordinates)
                    ↓
     ┌─────────────────────────────────────────────┐
     │           PHASE 2: get_data_mask()          │
     │                                             │
     │  Build list of 6.5M global grid centers     │
     │                    ↓                        │
     │  FOR EACH sub-polygon (loop ~100 times):    │
     │    1. Bounding box filter: 6.5M → ~50K      │
     │    2. Winding number test: ~50K → T/F       │
     │    3. Merge results into global mask (OR)   │
     │                                             │
     │  Final mask: (1801 × 3600) True/False grid  │
     └─────────────────────────────────────────────┘
                    ↓
     ┌─────────────────────────────────────────────┐
     │         PHASE 3: get_masked_mean()          │
     │                                             │
     │  Find bounding box of True cells in mask    │
     │                    ↓                        │
     │  Lazy-load rectangular slice from NetCDF    │
     │  (~1 MB instead of 600 MB)                  │
     │                    ↓                        │
     │  Apply mask: set outside cells → NaN        │
     │                    ↓                        │
     │  nanmean over lat & lon → 24 monthly values │
     └─────────────────────────────────────────────┘
                    ↓
          DataFrame (24 rows × N variables)
                    ↓
            5 charts rendered!
```

---

## Quick Reference: 3 Things the Previous Doc Got Wrong

| Previous (Inaccurate) | Actual Code Behaviour |
|---|---|
| One single global bounding box chop | Bounding box chop runs once **per sub-polygon** inside a loop |
| Code cherry-picks individual "Keep" grid cells from disk | Code loads a **rectangular bounding box slice** from disk, then NaN-masks the outsiders |
| Matplotlib uses Ray Casting (Even/Odd) | Matplotlib uses the **Winding Number** algorithm |
