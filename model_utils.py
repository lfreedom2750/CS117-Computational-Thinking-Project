import numpy as np
from scipy.ndimage import sobel
import lightgbm as lgb
from matplotlib.path import Path
import plotly.graph_objects as go

# ========================
# POLYGON → MASK
# ========================
def polygon_to_mask(H, W, polygon):
    Y, X = np.mgrid[:H, :W]
    points = np.vstack((X.flatten(), Y.flatten())).T
    path = Path(polygon)
    mask = path.contains_points(points)
    return mask.reshape(H, W)

# ========================
# PREPROCESS DEM
# ========================
def preprocess_dem(dem, building_mask):
    dem = dem.copy()
    dem[building_mask == 1] += 5
    return dem

# ========================
# TERRAIN FEATURES
# ========================
def compute_slope(dem):
    dx = sobel(dem, axis=1)
    dy = sobel(dem, axis=0)
    return np.sqrt(dx**2 + dy**2)

def compute_flow_accumulation(dem):
    flow = np.nanmax(dem) - dem
    flow[flow < 0] = 0
    return flow / np.nanmax(flow)

# ========================
# DRAINAGE FEATURES
# ========================
def generate_drains():
    return [
        {"x": 20, "y": 20, "diameter": 1.0},
        {"x": 70, "y": 50, "diameter": 1.2},
        {"x": 50, "y": 80, "diameter": 0.8},
    ]

def distance_to_nearest_drain(H, W, drains):
    dist = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            dist[i, j] = min(
                np.sqrt((i-d["y"])**2 + (j-d["x"])**2)
                for d in drains
            )
    return dist

def compute_q_max(drains):
    q = []
    for d in drains:
        area = np.pi * (d["diameter"]/2)**2
        q.append(area * 3.0)
    return np.mean(q)

# ========================
# FEATURE VECTOR
# ========================
def build_features(slope, flow, dist, rain, qmax, load):
    mask = ~np.isnan(slope)
    X = np.stack([
        slope[mask],
        flow[mask],
        dist[mask],
        np.full(mask.sum(), rain),
        np.full(mask.sum(), qmax),
        np.full(mask.sum(), load)
    ], axis=1)
    return X, mask

# ========================
# LABEL (SIMULATION)
# ========================
def generate_drain_time(slope, flow, dist, rain):
    time = 100 - 0.8*slope - 1.2*flow + 0.05*dist + 0.4*rain
    return np.clip(time, 10, 300)

# ========================
# TRAIN MODEL
# ========================
def train_model(X, y):
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y)
    return model

def plot_dem_3d(dem, title="3D DEM"):
    """
    Visualize DEM as 3D surface
    """
    z = dem.copy()

    # xử lý NaN để plot không bị lỗi
    if np.isnan(z).any():
        z = np.nan_to_num(z, nan=np.nanmin(z))

    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                colorscale="earth",
                showscale=True
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Elevation (m)",
            aspectratio=dict(x=1, y=1, z=0.4)
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

from scipy.ndimage import gaussian_filter

def smooth_dem(dem, sigma=2.0):
    """
    Smooth DEM to reduce spike artifacts in 3D visualization
    """
    dem_smooth = dem.copy()
    mask = np.isnan(dem_smooth)

    dem_smooth[mask] = np.nanmean(dem_smooth)
    dem_smooth = gaussian_filter(dem_smooth, sigma=sigma)
    dem_smooth[mask] = np.nan

    return dem_smooth
