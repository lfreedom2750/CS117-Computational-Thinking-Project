import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from model_utils import *

# ========================
# CONFIG
# ========================
H, W = 100, 100
N_REGIONS = 30

RAIN_INTENSITY = 80
RAIN_DURATION = 80

# ========================
# REGION GENERATOR
# ========================
def generate_region():
    DEM = np.random.rand(H, W) * 50

    building_mask = np.zeros((H, W))
    x0, y0 = np.random.randint(10, 60), np.random.randint(10, 60)
    building_mask[y0:y0+20, x0:x0+30] = 1

    polygon = [
        (10, 10),
        (90, 10),
        (80, np.random.randint(50, 80)),
        (20, np.random.randint(60, 90))
    ]

    boundary_mask = polygon_to_mask(H, W, polygon)
    DEM[~boundary_mask] = np.nan
    building_mask[~boundary_mask] = 0

    return DEM, building_mask


# ========================
# BUILD DATASET
# ========================
X_all, y_all, groups = [], [], []

for region_id in range(N_REGIONS):
    DEM, building_mask = generate_region()

    # preprocess
    DEM = preprocess_dem(DEM, building_mask)

    slope = compute_slope(DEM)
    flow = compute_flow_accumulation(DEM)

    drains = generate_drains()
    dist = distance_to_nearest_drain(H, W, drains)

    Q_max = compute_q_max(drains)
    load = RAIN_INTENSITY / (Q_max + 1e-6)

    # features
    X, mask = build_features(
        slope, flow, dist, RAIN_INTENSITY, Q_max, load
    )

    # add rain duration
    rain_duration_feat = np.full((X.shape[0], 1), RAIN_DURATION)
    X = np.hstack([X, rain_duration_feat])

    # label
    y_map = generate_drain_time(
        slope, flow, dist, RAIN_INTENSITY
    )
    y = y_map[mask]

    # group id
    group = np.full(len(y), region_id)

    X_all.append(X)
    y_all.append(y)
    groups.append(group)


X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)
groups = np.concatenate(groups)

print("Total samples:", len(y_all))
print("Total regions:", N_REGIONS)

# ========================
# TRAIN / TEST SPLIT (BY REGION)
# ========================
gss = GroupShuffleSplit(
    test_size=0.2,
    n_splits=1,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X_all, y_all, groups=groups)
)

X_train, X_test = X_all[train_idx], X_all[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]

print("Train regions:", np.unique(groups[train_idx]).size)
print("Test regions:", np.unique(groups[test_idx]).size)

# ========================
# TRAIN
# ========================
model = train_model(X_train, y_train)

# ========================
# EVALUATE
# ========================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"📊 Test MAE: {mae:.2f} minutes")

# ========================
# SAVE
# ========================
joblib.dump(model, "model.pkl")
print("✅ Model saved")
