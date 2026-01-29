import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from model_utils import *

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("🌧️ Urban Flood Drainage Prediction")

st.markdown("""
**Input**: DEM, Boundary Polygon, Building Mask, Drainage Infrastructure, Rainfall Intensity  
**Output**: Drainage Time Map
""")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")

# =========================
# SIDEBAR – INPUT
# =========================
st.sidebar.header("1️⃣ Terrain Data")

dem_file = st.sidebar.file_uploader("Upload DEM (.npy)", type=["npy"])
mask_file = st.sidebar.file_uploader("Upload Building Mask (.npy)", type=["npy"])

polygon_text = st.sidebar.text_area(
    "Boundary Polygon (x,y per line)",
    value="10,10\n90,10\n90,90\n10,90"
)

# -------------------------
st.sidebar.header("2️⃣ Drainage Infrastructure")

num_drains = st.sidebar.number_input(
    "Number of drains", min_value=1, max_value=10, value=3
)

drains = []
for i in range(num_drains):
    st.sidebar.markdown(f"**Drain {i+1}**")
    x = st.sidebar.number_input(f"x{i}", 0, 500, 20 + i*10)
    y = st.sidebar.number_input(f"y{i}", 0, 500, 20 + i*10)
    d = st.sidebar.slider(f"Diameter {i} (m)", 0.3, 2.0, 1.0, 0.1)

    drains.append({
        "x": x,
        "y": y,
        "diameter": d
    })

# -------------------------
st.sidebar.header("3️⃣ Rainfall")

rain_intensity = st.sidebar.slider(
    "Rain intensity (mm/h)", 10, 200, 80, 10
)

rain_duration = st.sidebar.slider(
    "Rain duration (hours)", 
    min_value=0.5,
    max_value=12.0,
    value=2.0,
    step=0.5
)


# -------------------------
run_btn = st.sidebar.button("🚀 Run Prediction")

# =========================
# MAIN PIPELINE
# =========================
if run_btn and dem_file and mask_file:

    # ---- Load terrain ----
    DEM_raw = np.load(dem_file)
    building_mask = np.load(mask_file)
    H, W = DEM_raw.shape

    # ---- Polygon ----
    polygon = [tuple(map(float, l.split(","))) for l in polygon_text.splitlines()]
    boundary_mask = polygon_to_mask(H, W, polygon)

    # ---- Preprocess DEM ----
    DEM = preprocess_dem(DEM_raw, building_mask)
    DEM[~boundary_mask] = np.nan

    # ---- Terrain features ----
    slope = compute_slope(DEM)
    flow = compute_flow_accumulation(DEM)

    # ---- Drainage features ----
    dist = distance_to_nearest_drain(H, W, drains)

    # Q_max chỉ phụ thuộc đường kính (đúng bài toán gốc)
    Q_max = compute_q_max(drains)
    load_index = rain_intensity / (Q_max + 1e-6)

    # ---- Feature vector ----
    X, valid_mask = build_features(
    slope, flow, dist,
    rain_intensity, Q_max, load_index
)

    rain_duration_feat = np.full((X.shape[0], 1), rain_duration)
    X = np.hstack([X, rain_duration_feat])


    # ---- Prediction ----
    pred = np.full((H, W), np.nan)
    pred_vals = model.predict(X)
    pred[valid_mask] = pred_vals

    # =========================
    # VISUALIZATION
    # =========================
    tab1, tab2, tab3 = st.tabs([
        "2D Results",
        "3D DEM (Original)",
        "3D DEM (After Processing)"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("DEM (after boundary & buildings)")
            fig, ax = plt.subplots()
            im = ax.imshow(DEM, cmap="terrain")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Predicted Drainage Time (minutes)")
            fig, ax = plt.subplots()
            im = ax.imshow(pred, cmap="Blues")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

        st.success(
            f"Average drainage time in polygon: {np.nanmean(pred):.1f} minutes"
        )

    with tab2:
        DEM_smooth = smooth_dem(DEM, sigma=2.5)
        st.plotly_chart(
            plot_dem_3d(DEM_smooth, "3D DEM – Original"),
            use_container_width=True
        )

    with tab3:
        DEM_smooth = smooth_dem(DEM, sigma=2.5)
        st.plotly_chart(
            plot_dem_3d(DEM_smooth, "3D DEM – After Polygon & Building Mask"),
            use_container_width=True
        )

elif run_btn:
    st.warning("⚠️ Please upload DEM and Building Mask.")
