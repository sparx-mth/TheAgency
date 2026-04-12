
# Updated app with calibration comparison (raw / linear / quadratic)

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

st.set_page_config(page_title="DA3 Marker Analysis", layout="wide")
st.title("📊 DA3 Marker Validator + Calibration")

COLOR_MAP = {
    "red": "#ff0000",
    "green": "#00ff00",
    "yellow": "#ffff00",
    "orange": "#ff8800",
    "purple": "#aa00ff",
    "cyan": "#00ffff",
}

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["gt_depth_geom_m","da3_depth_m"])
    df = df[df["gt_depth_geom_m"] > 1e-6]
    df["rel_time"] = df["ts"] - df["ts"].min()
    return df

file = st.file_uploader("Upload CSV", type=["csv"])
if not file:
    st.stop()

df = load_data(file)

# ===== Calibration coefficients (from your CSV) =====
LIN_M = 0.5005
LIN_B = 0.6114

QUAD_A = 0.05296
QUAD_B = 0.1069
QUAD_C = 1.1834

# ===== Apply calibration =====
df["da3_lin"] = LIN_M * df["da3_depth_m"] + LIN_B
df["da3_quad"] = QUAD_A * df["da3_depth_m"]**2 + QUAD_B * df["da3_depth_m"] + QUAD_C

# ===== Errors =====
df["err_raw"] = abs(df["da3_depth_m"] - df["gt_depth_geom_m"])
df["err_lin"] = abs(df["da3_lin"] - df["gt_depth_geom_m"])
df["err_quad"] = abs(df["da3_quad"] - df["gt_depth_geom_m"])

# ===== Ratios =====
df["ratio_raw"] = df["da3_depth_m"] / df["gt_depth_geom_m"]
df["ratio_lin"] = df["da3_lin"] / df["gt_depth_geom_m"]
df["ratio_quad"] = df["da3_quad"] / df["gt_depth_geom_m"]

tab1, tab2 = st.tabs(["Error Comparison", "Ratio Comparison"])

# ===== Error comparison =====
with tab1:
    st.subheader("Absolute Error Comparison")

    fig = px.scatter(
        df,
        x="gt_depth_geom_m",
        y="err_raw",
        color="color",
        color_discrete_map=COLOR_MAP,
        title="Raw Error vs GT"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df,
        x="gt_depth_geom_m",
        y="err_lin",
        color="color",
        color_discrete_map=COLOR_MAP,
        title="Linear Corrected Error vs GT"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df,
        x="gt_depth_geom_m",
        y="err_quad",
        color="color",
        color_discrete_map=COLOR_MAP,
        title="Quadratic Corrected Error vs GT"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### MAE Summary")
    st.write({
        "raw": float(df["err_raw"].mean()),
        "linear": float(df["err_lin"].mean()),
        "quadratic": float(df["err_quad"].mean()),
    })

# ===== Ratio comparison =====
with tab2:
    st.subheader("DA3 / GT Ratio Comparison")

    fig = px.scatter(
        df,
        x="gt_depth_geom_m",
        y="ratio_raw",
        color="color",
        color_discrete_map=COLOR_MAP,
        title="Raw Ratio vs GT"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df,
        x="gt_depth_geom_m",
        y="ratio_lin",
        color="color",
        color_discrete_map=COLOR_MAP,
        title="Linear Ratio vs GT"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df,
        x="gt_depth_geom_m",
        y="ratio_quad",
        color="color",
        color_discrete_map=COLOR_MAP,
        title="Quadratic Ratio vs GT"
    )
    st.plotly_chart(fig, use_container_width=True)

