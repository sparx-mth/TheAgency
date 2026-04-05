import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Depth Analysis Pro", layout="wide")
st.title("📊 Depth Performance & Visual Context")


@st.cache_data
def load_and_prepare_data(directory):
    data_dir = Path(directory).expanduser()
    files = list(data_dir.glob("*.csv"))
    all_data = []

    for f in files:
        try:
            df = pd.read_csv(f)
            # 1. Standardize orientation columns
            rename_map = {'r': 'roll', 'p': 'pitch', 'y': 'yaw', 'R': 'roll', 'P': 'pitch', 'Y': 'yaw'}
            df.rename(columns=rename_map, inplace=True)

            # 2. Add Run Metadata
            df['run_name'] = f.stem
            df['rel_time'] = df['ts'] - df['ts'].min()

            # 3. Identify Point columns
            err_cols = [c for c in df.columns if c.startswith('pt') and '_err' in c]
            if err_cols:
                df['spatial_std'] = df[err_cols].std(axis=1)

            # 4. Clean Data (ignore crashes)
            if 'mae' in df.columns and (df['mae'] > 8.0).any():
                cut_point = df[df['mae'] > 8.0].index[0]
                df = df.iloc[:cut_point]

            all_data.append(df)
        except Exception as e:
            st.error(f"Error loading {f.name}: {e}")

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# --- SIDEBAR ---
st.sidebar.header("Data Control")
path_input = st.sidebar.text_input("CSV Directory", "~/Documents/depth_validator_csv")
base_path = Path(path_input).expanduser()
df_all = load_and_prepare_data(path_input)

if not df_all.empty:
    runs = df_all['run_name'].unique().tolist()
    selected_runs = st.sidebar.multiselect("Select Runs", runs, default=runs)
    filtered_df = df_all[df_all['run_name'].isin(selected_runs)]

    # --- IMAGE GALLERY SECTION ---
    st.subheader("🖼️ Reference Frames")
    img_cols = st.columns(len(selected_runs) if selected_runs else 1)

    for idx, run in enumerate(selected_runs):
        # Naming Logic: da3_val_20260405_173327 -> 20260405_173327.jpg
        img_timestamp = run.replace("da3_val_", "")
        img_path = base_path / f"{img_timestamp}.jpg"

        with img_cols[idx % len(img_cols)]:
            if img_path.exists():
                st.image(str(img_path), caption=f"Run: {img_timestamp}", use_container_width=True)
            else:
                st.info(f"No image found for {run}\n(Expected: {img_timestamp}.jpg)")

    # --- TABS ---
    tab_glob, tab_pts, tab_dist, tab_axes = st.tabs([
        "Global Comparison", "Point Tracking", "Error vs Distance", "Orientation Axes"
    ])

    # 1. GLOBAL
    with tab_glob:
        metric = st.selectbox("Metric", ["mae", "rmse", "spatial_std", "jitter"])
        fig_glob = px.line(filtered_df, x='rel_time', y=metric, color='run_name',
                           template="plotly_dark", title=f"Global {metric.upper()} Over Time")
        st.plotly_chart(fig_glob, use_container_width=True)

    # 2. POINT TRACKING
    with tab_pts:
        err_cols = sorted([c for c in filtered_df.columns if '_err' in c])
        if err_cols:
            df_melted = filtered_df.melt(
                id_vars=['rel_time', 'run_name'],
                value_vars=err_cols,
                var_name='Point_ID', value_name='Error_m'
            )
            fig_pts = px.line(df_melted, x='rel_time', y='Error_m', color='Point_ID',
                              facet_col='run_name', template="plotly_dark",
                              title="Spatial Error Distribution per Pixel")
            st.plotly_chart(fig_pts, use_container_width=True)

    # 3. ERROR VS DISTANCE
    with tab_dist:
        st.subheader("Depth Accuracy Decay Analysis")
        dist_analysis = []
        err_cols = sorted([c for c in filtered_df.columns if '_err' in c])

        for i in range(len(err_cols)):
            # Pair ptX_err with gt_depth_X
            err_col = f'pt{i}_err'
            gt_col = f'gt_depth_{i}'

            if err_col in filtered_df.columns and gt_col in filtered_df.columns:
                tmp = filtered_df[['run_name', err_col, gt_col]].copy()
                tmp.columns = ['run_name', 'error', 'distance']
                tmp['point_id'] = f'Point {i}'
                dist_analysis.append(tmp)

        if dist_analysis:
            df_dist = pd.concat(dist_analysis)
            fig_dist = px.scatter(df_dist, x='distance', y='error', color='run_name',
                                  trendline="ols", opacity=0.4,
                                  template="plotly_dark", title="Trend: Error vs. Ground Truth Distance")
            st.plotly_chart(fig_dist, use_container_width=True)

    # 4. ORIENTATION
    with tab_axes:
        x_ax = st.selectbox("Orientation Axis", ["pitch", "roll", "yaw"])
        fig_ax = px.scatter(filtered_df, x=x_ax, y='mae', color='run_name',
                            trendline="ols", template="plotly_dark",
                            title=f"Trend: MAE vs {x_ax.capitalize()}")
        st.plotly_chart(fig_ax, use_container_width=True)

else:
    st.info("Please select a directory containing your CSV and JPG files.")