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
    files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix == '.csv']
    all_data = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df = df.sort_values('ts')
            # 1. Standardize orientation columns
            rename_map = {'r': 'roll', 'p': 'pitch', 'y': 'yaw', 'R': 'roll', 'P': 'pitch', 'Y': 'yaw'}
            df.rename(columns=rename_map, inplace=True)

            # 2. Add Run Metadata
            df['run_name'] = f.stem
            df['rel_time'] = df['ts'] - df['ts'].min()

            cols_to_fix = [c for c in df.columns if c not in ['ts', 'run_name']]
            df[cols_to_fix] = df[cols_to_fix].apply(pd.to_numeric, errors='coerce')

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
    jitter_cols = [c for c in df_all.columns if 'jitter' in c and c != 'jitter']
    if jitter_cols:
        df_all['jitter'] = df_all[jitter_cols].mean(axis=1)

    err_cols = [c for c in df_all.columns if c.startswith('pt') and '_err' in c]
    if err_cols:
        df_all['spatial_std'] = df_all[err_cols].std(axis=1)

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
        plot_df = filtered_df.drop_duplicates(subset=['run_name', 'rel_time'])

        fig_glob = px.line(
            plot_df,
            x='rel_time',
            y=metric,  # This will now work because 'jitter' was created above
            color='run_name',
            markers=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig_glob, use_container_width=True)

        # fig_glob = px.line(filtered_df, x='rel_time', y=metric, color='run_name',
        #                    template="plotly_dark", title=f"Global {metric.upper()} Over Time")
        # st.plotly_chart(fig_glob, use_container_width=True)

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

    # 3. ERROR VS DISTANCE (Full Analysis: Spatiotemporal + Calibration)
    with tab_dist:
        st.subheader("Depth Accuracy & Calibration Analysis")

        dist_analysis = []
        err_cols = sorted([c for c in filtered_df.columns if '_err' in c])

        # Data Preparation
        for i in range(len(err_cols)):
            err_col = f'pt{i}_err'
            gt_col = f'gt_depth_{i}'

            if err_col in filtered_df.columns and gt_col in filtered_df.columns:
                tmp = filtered_df[['run_name', 'rel_time', err_col, gt_col]].copy()
                tmp.columns = ['run_name', 'time', 'error', 'gt_distance']

                tmp['error'] = pd.to_numeric(tmp['error'], errors='coerce')
                tmp['gt_distance'] = pd.to_numeric(tmp['gt_distance'], errors='coerce')

                # Measured Distance = Ground Truth + Error
                tmp['measured_distance'] = tmp['gt_distance'] + tmp['error']

                tmp = tmp.dropna(subset=['error', 'gt_distance'])
                tmp['point_id'] = f'Pt {i}'
                dist_analysis.append(tmp)

        if dist_analysis:
            df_plot = pd.concat(dist_analysis)


            # Helper for Trend Equations
            def get_trend_details(df, x_col, y_col):
                clean = df[[x_col, y_col]].dropna()
                if len(clean) < 2: return "N/A", 1.0
                m, b = np.polyfit(clean[x_col], clean[y_col], 1)
                equation = f"y = {m:.4f}x + {b:.4f}"
                scale_factor = 1.0 - m
                return equation, scale_factor


            # --- PART 1: SPATIOTEMPORAL (Bubble & 3D) ---
            st.write("### 1. Error as a Function of Time & Distance")
            fig_2d = px.scatter(
                df_plot, x='time', y='gt_distance', size='error', color='error',
                size_max=5, hover_data=['point_id'], template="plotly_dark",
                color_continuous_scale='Reds', title="Error Magnitude Over Time"
            )
            fig_2d.update_traces(marker=dict(line=dict(width=0.1, color='rgba(100,100,100,0.5)')))
            st.plotly_chart(fig_2d, use_container_width=True)

            st.write("### 2. 3D Spatiotemporal Surface")
            fig_3d = px.scatter_3d(
                df_plot, x='time', y='gt_distance', z='error',
                color='error', template="plotly_dark", opacity=0.6
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            st.divider()

            # --- PART 2: CALIBRATION TRENDS ---
            col_a, col_b = st.columns(2)

            with col_a:
                st.write("### 3. Error vs. Ground Truth")
                eq1, _ = get_trend_details(df_plot, 'gt_distance', 'error')
                fig_gt = px.scatter(df_plot, x='gt_distance', y='error', color='run_name',
                                    opacity=0.3, template="plotly_dark", trendline="ols")
                fig_gt.add_annotation(text=f"Trend: {eq1}", xref="paper", yref="paper",
                                      x=0.05, y=0.95, showarrow=False, font=dict(color="yellow", size=14))
                st.plotly_chart(fig_gt, use_container_width=True)

            with col_b:
                st.write("### 4. Error vs. Measured Distance (DA3)")
                eq2, scale = get_trend_details(df_plot, 'measured_distance', 'error')
                fig_meas = px.scatter(df_plot, x='measured_distance', y='error', color='run_name',
                                      opacity=0.3, template="plotly_dark", trendline="ols")
                fig_meas.add_annotation(
                    text=f"<b>Equation:</b> {eq2}<br><b>Multiplier:</b> {scale:.4f}",
                    xref="paper", yref="paper", x=0.05, y=0.95, showarrow=False,
                    bgcolor="rgba(0,0,0,0.5)", font=dict(color="cyan", size=14)
                )
                st.plotly_chart(fig_meas, use_container_width=True)

            st.success(
                f"**Calibration Multiplier:** Multiply DA3 raw output by **{scale:.4f}** to align with Ground Truth.")

        else:
            st.warning("No valid point data found for distance analysis.")    # 4. ORIENTATION
    with tab_axes:
        x_ax = st.selectbox("Orientation Axis", ["pitch", "roll", "yaw"])
        fig_ax = px.scatter(filtered_df, x=x_ax, y='mae', color='run_name',
                            trendline="ols", template="plotly_dark",
                            title=f"Trend: MAE vs {x_ax.capitalize()}")
        st.plotly_chart(fig_ax, use_container_width=True)

else:
    st.info("Please select a directory containing your CSV and JPG files.")