
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

st.set_page_config(page_title="DA3 Marker Analysis", layout="wide")
st.title("📊 DA3 Marker Analysis")

MARKER_REQUIRED = {
    "ts", "marker_id", "color",
    "gt_depth_geom_m", "jitter_m"
}

COLOR_MAP = {
    "red": "#ff0000",
    "green": "#00ff00",
    "yellow": "#ffff00",
    "orange": "#ff8800",
    "purple": "#aa00ff",
    "cyan": "#00ffff",
}

DEPTH_VARIANTS = {
    "Raw": ("da3_raw_m", "err_raw_m"),
    "Linear": ("da3_lin_m", "err_lin_m"),
    "Quadratic": ("da3_quad_m", "err_quad_m"),
}


def marker_color_map(df: pd.DataFrame):
    mapping = {}
    for _, row in df[["marker_id", "color"]].drop_duplicates().iterrows():
        mapping[row["marker_id"]] = COLOR_MAP.get(str(row["color"]).lower(), "#ffffff")
    return mapping


@st.cache_data
def load_and_prepare_data(directory: str) -> pd.DataFrame:
    data_dir = Path(directory).expanduser()
    files = sorted([f for f in data_dir.iterdir() if f.is_file() and f.suffix == ".csv"])
    all_data = []

    for f in files:
        try:
            df = pd.read_csv(f)

            rename_map = {
                "r": "roll_deg", "p": "pitch_deg", "y": "yaw_deg",
                "roll": "roll_deg", "pitch": "pitch_deg", "yaw": "yaw_deg",
                "err_raw": "err_raw_m", "err_lin": "err_lin_m", "err_quad": "err_quad_m",
            }
            df.rename(columns=rename_map, inplace=True)

            is_marker = MARKER_REQUIRED.issubset(df.columns)

            if not is_marker:
                continue

            df["run_name"] = f.stem
            df["wall_id"] = f.stem
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
            df["rel_time"] = df["ts"] - df["ts"].min()
            df["format"] = "markers"

            non_numeric = {"marker_id", "color", "run_name", "wall_id", "format"}
            numeric_cols = [c for c in df.columns if c not in non_numeric]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

            if "detected" in df.columns:
                df["detected"] = df["detected"].fillna(0).astype(float)

            # Add missing legacy-compatible columns if needed
            if "da3_depth_m" in df.columns and "err_raw_m" not in df.columns:
                df["err_raw_m"] = (df["da3_depth_m"] - df["gt_depth_geom_m"]).abs()

            # Per-frame global metrics
            agg_map = {
                "rel_time": ("rel_time", "first"),
                "jitter": ("jitter_m", "mean"),
                "n_markers": ("marker_id", "count"),
                "roll_deg": ("roll_deg", "first"),
                "pitch_deg": ("pitch_deg", "first"),
                "yaw_deg": ("yaw_deg", "first"),
            }

            # Raw/default error
            if "err_raw_m" in df.columns:
                agg_map["mae_raw"] = ("err_raw_m", "mean")
                agg_map["rmse_raw"] = ("err_raw_m", lambda s: float(np.sqrt(np.mean(np.square(s.dropna())))) if len(s.dropna()) else np.nan)
            if "err_lin_m" in df.columns:
                agg_map["mae_lin"] = ("err_lin_m", "mean")
            if "err_quad_m" in df.columns:
                agg_map["mae_quad"] = ("err_quad_m", "mean")
            if "detected" in df.columns:
                agg_map["detection_rate"] = ("detected", "mean")

            per_frame = df.groupby(["run_name", "ts"], as_index=False).agg(**agg_map)
            df = df.merge(per_frame, on=["run_name", "ts"], how="left", suffixes=("", "_frame"))

            all_data.append(df)
        except Exception as e:
            st.error(f"Error loading {f.name}: {e}")

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def image_candidates(base_path: Path, run_name: str):
    stamp = (
        run_name.replace("da3_markers_", "")
        .replace("da3_landmarks_", "")
        .replace("da3_val_", "")
    )
    return [
        base_path / f"{stamp}_rgb_markers.jpg",
        base_path / f"{stamp}_rgb_landmarks.jpg",
        base_path / f"{stamp}.jpg",
        base_path / f"{run_name}.jpg",
    ]


st.sidebar.header("Data Control")
path_input = st.sidebar.text_input("CSV Directory", "~/Documents/depth_validator_csv")
base_path = Path(path_input).expanduser()

df_all = load_and_prepare_data(path_input)
if df_all.empty:
    st.info("No marker CSVs found in the selected directory.")
    st.stop()

runs = sorted(df_all["run_name"].dropna().unique().tolist())
selected_runs = st.sidebar.multiselect("Runs", runs, default=runs)
filtered_df = df_all[df_all["run_name"].isin(selected_runs)].copy()

marker_ids = sorted(filtered_df["marker_id"].dropna().unique().tolist())
selected_markers = st.sidebar.multiselect("Markers", marker_ids, default=marker_ids)
filtered_df = filtered_df[filtered_df["marker_id"].isin(selected_markers)].copy()

show_detected_only = st.sidebar.checkbox("Detected points only", value=True)
if show_detected_only and "detected" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["detected"] > 0].copy()

gt_valid = filtered_df["gt_depth_geom_m"].dropna()
if not gt_valid.empty:
    depth_min = float(gt_valid.min())
    depth_max = float(gt_valid.max())
    if depth_min < depth_max:
        selected_depth = st.sidebar.slider(
            "GT depth range (m)",
            min_value=depth_min,
            max_value=depth_max,
            value=(depth_min, depth_max),
        )
        filtered_df = filtered_df[
            filtered_df["gt_depth_geom_m"].between(selected_depth[0], selected_depth[1])
        ].copy()

if filtered_df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

marker_id_color_map = marker_color_map(filtered_df)

st.subheader("🖼️ Reference Frames")
img_cols = st.columns(max(1, min(len(selected_runs), 4)))
for idx, run in enumerate(selected_runs):
    found = None
    for cand in image_candidates(base_path, run):
        if cand.exists():
            found = cand
            break
    with img_cols[idx % len(img_cols)]:
        if found:
            st.image(str(found), caption=found.name, use_container_width=True)
        else:
            st.info(f"No image found for {run}")

tab_glob, tab_marker, tab_dist, tab_axes, tab_jitter = st.tabs([
    "Global", "By Marker", "Distance / Calibration", "Orientation", "Stability"
])

frame_df = (
    filtered_df.groupby(["run_name", "ts"], as_index=False)
    .agg(
        rel_time=("rel_time", "first"),
        jitter=("jitter_m", "mean"),
        n_markers=("marker_id", "count"),
        roll_deg=("roll_deg", "first"),
        pitch_deg=("pitch_deg", "first"),
        yaw_deg=("yaw_deg", "first"),
        detection_rate=("detected", "mean") if "detected" in filtered_df.columns else ("marker_id", "count"),
        mae_raw=("err_raw_m", "mean") if "err_raw_m" in filtered_df.columns else ("gt_depth_geom_m", "mean"),
        mae_lin=("err_lin_m", "mean") if "err_lin_m" in filtered_df.columns else ("gt_depth_geom_m", "mean"),
        mae_quad=("err_quad_m", "mean") if "err_quad_m" in filtered_df.columns else ("gt_depth_geom_m", "mean"),
    )
)

with tab_glob:
    metric_options = ["jitter", "n_markers"]
    for c in ["mae_raw", "mae_lin", "mae_quad", "detection_rate"]:
        if c in frame_df.columns:
            metric_options.append(c)

    metric = st.selectbox("Global metric", metric_options)
    fig = px.line(
        frame_df,
        x="rel_time",
        y=metric,
        color="run_name",
        markers=True,
        template="plotly_dark",
        title=f"{metric} over time",
    )
    st.plotly_chart(fig, use_container_width=True)

    summary_aggs = {
        "rows": ("marker_id", "count"),
        "mean_jitter": ("jitter_m", "mean"),
        "mean_gt_depth_m": ("gt_depth_geom_m", "mean"),
    }
    if "detected" in filtered_df.columns:
        summary_aggs["detection_rate"] = ("detected", "mean")
    for _, err_col in DEPTH_VARIANTS.values():
        if err_col in filtered_df.columns:
            summary_aggs[f"mean_{err_col}"] = (err_col, "mean")

    summary_df = filtered_df.groupby(["run_name"], as_index=False).agg(**summary_aggs)
    st.dataframe(summary_df, use_container_width=True)

with tab_marker:
    metric_options = ["jitter_m", "gt_depth_geom_m"]
    if "pixel_err_px" in filtered_df.columns:
        metric_options.append("pixel_err_px")
    for _, err_col in DEPTH_VARIANTS.values():
        if err_col in filtered_df.columns:
            metric_options.append(err_col)

    metric = st.selectbox("Marker metric", metric_options)

    marker_time_df = filtered_df.groupby(
        ["run_name", "marker_id", "color", "rel_time"], as_index=False
    ).agg(**{metric: (metric, "mean")})

    fig = px.line(
        marker_time_df,
        x="rel_time",
        y=metric,
        color="marker_id",
        color_discrete_map=marker_id_color_map,
        facet_col="run_name",
        hover_data=["color"],
        template="plotly_dark",
        title=f"{metric} by marker",
    )
    st.plotly_chart(fig, use_container_width=True)

    marker_summary_aggs = {
        "mean_jitter_m": ("jitter_m", "mean"),
        "mean_gt_depth_m": ("gt_depth_geom_m", "mean"),
    }
    if "pixel_err_px" in filtered_df.columns:
        marker_summary_aggs["mean_pixel_err_px"] = ("pixel_err_px", "mean")
    if "detected" in filtered_df.columns:
        marker_summary_aggs["detection_rate"] = ("detected", "mean")
    for _, err_col in DEPTH_VARIANTS.values():
        if err_col in filtered_df.columns:
            marker_summary_aggs[f"mean_{err_col}"] = (err_col, "mean")

    marker_summary = (
        filtered_df.groupby(["marker_id", "color"], as_index=False)
        .agg(**marker_summary_aggs)
        .sort_values(marker_summary_aggs.keys().__iter__().__next__())
    )
    st.dataframe(marker_summary, use_container_width=True)

with tab_dist:
    print(filtered_df.columns)
    depth_mode = st.selectbox("Depth mode", [k for k, (dcol, _) in DEPTH_VARIANTS.items() if DEPTH_VARIANTS[k][0] in filtered_df.columns])

    depth_col, err_col = DEPTH_VARIANTS[depth_mode]
    plot_df = filtered_df.dropna(subset=["gt_depth_geom_m", depth_col]).copy()
    plot_df = plot_df[plot_df["gt_depth_geom_m"] > 1e-6].copy()
    plot_df["signed_err_m"] = plot_df[depth_col] - plot_df["gt_depth_geom_m"]
    plot_df["scale_ratio"] = plot_df[depth_col] / plot_df["gt_depth_geom_m"]

    fig1 = px.scatter(
        plot_df,
        x="gt_depth_geom_m",
        y="scale_ratio",
        color="marker_id",
        color_discrete_map=marker_id_color_map,
        hover_data=["run_name", "color", depth_col],
        template="plotly_dark",
        trendline="ols",
        title=f"{depth_mode} / GT vs GT depth",
    )
    st.plotly_chart(fig1, use_container_width=True)

    ratio_time_df = plot_df.groupby(
        ["run_name", "ts", "rel_time", "marker_id", "color"], as_index=False
    ).agg(scale_ratio=("scale_ratio", "mean"))

    fig2 = px.line(
        ratio_time_df,
        x="rel_time",
        y="scale_ratio",
        color="marker_id",
        color_discrete_map=marker_id_color_map,
        facet_col="run_name",
        hover_data=["color"],
        template="plotly_dark",
        title=f"{depth_mode} / GT vs time",
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        plot_df,
        x="gt_depth_geom_m",
        y=err_col,
        color="marker_id",
        color_discrete_map=marker_id_color_map,
        hover_data=["run_name", "color"],
        template="plotly_dark",
        trendline="ols",
        title=f"{depth_mode} absolute error vs GT depth",
    )
    st.plotly_chart(fig3, use_container_width=True)

    if len(plot_df) >= 2:
        clean = plot_df[[depth_col, "gt_depth_geom_m"]].dropna()
        m, b = np.polyfit(clean[depth_col], clean["gt_depth_geom_m"], 1)
        st.success(f"Selected-data fit: gt_depth ≈ {m:.4f} * {depth_mode.lower()} + {b:.4f}")

with tab_axes:
    axis = st.selectbox("Orientation axis", ["pitch_deg", "roll_deg", "yaw_deg"])
    response_options = ["jitter_m", "gt_depth_geom_m"]
    for _, err_col in DEPTH_VARIANTS.values():
        if err_col in filtered_df.columns:
            response_options.append(err_col)
    ymetric = st.selectbox("Response", response_options)

    fig = px.scatter(
        filtered_df,
        x=axis,
        y=ymetric,
        color="marker_id",
        color_discrete_map=marker_id_color_map,
        hover_data=["run_name", "color"],
        template="plotly_dark",
        trendline="ols",
        title=f"{ymetric} vs {axis}",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_jitter:
    plot_df = filtered_df.dropna(subset=["jitter_m", "gt_depth_geom_m"]).copy()

    fig = px.line(
        plot_df,
        x="rel_time",
        y="jitter_m",
        color="marker_id",
        color_discrete_map=marker_id_color_map,
        facet_col="run_name",
        hover_data=["color"],
        template="plotly_dark",
        title="Jitter vs Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        plot_df,
        x="gt_depth_geom_m",
        y="jitter_m",
        color="marker_id",
        color_discrete_map=marker_id_color_map,
        hover_data=["run_name", "color"],
        template="plotly_dark",
        trendline="ols",
        title="Jitter vs Distance",
    )
    st.plotly_chart(fig2, use_container_width=True)

    jitter_summary = (
        plot_df.groupby(["run_name", "marker_id", "color"], as_index=False)
        .agg(
            mean_jitter=("jitter_m", "mean"),
            p95_jitter=("jitter_m", lambda s: np.nanpercentile(s.dropna(), 95) if len(s.dropna()) else np.nan),
            max_jitter=("jitter_m", "max"),
        )
        .sort_values(["run_name", "marker_id"])
    )
    st.dataframe(jitter_summary, use_container_width=True)
