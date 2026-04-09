
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

st.set_page_config(page_title="DA3 Landmark Analysis", layout="wide")
st.title("📊 DA3 Landmark Validator")

LANDMARK_COLUMNS = {
    "ts", "marker_id", "color",
    "det_u", "det_v", "gt_u", "gt_v",
    "roll_deg", "pitch_deg", "yaw_deg",
    "gt_depth_geom_m", "da3_depth_m",
    "abs_err_m", "jitter_m"
}


@st.cache_data
def load_and_prepare_data(directory: str) -> pd.DataFrame:
    data_dir = Path(directory).expanduser()
    files = sorted([f for f in data_dir.iterdir() if f.is_file() and f.suffix == ".csv"])
    all_data = []

    for f in files:
        try:
            df = pd.read_csv(f)

            # Standardize old/new angle names
            rename_map = {"r": "roll_deg", "p": "pitch_deg", "y": "yaw_deg",
                          "roll": "roll_deg", "pitch": "pitch_deg", "yaw": "yaw_deg"}
            df.rename(columns=rename_map, inplace=True)

            # Detect format
            is_landmark = LANDMARK_COLUMNS.issubset(df.columns)
            df["run_name"] = f.stem
            df["rel_time"] = df["ts"] - df["ts"].min()
            df["format"] = "landmarks" if is_landmark else "legacy"

            if is_landmark:
                numeric_cols = [c for c in df.columns if c not in {"object_id", "landmark_id", "run_name", "format"}]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
                # Per-frame global metrics for convenience
                per_frame = (
                    df.groupby(["run_name", "ts"], as_index=False)
                    .agg(
                        rel_time=("rel_time", "first"),
                        mae=("abs_err_m", "mean"),
                        rmse=("abs_err_m", lambda s: float(np.sqrt(np.mean(np.square(s.dropna())))) if len(s.dropna()) else np.nan),
                        jitter=("jitter_m", "mean"),
                        n_markers=("marker_id", "count"),
                        roll_deg=("roll_deg", "first"),
                        pitch_deg=("pitch_deg", "first"),
                        yaw_deg=("yaw_deg", "first"),
                    )
                )
                df = df.merge(per_frame, on=["run_name", "ts"], how="left", suffixes=("", "_frame"))
            else:
                cols_to_fix = [c for c in df.columns if c not in {"run_name", "format"}]
                df[cols_to_fix] = df[cols_to_fix].apply(pd.to_numeric, errors="coerce")
                jitter_cols = [c for c in df.columns if "jitter" in c and c != "jitter"]
                err_cols = [c for c in df.columns if c.startswith("pt") and "_err" in c]
                if jitter_cols and "jitter" not in df.columns:
                    df["jitter"] = df[jitter_cols].mean(axis=1)
                if err_cols and "spatial_std" not in df.columns:
                    df["spatial_std"] = df[err_cols].std(axis=1)

            all_data.append(df)
        except Exception as e:
            st.error(f"Error loading {f.name}: {e}")

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def image_candidates(base_path: Path, run_name: str):
    stamp = run_name.replace("da3_landmarks_", "").replace("da3_val_", "")
    return [
        base_path / f"{stamp}_rgb_landmarks.jpg",
        base_path / f"{stamp}.jpg",
        base_path / f"{run_name}.jpg",
    ]


st.sidebar.header("Data Control")
path_input = st.sidebar.text_input("CSV Directory", "~/Documents/depth_validator_csv")
base_path = Path(path_input).expanduser()
df_all = load_and_prepare_data(path_input)

if df_all.empty:
    st.info("Please select a directory containing your CSV files.")
    st.stop()

formats = df_all["format"].dropna().unique().tolist()
selected_format = st.sidebar.selectbox("Format", formats, index=0)
df_all = df_all[df_all["format"] == selected_format]

runs = df_all["run_name"].dropna().unique().tolist()
selected_runs = st.sidebar.multiselect("Select Runs", runs, default=runs)
filtered_df = df_all[df_all["run_name"].isin(selected_runs)].copy()

if filtered_df.empty:
    st.warning("No rows match the selected runs.")
    st.stop()

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

if selected_format == "landmarks":
    tab_glob, tab_obj, tab_lmk, tab_dist, tab_axes = st.tabs([
        "Global", "By Object", "By Landmark", "Distance / Calibration", "Orientation"
    ])

    frame_df = (
        filtered_df.groupby(["run_name", "ts"], as_index=False)
        .agg(
            rel_time=("rel_time", "first"),
            mae=("abs_err_m", "mean"),
            rmse=("abs_err_m", lambda s: float(np.sqrt(np.mean(np.square(s.dropna())))) if len(s.dropna()) else np.nan),
            jitter=("jitter_m", "mean"),
            n_landmarks=("landmark_id", "count"),
            roll_deg=("roll_deg", "first"),
            pitch_deg=("pitch_deg", "first"),
            yaw_deg=("yaw_deg", "first"),
        )
    )

    with tab_glob:
        metric = st.selectbox("Metric", ["mae", "rmse", "jitter", "n_landmarks"])
        fig = px.line(
            frame_df,
            x="rel_time",
            y=metric,
            color="run_name",
            markers=True,
            template="plotly_dark",
            title=f"{metric.upper()} over time",
        )
        st.plotly_chart(fig, use_container_width=True)

        summary_df = (
            filtered_df.groupby(["run_name"], as_index=False)
            .agg(
                mean_mae=("abs_err_m", "mean"),
                mean_rmse=("abs_err_m", lambda s: float(np.sqrt(np.mean(np.square(s.dropna())))) if len(s.dropna()) else np.nan),
                mean_jitter=("jitter_m", "mean"),
                rows=("landmark_id", "count"),
            )
        )
        st.dataframe(summary_df, use_container_width=True)

    with tab_obj:
        obj_df = (
            filtered_df.groupby(["run_name", "marker_id", "rel_time"], as_index=False)
            .agg(
                mae=("abs_err_m", "mean"),
                jitter=("jitter_m", "mean"),
                gt_depth_m=("gt_depth_m", "mean"),
                da3_depth_m=("da3_depth_m", "mean"),
            )
        )
        metric = st.selectbox("Object metric", ["mae", "jitter", "gt_depth_m", "da3_depth_m"])
        fig = px.line(
            obj_df,
            x="rel_time",
            y=metric,
            color="object_id",
            facet_col="run_name",
            template="plotly_dark",
            title=f"{metric} by object",
        )
        st.plotly_chart(fig, use_container_width=True)

        obj_summary = (
            filtered_df.groupby(["object_id"], as_index=False)
            .agg(
                mean_abs_err_m=("abs_err_m", "mean"),
                p95_abs_err_m=("abs_err_m", lambda s: np.nanpercentile(s.dropna(), 95) if len(s.dropna()) else np.nan),
                mean_jitter_m=("jitter_m", "mean"),
                mean_gt_depth_m=("gt_depth_m", "mean"),
            )
            .sort_values("mean_abs_err_m")
        )
        st.dataframe(obj_summary, use_container_width=True)

    with tab_lmk:
        landmark_keys = sorted(filtered_df["marker_id"].astype(str))
        selected_landmarks = st.multiselect("Landmarks", landmark_keys, default=landmark_keys[: min(8, len(landmark_keys))])
        plot_df = filtered_df.copy()
        plot_df["landmark_key"] = plot_df["object_id"].astype(str) + "/" + plot_df["landmark_id"].astype(str)
        plot_df = plot_df[plot_df["landmark_key"].isin(selected_landmarks)]
        fig = px.line(
            plot_df,
            x="rel_time",
            y="abs_err_m",
            color="landmark_key",
            facet_col="run_name",
            template="plotly_dark",
            title="Absolute error by landmark",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            plot_df,
            x="det_u",
            y="det_v",
            color="abs_err_m",
            hover_data=["marker_id", "gt_depth_geom_m", "da3_depth_m"],
            facet_col="run_name",
            template="plotly_dark",
            title="Projected landmark locations colored by error",
        )
        fig2.update_yaxes(autorange="reversed")
        st.plotly_chart(fig2, use_container_width=True)

    with tab_dist:
        plot_df = filtered_df.dropna(subset=["gt_depth_geom_m", "da3_depth_m", "abs_err_m"]).copy()
        plot_df["signed_err_m"] = plot_df["da3_depth_m"] - plot_df["gt_depth_geom_m"]
        plot_df["landmark_key"] = plot_df["object_id"].astype(str) + "/" + plot_df["landmark_id"].astype(str)

        fig = px.scatter(
            plot_df,
            x="gt_depth_geom_m",
            y="signed_err_m",
            color="object_id",
            hover_data=["landmark_key", "run_name"],
            template="plotly_dark",
            trendline="ols",
            title="Signed error vs GT depth",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            plot_df,
            x="da3_depth_m",
            y="signed_err_m",
            color="object_id",
            hover_data=["landmark_key", "run_name"],
            template="plotly_dark",
            trendline="ols",
            title="Signed error vs DA3 depth",
        )
        st.plotly_chart(fig2, use_container_width=True)

        if len(plot_df) >= 2:
            clean = plot_df[["da3_depth_m", "gt_depth_geom_m"]].dropna()
            m, b = np.polyfit(clean["gt_depth_geom_m"], clean["gt_depth_geom_m"], 1)
            st.success(f"Calibration fit: gt_depth ≈ {m:.4f} * da3_depth + {b:.4f}")

    with tab_axes:
        axis = st.selectbox("Orientation axis", ["pitch_deg", "roll_deg", "yaw_deg"])
        ymetric = st.selectbox("Response", ["abs_err_m", "jitter_m", "gt_depth_m"])
        fig = px.scatter(
            filtered_df,
            x=axis,
            y=ymetric,
            color="object_id",
            hover_data=["landmark_id", "run_name"],
            template="plotly_dark",
            trendline="ols",
            title=f"{ymetric} vs {axis}",
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("This app now expects landmark CSVs. The selected data looks like the old random-pixel format.")
