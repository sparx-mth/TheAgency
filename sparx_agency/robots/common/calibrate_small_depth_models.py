#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CalibrationModel:
    def fit(self, raw: np.ndarray, gt: np.ndarray):
        raise NotImplementedError

    def predict(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def formula(self) -> str:
        return self.__class__.__name__


class ConstantScaleModel(CalibrationModel):
    def fit(self, raw: np.ndarray, gt: np.ndarray):
        denom = np.sum(raw * raw)
        self.k = float(np.sum(raw * gt) / max(denom, 1e-12))
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.k * raw

    def formula(self) -> str:
        return f"z = {self.k:.8f} * raw"


class LinearModel(CalibrationModel):
    def fit(self, raw: np.ndarray, gt: np.ndarray):
        self.a, self.b = np.polyfit(raw, gt, deg=1)
        self.a = float(self.a)
        self.b = float(self.b)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.a * raw + self.b

    def formula(self) -> str:
        return f"z = {self.a:.8f} * raw + {self.b:.8f}"


class QuadraticModel(CalibrationModel):
    def fit(self, raw: np.ndarray, gt: np.ndarray):
        self.a, self.b, self.c = np.polyfit(raw, gt, deg=2)
        self.a = float(self.a)
        self.b = float(self.b)
        self.c = float(self.c)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.a * raw * raw + self.b * raw + self.c

    def formula(self) -> str:
        return f"z = {self.a:.8f} * raw^2 + {self.b:.8f} * raw + {self.c:.8f}"


class CubicModel(CalibrationModel):
    def fit(self, raw: np.ndarray, gt: np.ndarray):
        self.a, self.b, self.c, self.d = np.polyfit(raw, gt, deg=3)
        self.a = float(self.a)
        self.b = float(self.b)
        self.c = float(self.c)
        self.d = float(self.d)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.a * raw**3 + self.b * raw**2 + self.c * raw + self.d

    def formula(self) -> str:
        return (
            f"z = {self.a:.8f} * raw^3 + {self.b:.8f} * raw^2 "
            f"+ {self.c:.8f} * raw + {self.d:.8f}"
        )


class InverseAffineModel(CalibrationModel):
    """
    Fits:
        1 / gt = a * raw + b

    Predicts:
        gt = 1 / (a * raw + b)
    """

    def fit(self, raw: np.ndarray, gt: np.ndarray):
        inv_gt = 1.0 / np.clip(gt, 1e-6, None)
        self.a, self.b = np.polyfit(raw, inv_gt, deg=1)
        self.a = float(self.a)
        self.b = float(self.b)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        denom = self.a * raw + self.b
        denom = np.clip(denom, 1e-6, None)
        return 1.0 / denom

    def formula(self) -> str:
        return f"z = 1 / ({self.a:.8f} * raw + {self.b:.8f})"


class PowerLawModel(CalibrationModel):
    """
    Fits:
        log(gt) = a * log(raw) + b

    Predicts:
        gt = exp(b) * raw^a
    """

    def fit(self, raw: np.ndarray, gt: np.ndarray):
        valid = (raw > 0) & (gt > 0)
        if np.sum(valid) < 3:
            raise RuntimeError("Not enough positive samples for power-law fit")

        x = np.log(raw[valid])
        y = np.log(gt[valid])

        self.a, self.b = np.polyfit(x, y, deg=1)
        self.a = float(self.a)
        self.b = float(self.b)
        self.A = float(np.exp(self.b))
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        raw_safe = np.clip(raw, 1e-6, None)
        return self.A * np.power(raw_safe, self.a)

    def formula(self) -> str:
        return f"z = {self.A:.8f} * raw^{self.a:.8f}"


class InterpByDistanceMedianModel(CalibrationModel):
    """
    Builds a lookup table from per-distance medians:
        median(raw at distance d) -> d

    Then interpolates.
    """

    def fit(self, raw: np.ndarray, gt: np.ndarray):
        df = pd.DataFrame({"raw": raw, "gt": gt})
        grouped = df.groupby("gt", as_index=False)["raw"].median()

        x_raw = grouped["raw"].to_numpy(dtype=np.float64)
        y_gt = grouped["gt"].to_numpy(dtype=np.float64)

        order = np.argsort(x_raw)
        x_raw = x_raw[order]
        y_gt = y_gt[order]

        unique_raw = []
        unique_gt = []

        for value in np.unique(x_raw):
            mask = x_raw == value
            unique_raw.append(float(value))
            unique_gt.append(float(np.mean(y_gt[mask])))

        self.x_raw = np.array(unique_raw, dtype=np.float64)
        self.y_gt = np.array(unique_gt, dtype=np.float64)

        if self.x_raw.size < 2:
            raise RuntimeError("Not enough unique raw values for interpolation")

        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return np.interp(raw, self.x_raw, self.y_gt)

    def formula(self) -> str:
        pairs = [f"({x:.4f}->{y:.2f})" for x, y in zip(self.x_raw, self.y_gt)]
        return "interp table: " + ", ".join(pairs)


class IsotonicModel(CalibrationModel):
    """
    Monotonic calibration:
        larger raw should not predict smaller metric depth.

    Requires scikit-learn.
    """

    def fit(self, raw: np.ndarray, gt: np.ndarray):
        try:
            from sklearn.isotonic import IsotonicRegression
        except Exception as exc:
            raise RuntimeError("scikit-learn is not installed") from exc

        self.model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        self.model.fit(raw, gt)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.model.predict(raw)

    def formula(self) -> str:
        return "isotonic increasing calibration"


def calcMetrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    residual = pred - gt
    abs_err = np.abs(residual)

    return {
        "mae_m": float(np.mean(abs_err)),
        "rmse_m": float(np.sqrt(np.mean(residual * residual))),
        "median_abs_m": float(np.median(abs_err)),
        "p90_abs_m": float(np.percentile(abs_err, 90)),
        "max_abs_m": float(np.max(abs_err)),
        "bias_m": float(np.mean(residual)),
    }


def makeModels() -> Dict[str, CalibrationModel]:
    return {
        "constant_scale": ConstantScaleModel(),
        "linear": LinearModel(),
        "quadratic": QuadraticModel(),
        "cubic": CubicModel(),
        "inverse_affine": InverseAffineModel(),
        "power_law": PowerLawModel(),
        "interp_distance_median": InterpByDistanceMedianModel(),
        "isotonic": IsotonicModel(),
    }


def loadCalibrationCsv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = ["gt_m", "small_raw_median"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=required).copy()
    df["gt_m"] = df["gt_m"].astype(float)
    df["small_raw_median"] = df["small_raw_median"].astype(float)

    if "distance_folder" not in df.columns:
        df["distance_folder"] = df["gt_m"].map(lambda x: str(x).replace(".", "_"))

    df = df[np.isfinite(df["gt_m"]) & np.isfinite(df["small_raw_median"])]
    df = df[df["small_raw_median"] > 0]

    if df.empty:
        raise RuntimeError("No valid rows after filtering")

    return df


def fitEvaluateAll(df: pd.DataFrame, out_dir: Path, clip_min: float, clip_max: float):
    raw = df["small_raw_median"].to_numpy(dtype=np.float64)
    gt = df["gt_m"].to_numpy(dtype=np.float64)

    summary_rows = []
    prediction_dfs = {}

    for name, model in makeModels().items():
        try:
            model.fit(raw, gt)
            pred = model.predict(raw)
            pred = np.clip(pred, clip_min, clip_max)

            metrics = calcMetrics(gt, pred)

            row = {
                "model": name,
                "mode": "fit_all",
                "formula": model.formula(),
                **metrics,
            }
            summary_rows.append(row)

            pred_df = df.copy()
            pred_df["model"] = name
            pred_df["pred_m"] = pred
            pred_df["err_m"] = pred_df["pred_m"] - pred_df["gt_m"]
            pred_df["abs_err_m"] = np.abs(pred_df["err_m"])
            prediction_dfs[name] = pred_df

        except Exception as exc:
            summary_rows.append({
                "model": name,
                "mode": "fit_all",
                "formula": f"FAILED: {exc}",
                "mae_m": np.nan,
                "rmse_m": np.nan,
                "median_abs_m": np.nan,
                "p90_abs_m": np.nan,
                "max_abs_m": np.nan,
                "bias_m": np.nan,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "fit_all_summary.csv", index=False)

    all_preds = pd.concat(prediction_dfs.values(), ignore_index=True)
    all_preds.to_csv(out_dir / "fit_all_predictions.csv", index=False)

    return summary, all_preds


def leaveOneDistanceOut(df: pd.DataFrame, out_dir: Path, clip_min: float, clip_max: float):
    all_gt_values = sorted(df["gt_m"].unique(), reverse=True)

    summary_rows = []
    pred_rows = []

    for model_name in makeModels().keys():
        fold_metrics = []

        for heldout_gt in all_gt_values:
            train_df = df[df["gt_m"] != heldout_gt]
            test_df = df[df["gt_m"] == heldout_gt]

            if train_df.empty or test_df.empty:
                continue

            raw_train = train_df["small_raw_median"].to_numpy(dtype=np.float64)
            gt_train = train_df["gt_m"].to_numpy(dtype=np.float64)

            raw_test = test_df["small_raw_median"].to_numpy(dtype=np.float64)
            gt_test = test_df["gt_m"].to_numpy(dtype=np.float64)

            model = makeModels()[model_name]

            try:
                model.fit(raw_train, gt_train)
                pred = model.predict(raw_test)
                pred = np.clip(pred, clip_min, clip_max)

                metrics = calcMetrics(gt_test, pred)
                metrics["heldout_gt_m"] = float(heldout_gt)
                metrics["model"] = model_name
                metrics["formula"] = model.formula()
                metrics["n_test"] = int(len(test_df))
                fold_metrics.append(metrics)

                fold_pred = test_df.copy()
                fold_pred["model"] = model_name
                fold_pred["heldout_gt_m"] = float(heldout_gt)
                fold_pred["pred_m"] = pred
                fold_pred["err_m"] = fold_pred["pred_m"] - fold_pred["gt_m"]
                fold_pred["abs_err_m"] = np.abs(fold_pred["err_m"])

                pred_rows.append(fold_pred)

            except Exception as exc:
                fold_metrics.append({
                    "heldout_gt_m": float(heldout_gt),
                    "model": model_name,
                    "formula": f"FAILED: {exc}",
                    "n_test": int(len(test_df)),
                    "mae_m": np.nan,
                    "rmse_m": np.nan,
                    "median_abs_m": np.nan,
                    "p90_abs_m": np.nan,
                    "max_abs_m": np.nan,
                    "bias_m": np.nan,
                })

        valid = [m for m in fold_metrics if np.isfinite(m["mae_m"])]

        if valid:
            summary_rows.append({
                "model": model_name,
                "mode": "leave_one_distance_out",
                "mae_m": float(np.mean([m["mae_m"] for m in valid])),
                "rmse_m": float(np.mean([m["rmse_m"] for m in valid])),
                "median_abs_m": float(np.mean([m["median_abs_m"] for m in valid])),
                "p90_abs_m": float(np.mean([m["p90_abs_m"] for m in valid])),
                "max_abs_m": float(np.max([m["max_abs_m"] for m in valid])),
                "bias_m": float(np.mean([m["bias_m"] for m in valid])),
                "n_folds": int(len(valid)),
            })
        else:
            summary_rows.append({
                "model": model_name,
                "mode": "leave_one_distance_out",
                "mae_m": np.nan,
                "rmse_m": np.nan,
                "median_abs_m": np.nan,
                "p90_abs_m": np.nan,
                "max_abs_m": np.nan,
                "bias_m": np.nan,
                "n_folds": 0,
            })

        pd.DataFrame(fold_metrics).to_csv(
            out_dir / f"leave_one_distance_folds_{model_name}.csv",
            index=False,
        )

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("mae_m")
    summary.to_csv(out_dir / "leave_one_distance_summary.csv", index=False)

    if pred_rows:
        preds = pd.concat(pred_rows, ignore_index=True)
        preds.to_csv(out_dir / "leave_one_distance_predictions.csv", index=False)
    else:
        preds = pd.DataFrame()

    return summary, preds


def writeBestModelCode(summary: pd.DataFrame, df: pd.DataFrame, out_dir: Path, clip_min: float, clip_max: float):
    best_name = str(summary.iloc[0]["model"])

    raw = df["small_raw_median"].to_numpy(dtype=np.float64)
    gt = df["gt_m"].to_numpy(dtype=np.float64)

    model = makeModels()[best_name]
    model.fit(raw, gt)

    code_path = out_dir / "best_model_function.py"

    if best_name == "constant_scale":
        body = f"z = {model.k:.10f} * raw"
    elif best_name == "linear":
        body = f"z = {model.a:.10f} * raw + {model.b:.10f}"
    elif best_name == "quadratic":
        body = f"z = {model.a:.10f} * raw * raw + {model.b:.10f} * raw + {model.c:.10f}"
    elif best_name == "cubic":
        body = (
            f"z = {model.a:.10f} * raw**3 + {model.b:.10f} * raw**2 "
            f"+ {model.c:.10f} * raw + {model.d:.10f}"
        )
    elif best_name == "inverse_affine":
        body = (
            f"denom = np.clip({model.a:.10f} * raw + {model.b:.10f}, 1e-6, None)\n"
            f"    z = 1.0 / denom"
        )
    elif best_name == "power_law":
        body = f"z = {model.A:.10f} * np.power(np.clip(raw, 1e-6, None), {model.a:.10f})"
    else:
        body = "# This model needs a saved lookup/sklearn object.\n    z = raw"

    with open(code_path, "w") as f:
        f.write("import numpy as np\n\n")
        f.write("def smallRawToMeters(raw):\n")
        f.write("    raw = np.asarray(raw, dtype=np.float32)\n")
        for line in body.splitlines():
            f.write(f"    {line}\n")
        f.write(f"    return np.clip(z, {clip_min:.6f}, {clip_max:.6f}).astype(np.float32)\n")

    return code_path, best_name


def makePlots(df: pd.DataFrame, fit_preds: pd.DataFrame, loo_preds: pd.DataFrame, out_dir: Path):
    raw = df["small_raw_median"].to_numpy(dtype=np.float64)
    gt = df["gt_m"].to_numpy(dtype=np.float64)

    plt.figure()
    plt.scatter(raw, gt, s=12)
    plt.xlabel("small_raw_median")
    plt.ylabel("gt_m")
    plt.title("Raw small depth vs GT")
    plt.grid(True)
    plt.savefig(out_dir / "scatter_raw_vs_gt.png", dpi=160)
    plt.close()

    if not fit_preds.empty:
        model_names = sorted(fit_preds["model"].unique())

        x_grid = np.linspace(np.min(raw), np.max(raw), 200)

        plt.figure()
        plt.scatter(raw, gt, s=10, label="samples")

        for name in model_names:
            model = makeModels()[name]
            try:
                model.fit(raw, gt)
                y_grid = model.predict(x_grid)
                plt.plot(x_grid, y_grid, label=name)
            except Exception:
                pass

        plt.xlabel("small_raw_median")
        plt.ylabel("predicted / GT meters")
        plt.title("Calibration curves")
        plt.grid(True)
        plt.legend()
        plt.savefig(out_dir / "calibration_curves.png", dpi=160)
        plt.close()

    if not loo_preds.empty:
        grouped = loo_preds.groupby("model")["abs_err_m"].mean().sort_values()

        plt.figure()
        plt.bar(grouped.index, grouped.values)
        plt.ylabel("Leave-one-distance MAE [m]")
        plt.title("Calibration model comparison")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "leave_one_distance_mae.png", dpi=160)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clip-min", type=float, default=0.5)
    parser.add_argument("--clip-max", type=float, default=6.0)
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = loadCalibrationCsv(csv_path)

    print(f"Loaded {len(df)} samples")
    print("Distances:", sorted(df["gt_m"].unique(), reverse=True))

    fit_summary, fit_preds = fitEvaluateAll(
        df,
        out_dir,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    loo_summary, loo_preds = leaveOneDistanceOut(
        df,
        out_dir,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    makePlots(df, fit_preds, loo_preds, out_dir)

    code_path, best_name = writeBestModelCode(
        loo_summary,
        df,
        out_dir,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    print("\n=== Fit-all summary ===")
    print(fit_summary.sort_values("mae_m")[["model", "mae_m", "rmse_m", "max_abs_m", "formula"]])

    print("\n=== Leave-one-distance-out summary ===")
    print(loo_summary[["model", "mae_m", "rmse_m", "max_abs_m", "n_folds"]])

    print(f"\nBest validation model: {best_name}")
    print(f"Output dir: {out_dir}")
    print(f"Best model function: {code_path}")


if __name__ == "__main__":
    main()