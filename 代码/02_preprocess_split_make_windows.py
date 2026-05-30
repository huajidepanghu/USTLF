# 02_preprocess_split_make_windows.py
# -*- coding: utf-8 -*-
"""
Preprocess Elia load data, build temporal covariates, split datasets,
normalize features, and construct sliding-window samples.

Input:
    outputs/elia_load_raw_standard.csv

Expected input columns:
    datetime
    resolution_code
    load_mw

Outputs:
    outputs/elia_train.csv
    outputs/elia_val.csv
    outputs/elia_test.csv
    outputs/elia_full_processed.csv
    outputs/normalization_params.csv
    outputs/normalization_params.json

    outputs/windows/H1/train_H1.npz
    outputs/windows/H1/val_H1.npz
    outputs/windows/H1/test_H1.npz
    outputs/windows/H4/train_H4.npz
    outputs/windows/H4/val_H4.npz
    outputs/windows/H4/test_H4.npz
    outputs/windows/H8/train_H8.npz
    outputs/windows/H8/val_H8.npz
    outputs/windows/H8/test_H8.npz
    outputs/windows/H12/train_H12.npz
    outputs/windows/H12/val_H12.npz
    outputs/windows/H12/test_H12.npz

Main procedures:
    1. load standardized Elia load data
    2. complete 15-min timestamp continuity
    3. handle missing load values
    4. build temporal covariates
    5. split train/validation/test sets chronologically
    6. normalize features using training-set statistics only
    7. construct sliding-window samples with L = 96 and H = 1, 4, 8, 12
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


# =========================
# Split dates used in paper
# =========================

TRAIN_START = "2025-01-01 00:00:00"
TRAIN_END = "2025-09-13 23:45:00"

VAL_START = "2025-09-14 00:00:00"
VAL_END = "2025-10-19 23:45:00"

TEST_START = "2025-10-20 00:00:00"
TEST_END = "2025-12-31 23:45:00"


# =========================
# Feature settings
# =========================

RAW_TARGET_COL = "load_mw"
NORM_TARGET_COL = "load_mw_norm"

FEATURE_COLUMNS = [
    "load_mw",
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

NORMALIZED_FEATURE_COLUMNS = [
    "load_mw_norm",
    "time_of_day_sin_norm",
    "time_of_day_cos_norm",
    "day_of_week_sin_norm",
    "day_of_week_cos_norm",
]


# =========================
# Data loading
# =========================

def load_standard_data(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    required_cols = ["datetime", "resolution_code", "load_mw"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns are: {list(df.columns)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# =========================
# Preprocessing
# =========================

def complete_15min_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a complete 15-minute timestamp index and align load values to it.
    Missing load values are handled later.
    """
    df = df.copy()

    df = df.drop_duplicates(subset=["datetime"], keep="first")
    df = df.set_index("datetime").sort_index()

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="15min"
    )

    df = df.reindex(full_index)
    df.index.name = "datetime"
    df = df.reset_index()

    if "resolution_code" in df.columns:
        df["resolution_code"] = df["resolution_code"].fillna("PT15M")
    else:
        df["resolution_code"] = "PT15M"

    return df


def handle_missing_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing load values.

    - Missing values inside the sequence are linearly interpolated.
    - Remaining leading/trailing missing values are forward/backward filled.
    """
    df = df.copy()

    missing_before = df["load_mw"].isna().sum()

    df["load_mw"] = df["load_mw"].interpolate(method="linear")
    df["load_mw"] = df["load_mw"].ffill().bfill()

    missing_after = df["load_mw"].isna().sum()

    print(f"[Missing load values] Before: {missing_before}, After: {missing_after}")

    return df


def build_temporal_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build sine-cosine temporal covariates.

    Features:
        time_of_day_sin
        time_of_day_cos
        day_of_week_sin
        day_of_week_cos
    """
    df = df.copy()

    dt = pd.to_datetime(df["datetime"])

    minutes_of_day = dt.dt.hour * 60 + dt.dt.minute
    day_of_week = dt.dt.dayofweek

    df["time_of_day_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440.0)
    df["time_of_day_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440.0)

    df["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    return df


# =========================
# Split and normalize
# =========================

def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically according to the dates reported in the manuscript.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    train_df = df[
        (df["datetime"] >= pd.Timestamp(TRAIN_START)) &
        (df["datetime"] <= pd.Timestamp(TRAIN_END))
    ].copy()

    val_df = df[
        (df["datetime"] >= pd.Timestamp(VAL_START)) &
        (df["datetime"] <= pd.Timestamp(VAL_END))
    ].copy()

    test_df = df[
        (df["datetime"] >= pd.Timestamp(TEST_START)) &
        (df["datetime"] <= pd.Timestamp(TEST_END))
    ].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "One of the dataset splits is empty. Please check the input data time range "
            "and the predefined split dates."
        )

    print("\n========== Dataset Split Report ==========")
    print(f"Train: {train_df['datetime'].min()} -> {train_df['datetime'].max()}, n={len(train_df)}")
    print(f"Val:   {val_df['datetime'].min()} -> {val_df['datetime'].max()}, n={len(val_df)}")
    print(f"Test:  {test_df['datetime'].min()} -> {test_df['datetime'].max()}, n={len(test_df)}")

    return train_df, val_df, test_df


def fit_minmax_params(train_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Fit min-max normalization parameters using the training set only.
    """
    params = {}

    for col in feature_cols:
        col_min = float(train_df[col].min())
        col_max = float(train_df[col].max())

        if np.isclose(col_max, col_min):
            print(f"[Warning] Feature {col} has nearly constant values in the training set.")

        params[col] = {
            "min": col_min,
            "max": col_max
        }

    return params


def apply_minmax_normalization(
    df: pd.DataFrame,
    params: dict,
    feature_cols: list[str]
) -> pd.DataFrame:
    """
    Apply min-max normalization using parameters fitted only on the training set.
    """
    df = df.copy()

    for col in feature_cols:
        col_min = params[col]["min"]
        col_max = params[col]["max"]
        denom = col_max - col_min

        out_col = col + "_norm"

        if np.isclose(denom, 0.0):
            df[out_col] = 0.0
        else:
            df[out_col] = (df[col] - col_min) / denom

    return df


# =========================
# Sliding-window construction
# =========================

def make_sliding_windows(
    df: pd.DataFrame,
    input_len: int,
    horizon: int,
    feature_cols: list[str],
    target_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct sliding-window samples.

    X shape:
        [num_samples, input_len, num_features]

    y shape:
        [num_samples, horizon]

    time_index shape:
        [num_samples]
        The timestamp of the forecasting origin, i.e., the last input time step.
    """
    df = df.copy().reset_index(drop=True)

    feature_array = df[feature_cols].to_numpy(dtype=np.float32)
    target_array = df[target_col].to_numpy(dtype=np.float32)
    time_array = df["datetime"].astype(str).to_numpy()

    X_list = []
    y_list = []
    origin_time_list = []

    total_len = len(df)

    max_start = total_len - input_len - horizon + 1

    if max_start <= 0:
        raise ValueError(
            f"Not enough samples to build windows. "
            f"total_len={total_len}, input_len={input_len}, horizon={horizon}"
        )

    for start_idx in range(max_start):
        input_start = start_idx
        input_end = start_idx + input_len
        target_start = input_end
        target_end = input_end + horizon

        X = feature_array[input_start:input_end, :]
        y = target_array[target_start:target_end]

        origin_time = time_array[input_end - 1]

        X_list.append(X)
        y_list.append(y)
        origin_time_list.append(origin_time)

    X_all = np.stack(X_list, axis=0)
    y_all = np.stack(y_list, axis=0)
    origin_times = np.array(origin_time_list)

    return X_all, y_all, origin_times


def save_windows(
    split_name: str,
    df: pd.DataFrame,
    output_dir: str | Path,
    input_len: int,
    horizons: list[int],
    feature_cols: list[str],
    target_col: str
) -> None:
    """
    Save sliding-window samples for each forecasting horizon.
    """
    output_dir = Path(output_dir)

    for horizon in horizons:
        X, y, origin_times = make_sliding_windows(
            df=df,
            input_len=input_len,
            horizon=horizon,
            feature_cols=feature_cols,
            target_col=target_col
        )

        horizon_dir = output_dir / "windows" / f"H{horizon}"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        output_path = horizon_dir / f"{split_name}_H{horizon}.npz"

        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            origin_times=origin_times,
            feature_cols=np.array(feature_cols),
            target_col=np.array([target_col]),
            input_len=np.array([input_len]),
            horizon=np.array([horizon])
        )

        print(
            f"[Window saved] {output_path} | "
            f"X={X.shape}, y={y.shape}, origins={origin_times.shape}"
        )


# =========================
# Save CSV outputs
# =========================

def save_processed_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    params: dict,
    output_dir: str | Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "elia_train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(output_dir / "elia_val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(output_dir / "elia_test.csv", index=False, encoding="utf-8-sig")
    full_df.to_csv(output_dir / "elia_full_processed.csv", index=False, encoding="utf-8-sig")

    params_df = pd.DataFrame(
        [
            {"feature": feature, "min": values["min"], "max": values["max"]}
            for feature, values in params.items()
        ]
    )
    params_df.to_csv(output_dir / "normalization_params.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "normalization_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)

    print("\n========== Saved Processed CSV Files ==========")
    print(output_dir / "elia_train.csv")
    print(output_dir / "elia_val.csv")
    print(output_dir / "elia_test.csv")
    print(output_dir / "elia_full_processed.csv")
    print(output_dir / "normalization_params.csv")
    print(output_dir / "normalization_params.json")


def quality_report(df: pd.DataFrame, name: str) -> None:
    print(f"\n========== Quality Report: {name} ==========")
    print(f"Records: {len(df)}")
    print(f"Start: {df['datetime'].min()}")
    print(f"End:   {df['datetime'].max()}")
    print(f"Missing load_mw: {df['load_mw'].isna().sum()}")
    print(f"Mean load_mw: {df['load_mw'].mean():.4f}")
    print(f"Std load_mw:  {df['load_mw'].std():.4f}")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Elia load data, build temporal covariates, split datasets, "
            "normalize features, and construct sliding-window samples."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default="outputs/elia_load_raw_standard.csv",
        help="Path to standardized Elia load data generated by 01_load_data.py."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save processed files and window samples."
    )

    parser.add_argument(
        "--input_len",
        type=int,
        default=96,
        help="Input window length L."
    )

    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 4, 8, 12],
        help="Forecasting horizons H."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load standardized data.
    df = load_standard_data(args.input)

    # 2. Complete timestamp continuity and handle missing load values.
    df = complete_15min_index(df)
    df = handle_missing_load(df)

    # 3. Build temporal covariates.
    df = build_temporal_covariates(df)

    # 4. Split chronologically.
    train_df, val_df, test_df = split_by_time(df)

    # 5. Fit normalization parameters using training set only.
    params = fit_minmax_params(train_df, FEATURE_COLUMNS)

    # 6. Apply normalization to train/val/test using training-set statistics.
    train_df = apply_minmax_normalization(train_df, params, FEATURE_COLUMNS)
    val_df = apply_minmax_normalization(val_df, params, FEATURE_COLUMNS)
    test_df = apply_minmax_normalization(test_df, params, FEATURE_COLUMNS)

    # 7. Concatenate processed full dataset.
    full_df = pd.concat(
        [train_df, val_df, test_df],
        axis=0
    ).sort_values("datetime").reset_index(drop=True)

    # 8. Print reports.
    quality_report(train_df, "Train")
    quality_report(val_df, "Validation")
    quality_report(test_df, "Test")

    # 9. Save processed CSV files and normalization parameters.
    save_processed_csvs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        full_df=full_df,
        params=params,
        output_dir=args.output_dir
    )

    # 10. Save sliding-window samples for each horizon.
    save_windows(
        split_name="train",
        df=train_df,
        output_dir=args.output_dir,
        input_len=args.input_len,
        horizons=args.horizons,
        feature_cols=NORMALIZED_FEATURE_COLUMNS,
        target_col=NORM_TARGET_COL
    )

    save_windows(
        split_name="val",
        df=val_df,
        output_dir=args.output_dir,
        input_len=args.input_len,
        horizons=args.horizons,
        feature_cols=NORMALIZED_FEATURE_COLUMNS,
        target_col=NORM_TARGET_COL
    )

    save_windows(
        split_name="test",
        df=test_df,
        output_dir=args.output_dir,
        input_len=args.input_len,
        horizons=args.horizons,
        feature_cols=NORMALIZED_FEATURE_COLUMNS,
        target_col=NORM_TARGET_COL
    )

    print("\nAll preprocessing, splitting, normalization, and window construction steps are completed.")


if __name__ == "__main__":
    main()