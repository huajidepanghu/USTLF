# 03_rolling_iceemdan_decomposition.py
# -*- coding: utf-8 -*-
"""
Rolling causal ICEEMDAN decomposition with CPU parallelization.

Input:
    outputs/elia_full_processed.csv

Expected input columns:
    datetime
    load_mw_norm

Outputs:
    outputs/iceemdan_rolling/train_iceemdan.npz
    outputs/iceemdan_rolling/val_iceemdan.npz
    outputs/iceemdan_rolling/test_iceemdan.npz

Main procedures:
    1. Load processed Elia load data.
    2. For each forecasting origin, use only historical observations.
    3. Apply ICEEMDAN on the causal decomposition window W_dec = 288.
    4. Save rolling IMF results for later feature extraction and K-means grouping.
    5. Use multiprocessing on CPU to accelerate decomposition.

Important:
    - This script performs rolling decomposition only.
    - IMF feature extraction and K-means grouping are handled in the next module.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass

try:
    from PyEMD import EMD
except ImportError as exc:
    raise ImportError(
        "PyEMD is required for this script. Please install it with:\n"
        "    pip install EMD-signal\n"
    ) from exc


# =========================
# Default split dates
# =========================

TRAIN_START = "2025-01-01 00:00:00"
TRAIN_END = "2025-09-13 23:45:00"

VAL_START = "2025-09-14 00:00:00"
VAL_END = "2025-10-19 23:45:00"

TEST_START = "2025-10-20 00:00:00"
TEST_END = "2025-12-31 23:45:00"


# =========================
# ICEEMDAN configuration
# =========================

@dataclass
class ICEEMDANConfig:
    ensemble_size: int = 100
    noise_width: float = 0.2
    max_imfs: int | None = None
    random_seed: int = 42
    spline_kind: str = "cubic"


def _standardize_signal(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Standardize signal to improve decomposition stability.

    Returns:
        x_std: standardized signal
        mean: original mean
        std: original std
    """
    mean = float(np.mean(x))
    std = float(np.std(x))

    if std < 1e-12:
        return x - mean, mean, 1.0

    return (x - mean) / std, mean, std


def _restore_signal(x_std: np.ndarray, mean: float, std: float) -> np.ndarray:
    return x_std * std + mean


def _get_first_imf(signal: np.ndarray, emd: EMD) -> np.ndarray:
    """
    Extract the first IMF using EMD.
    If EMD fails to extract an IMF, return zeros.
    """
    imfs = emd.emd(signal)

    if imfs is None or len(imfs) == 0:
        return np.zeros_like(signal)

    return imfs[0]


def iceemdan_decompose(
    signal: np.ndarray,
    config: ICEEMDANConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    A practical ICEEMDAN implementation based on repeated EMD operations.

    Args:
        signal:
            One-dimensional input sequence.
        config:
            ICEEMDANConfig.

    Returns:
        imfs:
            Array with shape [num_imfs, signal_length].
        residual:
            Array with shape [signal_length].

    Notes:
        This implementation follows the noise-assisted iterative decomposition idea
        of ICEEMDAN. It is designed for reproducible rolling decomposition in the
        forecasting pipeline.
    """
    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError(f"Input signal must be one-dimensional, got shape {x.shape}")

    if len(x) < 8:
        raise ValueError("Input signal is too short for EMD-based decomposition.")

    x_std, mean, std = _standardize_signal(x)

    rng = np.random.default_rng(config.random_seed)

    emd = EMD()
    emd.spline_kind = config.spline_kind

    n = len(x_std)
    ensemble_size = config.ensemble_size
    noise_width = config.noise_width

    # Pre-generate white noise and its first IMF.
    noises = rng.standard_normal(size=(ensemble_size, n))
    noise_imf_1 = np.zeros_like(noises)

    for i in range(ensemble_size):
        noise_imf_1[i] = _get_first_imf(noises[i], emd)

    # First IMF estimation.
    local_means = np.zeros((ensemble_size, n), dtype=np.float64)

    for i in range(ensemble_size):
        noisy_signal = x_std + noise_width * noise_imf_1[i]
        imfs_i = emd.emd(noisy_signal)

        if imfs_i is None or len(imfs_i) == 0:
            local_means[i] = noisy_signal
        else:
            local_means[i] = noisy_signal - imfs_i[0]

    residual = np.mean(local_means, axis=0)
    first_imf = x_std - residual

    imfs = [first_imf]

    max_possible_imfs = int(np.floor(np.log2(n))) + 1
    if config.max_imfs is not None:
        max_possible_imfs = min(max_possible_imfs, config.max_imfs)

    current_residual = residual.copy()

    # Iteratively extract subsequent IMFs.
    for k in range(1, max_possible_imfs):
        # Stop if residual becomes nearly monotonic or too small.
        if np.std(current_residual) < 1e-10:
            break

        # Generate noise modes of corresponding order.
        local_means_k = np.zeros((ensemble_size, n), dtype=np.float64)

        for i in range(ensemble_size):
            noise_imfs = emd.emd(noises[i])

            if noise_imfs is None or len(noise_imfs) <= k:
                noise_mode = np.zeros(n, dtype=np.float64)
            else:
                noise_mode = noise_imfs[k]

            noisy_residual = current_residual + noise_width * noise_mode
            imfs_i = emd.emd(noisy_residual)

            if imfs_i is None or len(imfs_i) == 0:
                local_means_k[i] = noisy_residual
            else:
                local_means_k[i] = noisy_residual - imfs_i[0]

        next_residual = np.mean(local_means_k, axis=0)
        next_imf = current_residual - next_residual

        if np.std(next_imf) < 1e-10:
            break

        imfs.append(next_imf)
        current_residual = next_residual

    imfs_arr = np.stack(imfs, axis=0)

    # Restore scale.
    imfs_arr = imfs_arr * std
    residual_restored = _restore_signal(current_residual, mean, std)

    return imfs_arr.astype(np.float32), residual_restored.astype(np.float32)


# =========================
# Rolling decomposition
# =========================

def build_forecast_origins(
    df: pd.DataFrame,
    split_start: str,
    split_end: str,
    input_len: int,
    horizon: int,
    decomposition_window: int
) -> list[int]:
    """
    Build forecast origin indices for one split.

    Forecast origin is defined as the last input time step.
    To ensure each sample has enough future values for horizon H,
    origin index must satisfy:
        origin_idx + horizon < len(df)

    To ensure causal decomposition window:
        origin_idx - decomposition_window + 1 >= 0
    """
    datetimes = pd.to_datetime(df["datetime"])
    start_ts = pd.Timestamp(split_start)
    end_ts = pd.Timestamp(split_end)

    split_indices = np.where((datetimes >= start_ts) & (datetimes <= end_ts))[0]

    if len(split_indices) == 0:
        raise ValueError(f"No data found for split range: {split_start} to {split_end}")

    split_start_idx = int(split_indices[0])
    split_end_idx = int(split_indices[-1])

    origin_indices = []

    # The first origin must have enough input length within the split.
    first_origin = split_start_idx + input_len - 1
    last_origin = split_end_idx - horizon

    for origin_idx in range(first_origin, last_origin + 1):
        if origin_idx - decomposition_window + 1 < 0:
            continue
        origin_indices.append(origin_idx)

    return origin_indices


def _decompose_one_task(args: tuple) -> dict:
    """
    Worker function for multiprocessing.
    """
    (
        task_id,
        origin_idx,
        signal_values,
        datetimes,
        decomposition_window,
        input_len,
        config_dict
    ) = args

    config = ICEEMDANConfig(**config_dict)

    window_start = origin_idx - decomposition_window + 1
    window_end = origin_idx + 1

    causal_window = signal_values[window_start:window_end]

    # Make random seed sample-specific for reproducibility across parallel workers.
    config.random_seed = int(config.random_seed + task_id)

    try:
        imfs, residual = iceemdan_decompose(causal_window, config)

        # Only keep the most recent input_len part for later forecasting.
        imfs_recent = imfs[:, -input_len:]
        residual_recent = residual[-input_len:]

        status = "ok"
        error_msg = ""

    except Exception as exc:
        imfs_recent = np.empty((0, input_len), dtype=np.float32)
        residual_recent = np.empty((input_len,), dtype=np.float32)
        status = "failed"
        error_msg = str(exc)

    return {
        "task_id": task_id,
        "origin_idx": origin_idx,
        "origin_time": str(datetimes[origin_idx]),
        "window_start_time": str(datetimes[window_start]),
        "window_end_time": str(datetimes[origin_idx]),
        "imfs": imfs_recent,
        "residual": residual_recent,
        "status": status,
        "error_msg": error_msg
    }


def run_parallel_decomposition(
    df: pd.DataFrame,
    origin_indices: list[int],
    split_name: str,
    output_dir: str | Path,
    decomposition_window: int,
    input_len: int,
    config: ICEEMDANConfig,
    num_workers: int
) -> None:
    """
    Run rolling ICEEMDAN decomposition in parallel and save results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_values = df["load_mw_norm"].to_numpy(dtype=np.float64)
    datetimes = pd.to_datetime(df["datetime"]).to_numpy()

    config_dict = config.__dict__.copy()

    tasks = [
        (
            task_id,
            origin_idx,
            signal_values,
            datetimes,
            decomposition_window,
            input_len,
            config_dict
        )
        for task_id, origin_idx in enumerate(origin_indices)
    ]

    print(f"\n========== Rolling ICEEMDAN: {split_name} ==========")
    print(f"Number of origins: {len(tasks)}")
    print(f"CPU workers: {num_workers}")
    print(f"W_dec: {decomposition_window}, L: {input_len}")
    print(f"Ensemble size: {config.ensemble_size}, noise width: {config.noise_width}")

    if len(tasks) == 0:
        raise ValueError(f"No valid forecast origins for split: {split_name}")

    if num_workers <= 1:
        results = [_decompose_one_task(task) for task in tasks]
    else:
        with Pool(processes=num_workers) as pool:
            results = list(pool.imap(_decompose_one_task, tasks, chunksize=1))

    # Keep original order.
    results = sorted(results, key=lambda x: x["task_id"])

    failed = [r for r in results if r["status"] != "ok"]
    print(f"Completed: {len(results) - len(failed)} / {len(results)}")
    print(f"Failed:    {len(failed)} / {len(results)}")

    if failed:
        warnings.warn(
            f"{len(failed)} decomposition tasks failed in split {split_name}. "
            "Failed samples will contain empty IMF arrays."
        )

    # Save variable-length IMF arrays as object arrays.
    origin_indices_arr = np.array([r["origin_idx"] for r in results], dtype=np.int64)
    origin_times_arr = np.array([r["origin_time"] for r in results])
    window_start_times_arr = np.array([r["window_start_time"] for r in results])
    window_end_times_arr = np.array([r["window_end_time"] for r in results])
    statuses_arr = np.array([r["status"] for r in results])
    errors_arr = np.array([r["error_msg"] for r in results])

    imfs_obj = np.empty(len(results), dtype=object)
    residual_obj = np.empty(len(results), dtype=object)

    for i, r in enumerate(results):
        imfs_obj[i] = r["imfs"]
        residual_obj[i] = r["residual"]

    output_path = output_dir / f"{split_name}_iceemdan.npz"

    np.savez_compressed(
        output_path,
        origin_indices=origin_indices_arr,
        origin_times=origin_times_arr,
        window_start_times=window_start_times_arr,
        window_end_times=window_end_times_arr,
        imfs=imfs_obj,
        residuals=residual_obj,
        statuses=statuses_arr,
        errors=errors_arr,
        decomposition_window=np.array([decomposition_window]),
        input_len=np.array([input_len]),
        ensemble_size=np.array([config.ensemble_size]),
        noise_width=np.array([config.noise_width]),
        random_seed=np.array([config.random_seed])
    )

    print(f"Saved rolling ICEEMDAN results to: {output_path}")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling causal ICEEMDAN decomposition with CPU parallelization."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="outputs/elia_full_processed.csv",
        help="Path to processed full dataset."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/iceemdan_rolling",
        help="Directory to save rolling ICEEMDAN results."
    )

    parser.add_argument(
        "--decomposition_window",
        type=int,
        default=288,
        help="Rolling decomposition window length W_dec."
    )

    parser.add_argument(
        "--input_len",
        type=int,
        default=96,
        help="Input window length L."
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forecasting horizon H used to build valid forecast origins."
    )

    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=100,
        help="Number of ICEEMDAN ensemble realizations."
    )

    parser.add_argument(
        "--noise_width",
        type=float,
        default=0.2,
        help="Noise amplitude coefficient."
    )

    parser.add_argument(
        "--max_imfs",
        type=int,
        default=None,
        help="Maximum number of IMFs. Default: automatically determined."
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(cpu_count() - 1, 1),
        help="Number of CPU worker processes."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to decompose."
    )

    return parser.parse_args()


def load_processed_data(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    required_cols = ["datetime", "load_mw_norm"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns are: {list(df.columns)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["load_mw_norm"] = pd.to_numeric(df["load_mw_norm"], errors="coerce")

    df = df.dropna(subset=["datetime", "load_mw_norm"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def main() -> None:
    args = parse_args()

    df = load_processed_data(args.input)

    config = ICEEMDANConfig(
        ensemble_size=args.ensemble_size,
        noise_width=args.noise_width,
        max_imfs=args.max_imfs,
        random_seed=args.random_seed
    )

    split_ranges = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (VAL_START, VAL_END),
        "test": (TEST_START, TEST_END)
    }

    if args.split == "all":
        selected_splits = ["train", "val", "test"]
    else:
        selected_splits = [args.split]

    for split_name in selected_splits:
        split_start, split_end = split_ranges[split_name]

        origin_indices = build_forecast_origins(
            df=df,
            split_start=split_start,
            split_end=split_end,
            input_len=args.input_len,
            horizon=args.horizon,
            decomposition_window=args.decomposition_window
        )

        run_parallel_decomposition(
            df=df,
            origin_indices=origin_indices,
            split_name=split_name,
            output_dir=args.output_dir,
            decomposition_window=args.decomposition_window,
            input_len=args.input_len,
            config=config,
            num_workers=args.num_workers
        )

    print("\nAll rolling ICEEMDAN decomposition tasks are completed.")


if __name__ == "__main__":
    main()