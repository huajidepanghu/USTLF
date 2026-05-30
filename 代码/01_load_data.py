# 01_load_data.py
# -*- coding: utf-8 -*-
"""
Load raw Elia grid load data and convert it into a standard time-series format.

Input:
    data/ods003.csv

Expected raw columns:
    Datetime
    Resolution code
    Elia Grid Load

Output:
    outputs/elia_load_raw_standard.csv

This script only performs:
    1. raw CSV reading
    2. field selection
    3. timestamp parsing
    4. PT15M filtering
    5. chronological sorting
    6. basic data-quality checks
"""

from pathlib import Path
import argparse
import pandas as pd


def load_elia_data(input_path: str | Path) -> pd.DataFrame:
    """
    Load raw Elia CSV data.

    The Elia Open Data file is separated by semicolons.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(
        input_path,
        sep=";",
        encoding="utf-8-sig"
    )

    required_cols = ["Datetime", "Resolution code", "Elia Grid Load"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns are: {list(df.columns)}"
        )

    return df


def standardize_elia_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize raw Elia data into a clean time-series format.

    Output columns:
        datetime
        resolution_code
        load_mw
    """
    df = df.copy()

    # Keep only the required fields.
    df = df[["Datetime", "Resolution code", "Elia Grid Load"]]

    # Rename columns for easier processing in subsequent modules.
    df = df.rename(
        columns={
            "Datetime": "datetime",
            "Resolution code": "resolution_code",
            "Elia Grid Load": "load_mw"
        }
    )

    # Parse timestamp. The original data include timezone information.
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Convert load values to numeric.
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")

    # Remove rows with invalid timestamps or invalid load values.
    before_drop = len(df)
    df = df.dropna(subset=["datetime", "load_mw"])
    after_drop = len(df)

    if before_drop != after_drop:
        print(f"[Warning] Dropped {before_drop - after_drop} rows with invalid datetime or load values.")

    # Keep only 15-minute resolution records.
    df = df[df["resolution_code"] == "PT15M"].copy()

    # Sort chronologically.
    df = df.sort_values("datetime").reset_index(drop=True)

    # Remove duplicated timestamps if any.
    duplicated_count = df["datetime"].duplicated().sum()
    if duplicated_count > 0:
        print(f"[Warning] Found {duplicated_count} duplicated timestamps. Keeping the first occurrence.")
        df = df.drop_duplicates(subset=["datetime"], keep="first").reset_index(drop=True)

    return df


def basic_quality_report(df: pd.DataFrame) -> None:
    """
    Print a basic data-quality report.
    """
    print("\n========== Basic Data Quality Report ==========")
    print(f"Number of records: {len(df)}")

    if len(df) == 0:
        print("[Error] Empty dataframe after preprocessing.")
        return

    print(f"Start time: {df['datetime'].min()}")
    print(f"End time:   {df['datetime'].max()}")
    print(f"Resolution codes: {df['resolution_code'].unique().tolist()}")

    print("\nLoad statistics:")
    print(df["load_mw"].describe())

    missing_load = df["load_mw"].isna().sum()
    duplicated_time = df["datetime"].duplicated().sum()

    print(f"\nMissing load values: {missing_load}")
    print(f"Duplicated timestamps: {duplicated_time}")

    # Check expected 15-minute interval continuity.
    time_diff = df["datetime"].diff().dropna()
    abnormal_intervals = time_diff[time_diff != pd.Timedelta(minutes=15)]

    print(f"Abnormal time intervals: {len(abnormal_intervals)}")
    if len(abnormal_intervals) > 0:
        print("[Warning] Examples of abnormal intervals:")
        print(abnormal_intervals.head())


def save_standard_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save standardized Elia load data.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nStandardized data saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and standardize Elia grid load data."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/ods003.csv",
        help="Path to the raw Elia CSV file."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/elia_load_raw_standard.csv",
        help="Path to save the standardized output CSV file."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df = load_elia_data(args.input)
    standard_df = standardize_elia_data(raw_df)

    basic_quality_report(standard_df)
    save_standard_data(standard_df, args.output)


if __name__ == "__main__":
    main()