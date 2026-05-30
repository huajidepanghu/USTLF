# 04_imf_feature_kmeans_grouping.py
# -*- coding: utf-8 -*-
"""
Extract IMF features and perform K-means-based IMF grouping.

Input:
    outputs/iceemdan_rolling/train_iceemdan.npz
    outputs/iceemdan_rolling/val_iceemdan.npz
    outputs/iceemdan_rolling/test_iceemdan.npz

Expected contents in each npz:
    imfs: object array, each element shape [num_imfs, L]
    residuals: object array, each element shape [L]
    origin_indices
    origin_times

Outputs:
    outputs/imf_grouping/kmeans_centroids.csv
    outputs/imf_grouping/group_statistics.csv
    outputs/imf_grouping/train_grouped.npz
    outputs/imf_grouping/val_grouped.npz
    outputs/imf_grouping/test_grouped.npz

Main procedures:
    1. Extract IMF features:
        - energy proportion
        - correlation with original reconstructed signal
        - dominant period
    2. Fit K-means using training IMF features only.
    3. Assign validation/test IMFs using fixed training centroids.
    4. Label clusters as high-, mid-, and low-frequency groups by dominant period.
    5. Reconstruct group-level sequences:
        - high_group
        - mid_group
        - low_group, including residual
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# =========================
# Feature extraction
# =========================

def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation safely.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("safe_corrcoef only accepts one-dimensional arrays.")

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])


def dominant_period(signal: np.ndarray, sampling_interval: float = 1.0) -> float:
    """
    Estimate dominant period from FFT spectrum.

    Args:
        signal:
            One-dimensional IMF signal.
        sampling_interval:
            Sampling interval. For 15-min data, this can be set as 1 step
            because the paper uses relative dominant-period features.

    Returns:
        Dominant period in number of sampling steps.
    """
    x = np.asarray(signal, dtype=np.float64)

    if len(x) < 4:
        return float(len(x))

    x = x - np.mean(x)

    if np.std(x) < 1e-12:
        return float(len(x))

    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(x), d=sampling_interval)

    # Remove zero-frequency component.
    if len(freqs) <= 1:
        return float(len(x))

    freqs = freqs[1:]
    power = power[1:]

    if np.all(power < 1e-12):
        return float(len(x))

    dominant_idx = int(np.argmax(power))
    dominant_freq = float(freqs[dominant_idx])

    if dominant_freq <= 1e-12:
        return float(len(x))

    period = 1.0 / dominant_freq

    # Limit the dominant period to a reasonable range.
    period = float(np.clip(period, 1.0, len(x)))

    return period


def extract_imf_features_for_sample(
    imfs: np.ndarray,
    residual: np.ndarray | None = None
) -> np.ndarray:
    """
    Extract IMF features for one rolling sample.

    Feature vector:
        [energy_proportion, correlation, dominant_period]

    Args:
        imfs:
            Shape [num_imfs, L].
        residual:
            Shape [L]. Optional. Used to reconstruct the original signal
            for correlation calculation.

    Returns:
        features:
            Shape [num_imfs, 3].
    """
    imfs = np.asarray(imfs, dtype=np.float64)

    if imfs.ndim != 2:
        raise ValueError(f"imfs must be a 2D array, got shape {imfs.shape}")

    num_imfs, seq_len = imfs.shape

    if num_imfs == 0:
        return np.empty((0, 3), dtype=np.float32)

    if residual is None or len(residual) != seq_len:
        reconstructed = np.sum(imfs, axis=0)
    else:
        reconstructed = np.sum(imfs, axis=0) + np.asarray(residual, dtype=np.float64)

    imf_energy = np.sum(imfs ** 2, axis=1)
    total_energy = float(np.sum(imf_energy))

    if total_energy < 1e-12:
        energy_prop = np.zeros(num_imfs, dtype=np.float64)
    else:
        energy_prop = imf_energy / total_energy

    corr_vals = np.array(
        [safe_corrcoef(imfs[i], reconstructed) for i in range(num_imfs)],
        dtype=np.float64
    )

    period_vals = np.array(
        [dominant_period(imfs[i]) for i in range(num_imfs)],
        dtype=np.float64
    )

    features = np.stack(
        [energy_prop, corr_vals, period_vals],
        axis=1
    )

    return features.astype(np.float32)


def collect_features_from_split(npz_path: str | Path) -> tuple[np.ndarray, list[dict]]:
    """
    Collect all IMF features from one split.

    Returns:
        all_features:
            Shape [total_num_imfs, 3].
        feature_index:
            List of metadata dictionaries.
    """
    data = np.load(npz_path, allow_pickle=True)

    imfs_obj = data["imfs"]
    residuals_obj = data["residuals"]
    origin_times = data["origin_times"]
    origin_indices = data["origin_indices"]

    all_features = []
    feature_index = []

    for sample_idx in range(len(imfs_obj)):
        imfs = imfs_obj[sample_idx]
        residual = residuals_obj[sample_idx]

        if imfs is None or len(imfs) == 0:
            continue

        features = extract_imf_features_for_sample(imfs, residual)

        for imf_idx in range(features.shape[0]):
            all_features.append(features[imf_idx])
            feature_index.append(
                {
                    "sample_idx": sample_idx,
                    "imf_idx": imf_idx,
                    "origin_idx": int(origin_indices[sample_idx]),
                    "origin_time": str(origin_times[sample_idx])
                }
            )

    if len(all_features) == 0:
        raise ValueError(f"No valid IMF features extracted from {npz_path}")

    all_features = np.vstack(all_features).astype(np.float32)

    return all_features, feature_index


# =========================
# K-means training
# =========================

def fit_kmeans_on_train(
    train_features: np.ndarray,
    n_clusters: int,
    random_seed: int
) -> tuple[StandardScaler, KMeans, np.ndarray, float]:
    """
    Fit feature scaler and K-means using training IMF features only.
    """
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        n_init=20,
        max_iter=500
    )

    labels = kmeans.fit_predict(train_features_scaled)

    if n_clusters > 1 and len(np.unique(labels)) > 1:
        sil_score = float(silhouette_score(train_features_scaled, labels))
    else:
        sil_score = float("nan")

    return scaler, kmeans, labels, sil_score


def assign_semantic_labels(
    centroids_original_scale: np.ndarray
) -> dict[int, str]:
    """
    Assign semantic labels to clusters according to dominant period.

    Shortest dominant period -> high-frequency
    Longest dominant period  -> low-frequency
    Remaining cluster        -> mid-frequency
    """
    dominant_period_col = 2
    periods = centroids_original_scale[:, dominant_period_col]

    sorted_cluster_ids = np.argsort(periods)

    semantic_map = {}

    if len(sorted_cluster_ids) == 3:
        semantic_map[int(sorted_cluster_ids[0])] = "high"
        semantic_map[int(sorted_cluster_ids[1])] = "mid"
        semantic_map[int(sorted_cluster_ids[2])] = "low"
    else:
        # General fallback for M != 3.
        for rank, cluster_id in enumerate(sorted_cluster_ids):
            if rank == 0:
                semantic_map[int(cluster_id)] = "high"
            elif rank == len(sorted_cluster_ids) - 1:
                semantic_map[int(cluster_id)] = "low"
            else:
                semantic_map[int(cluster_id)] = f"mid_{rank}"

    return semantic_map


def save_kmeans_metadata(
    scaler: StandardScaler,
    kmeans: KMeans,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    silhouette: float,
    semantic_map: dict[int, str],
    output_dir: str | Path
) -> None:
    """
    Save centroids, scaler parameters, and cluster statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    feature_names = ["energy", "correlation", "dominant_period"]

    centroid_rows = []
    for cluster_id in range(kmeans.n_clusters):
        row = {
            "cluster_id": cluster_id,
            "semantic_label": semantic_map.get(cluster_id, f"cluster_{cluster_id}")
        }
        for i, name in enumerate(feature_names):
            row[f"centroid_{name}"] = float(centroids_original[cluster_id, i])
        centroid_rows.append(row)

    centroids_df = pd.DataFrame(centroid_rows)
    centroids_df.to_csv(output_dir / "kmeans_centroids.csv", index=False, encoding="utf-8-sig")

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": feature_names,
        "n_clusters": int(kmeans.n_clusters),
        "silhouette_score_train": silhouette,
        "semantic_map": {str(k): v for k, v in semantic_map.items()}
    }

    with open(output_dir / "kmeans_scaler_metadata.json", "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=4)

    cluster_counts = pd.Series(train_labels).value_counts().sort_index()

    stats_rows = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_features = train_features[train_labels == cluster_id]
        row = {
            "cluster_id": cluster_id,
            "semantic_label": semantic_map.get(cluster_id, f"cluster_{cluster_id}"),
            "num_imfs_train": int(cluster_counts.get(cluster_id, 0)),
            "train_feature_energy_mean": float(cluster_features[:, 0].mean()) if len(cluster_features) > 0 else np.nan,
            "train_feature_correlation_mean": float(cluster_features[:, 1].mean()) if len(cluster_features) > 0 else np.nan,
            "train_feature_dominant_period_mean": float(cluster_features[:, 2].mean()) if len(cluster_features) > 0 else np.nan
        }
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_dir / "train_cluster_statistics.csv", index=False, encoding="utf-8-sig")

    print("\n========== K-means Metadata ==========")
    print(f"Train silhouette score: {silhouette:.6f}")
    print(f"Saved: {output_dir / 'kmeans_centroids.csv'}")
    print(f"Saved: {output_dir / 'kmeans_scaler_metadata.json'}")
    print(f"Saved: {output_dir / 'train_cluster_statistics.csv'}")


# =========================
# Group reconstruction
# =========================

def group_one_sample(
    imfs: np.ndarray,
    residual: np.ndarray,
    scaler: StandardScaler,
    kmeans: KMeans,
    semantic_map: dict[int, str]
) -> dict:
    """
    Group IMFs into high-, mid-, and low-frequency reconstructed sequences.

    The residual is added to the low-frequency group.
    """
    imfs = np.asarray(imfs, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)

    if imfs.ndim != 2:
        raise ValueError(f"imfs must be 2D, got shape {imfs.shape}")

    num_imfs, seq_len = imfs.shape

    high_group = np.zeros(seq_len, dtype=np.float32)
    mid_group = np.zeros(seq_len, dtype=np.float32)
    low_group = np.zeros(seq_len, dtype=np.float32)

    if num_imfs == 0:
        low_group += residual
        return {
            "high": high_group,
            "mid": mid_group,
            "low": low_group,
            "labels": np.array([], dtype=np.int64),
            "semantic_labels": np.array([], dtype=object),
            "features": np.empty((0, 3), dtype=np.float32)
        }

    features = extract_imf_features_for_sample(imfs, residual)
    features_scaled = scaler.transform(features)
    labels = kmeans.predict(features_scaled)

    semantic_labels = []

    for imf_idx, cluster_id in enumerate(labels):
        semantic = semantic_map.get(int(cluster_id), f"cluster_{cluster_id}")
        semantic_labels.append(semantic)

        if semantic == "high":
            high_group += imfs[imf_idx]
        elif semantic == "low":
            low_group += imfs[imf_idx]
        else:
            # For M = 3, this is "mid".
            # For M > 3, mid-like groups are also merged into mid_group.
            mid_group += imfs[imf_idx]

    # Residual term mainly contains slowly varying trend information.
    low_group += residual

    return {
        "high": high_group,
        "mid": mid_group,
        "low": low_group,
        "labels": labels.astype(np.int64),
        "semantic_labels": np.array(semantic_labels, dtype=object),
        "features": features.astype(np.float32)
    }


def group_split(
    split_npz_path: str | Path,
    scaler: StandardScaler,
    kmeans: KMeans,
    semantic_map: dict[int, str],
    output_path: str | Path
) -> pd.DataFrame:
    """
    Apply fixed K-means centroids to one split and save grouped sequences.

    Returns:
        split-level statistics dataframe.
    """
    split_npz_path = Path(split_npz_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(split_npz_path, allow_pickle=True)

    imfs_obj = data["imfs"]
    residuals_obj = data["residuals"]
    origin_indices = data["origin_indices"]
    origin_times = data["origin_times"]
    statuses = data["statuses"] if "statuses" in data.files else np.array(["ok"] * len(imfs_obj))

    n_samples = len(imfs_obj)

    high_arr = []
    mid_arr = []
    low_arr = []
    label_obj = np.empty(n_samples, dtype=object)
    semantic_label_obj = np.empty(n_samples, dtype=object)
    feature_obj = np.empty(n_samples, dtype=object)

    stat_rows = []

    for sample_idx in range(n_samples):
        imfs = imfs_obj[sample_idx]
        residual = residuals_obj[sample_idx]

        if statuses[sample_idx] != "ok" or imfs is None or len(imfs) == 0:
            seq_len = len(residual) if residual is not None and len(residual) > 0 else 96
            grouped = {
                "high": np.zeros(seq_len, dtype=np.float32),
                "mid": np.zeros(seq_len, dtype=np.float32),
                "low": np.asarray(residual, dtype=np.float32) if residual is not None and len(residual) > 0 else np.zeros(seq_len, dtype=np.float32),
                "labels": np.array([], dtype=np.int64),
                "semantic_labels": np.array([], dtype=object),
                "features": np.empty((0, 3), dtype=np.float32)
            }
        else:
            grouped = group_one_sample(
                imfs=imfs,
                residual=residual,
                scaler=scaler,
                kmeans=kmeans,
                semantic_map=semantic_map
            )

        high_arr.append(grouped["high"])
        mid_arr.append(grouped["mid"])
        low_arr.append(grouped["low"])

        label_obj[sample_idx] = grouped["labels"]
        semantic_label_obj[sample_idx] = grouped["semantic_labels"]
        feature_obj[sample_idx] = grouped["features"]

        semantic_labels = grouped["semantic_labels"]

        stat_rows.append(
            {
                "sample_idx": sample_idx,
                "origin_idx": int(origin_indices[sample_idx]),
                "origin_time": str(origin_times[sample_idx]),
                "num_imfs": int(len(semantic_labels)),
                "num_high": int(np.sum(semantic_labels == "high")) if len(semantic_labels) > 0 else 0,
                "num_mid": int(np.sum(np.char.startswith(semantic_labels.astype(str), "mid"))) if len(semantic_labels) > 0 else 0,
                "num_low": int(np.sum(semantic_labels == "low")) if len(semantic_labels) > 0 else 0
            }
        )

    high_arr = np.stack(high_arr, axis=0).astype(np.float32)
    mid_arr = np.stack(mid_arr, axis=0).astype(np.float32)
    low_arr = np.stack(low_arr, axis=0).astype(np.float32)

    np.savez_compressed(
        output_path,
        origin_indices=origin_indices,
        origin_times=origin_times,
        high_group=high_arr,
        mid_group=mid_arr,
        low_group=low_arr,
        imf_cluster_labels=label_obj,
        imf_semantic_labels=semantic_label_obj,
        imf_features=feature_obj
    )

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(
        output_path.with_suffix(".statistics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n[Grouped split saved] {output_path}")
    print(f"high_group: {high_arr.shape}, mid_group: {mid_arr.shape}, low_group: {low_arr.shape}")
    print(f"statistics: {output_path.with_suffix('.statistics.csv')}")

    return stats_df


def summarize_group_statistics(
    split_stats: dict[str, pd.DataFrame],
    output_dir: str | Path
) -> None:
    """
    Summarize average number of IMFs assigned to each group.
    """
    output_dir = Path(output_dir)

    rows = []

    for split_name, stats_df in split_stats.items():
        rows.append(
            {
                "split": split_name,
                "avg_num_imfs": float(stats_df["num_imfs"].mean()),
                "avg_num_high": float(stats_df["num_high"].mean()),
                "avg_num_mid": float(stats_df["num_mid"].mean()),
                "avg_num_low": float(stats_df["num_low"].mean())
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "group_statistics_summary.csv", index=False, encoding="utf-8-sig")

    print("\n========== Group Statistics Summary ==========")
    print(summary_df)
    print(f"Saved: {output_dir / 'group_statistics_summary.csv'}")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract IMF features and perform K-means-based IMF grouping."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/iceemdan_rolling",
        help="Directory containing train_iceemdan.npz, val_iceemdan.npz, and test_iceemdan.npz."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/imf_grouping",
        help="Directory to save K-means metadata and grouped sequences."
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=3,
        help="Number of K-means clusters. Default is 3 for high/mid/low groups."
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for K-means."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = input_dir / "train_iceemdan.npz"
    val_path = input_dir / "val_iceemdan.npz"
    test_path = input_dir / "test_iceemdan.npz"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # 1. Collect training IMF features.
    print("\n========== Extract Training IMF Features ==========")
    train_features, train_feature_index = collect_features_from_split(train_path)
    print(f"Training IMF features shape: {train_features.shape}")

    train_feature_index_df = pd.DataFrame(train_feature_index)
    train_feature_df = pd.DataFrame(
        train_features,
        columns=["energy", "correlation", "dominant_period"]
    )
    train_feature_all_df = pd.concat([train_feature_index_df, train_feature_df], axis=1)
    train_feature_all_df.to_csv(
        output_dir / "train_imf_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 2. Fit K-means using training features only.
    print("\n========== Fit K-means on Training Features ==========")
    scaler, kmeans, train_labels, sil_score = fit_kmeans_on_train(
        train_features=train_features,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed
    )

    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    semantic_map = assign_semantic_labels(centroids_original)

    save_kmeans_metadata(
        scaler=scaler,
        kmeans=kmeans,
        train_features=train_features,
        train_labels=train_labels,
        silhouette=sil_score,
        semantic_map=semantic_map,
        output_dir=output_dir
    )

    # 3. Group train/val/test using fixed training centroids.
    print("\n========== Apply Fixed K-means Centroids to All Splits ==========")

    split_stats = {}

    split_stats["train"] = group_split(
        split_npz_path=train_path,
        scaler=scaler,
        kmeans=kmeans,
        semantic_map=semantic_map,
        output_path=output_dir / "train_grouped.npz"
    )

    split_stats["val"] = group_split(
        split_npz_path=val_path,
        scaler=scaler,
        kmeans=kmeans,
        semantic_map=semantic_map,
        output_path=output_dir / "val_grouped.npz"
    )

    split_stats["test"] = group_split(
        split_npz_path=test_path,
        scaler=scaler,
        kmeans=kmeans,
        semantic_map=semantic_map,
        output_path=output_dir / "test_grouped.npz"
    )

    summarize_group_statistics(
        split_stats=split_stats,
        output_dir=output_dir
    )

    print("\nAll IMF feature extraction and K-means grouping steps are completed.")


if __name__ == "__main__":
    main()