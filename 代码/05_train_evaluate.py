# 05_train_evaluate.py
# -*- coding: utf-8 -*-
"""
Train and evaluate the proposed frequency-aware forecasting model.

This script combines:
    10. training module
    11. testing and evaluation module

Inputs:
    outputs/imf_grouping/train_grouped.npz
    outputs/imf_grouping/val_grouped.npz
    outputs/imf_grouping/test_grouped.npz

    outputs/windows/H{H}/train_H{H}.npz
    outputs/windows/H{H}/val_H{H}.npz
    outputs/windows/H{H}/test_H{H}.npz

    outputs/normalization_params.json
    models.py

Outputs:
    outputs/training/H{H}/best_model.pt
    outputs/training/H{H}/training_log.csv
    outputs/training/H{H}/test_predictions.csv
    outputs/training/H{H}/test_metrics.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import build_model


# ============================================================
# Reproducibility
# ============================================================

def set_random_seed(seed: int = 42) -> None:
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================

class GroupedLoadDataset(Dataset):
    """
    Dataset for grouped high/mid/low modal sequences.

    Each sample:
        high_group: [L]
        mid_group:  [L]
        low_group:  [L]
        y:          [H]
    """

    def __init__(
        self,
        grouped_npz_path: str | Path,
        window_npz_path: str | Path,
    ):
        grouped_npz_path = Path(grouped_npz_path)
        window_npz_path = Path(window_npz_path)

        if not grouped_npz_path.exists():
            raise FileNotFoundError(f"Grouped file not found: {grouped_npz_path}")

        if not window_npz_path.exists():
            raise FileNotFoundError(f"Window file not found: {window_npz_path}")

        grouped_data = np.load(grouped_npz_path, allow_pickle=True)
        window_data = np.load(window_npz_path, allow_pickle=True)

        self.high_group = grouped_data["high_group"].astype(np.float32)
        self.mid_group = grouped_data["mid_group"].astype(np.float32)
        self.low_group = grouped_data["low_group"].astype(np.float32)

        self.y = window_data["y"].astype(np.float32)

        self.origin_times = grouped_data["origin_times"]

        # Align sample numbers if a minor mismatch occurs.
        # In a correctly generated pipeline, they should be the same.
        n_grouped = len(self.high_group)
        n_y = len(self.y)
        n = min(n_grouped, n_y)

        if n_grouped != n_y:
            print(
                f"[Warning] Sample number mismatch: grouped={n_grouped}, y={n_y}. "
                f"Using the first {n} aligned samples."
            )

        self.high_group = self.high_group[:n]
        self.mid_group = self.mid_group[:n]
        self.low_group = self.low_group[:n]
        self.y = self.y[:n]
        self.origin_times = self.origin_times[:n]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        high = torch.from_numpy(self.high_group[idx])
        mid = torch.from_numpy(self.mid_group[idx])
        low = torch.from_numpy(self.low_group[idx])
        y = torch.from_numpy(self.y[idx])

        return high, mid, low, y


def build_dataloaders(
    grouped_dir: str | Path,
    windows_dir: str | Path,
    horizon: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    grouped_dir = Path(grouped_dir)
    windows_dir = Path(windows_dir)

    train_dataset = GroupedLoadDataset(
        grouped_npz_path=grouped_dir / "train_grouped.npz",
        window_npz_path=windows_dir / f"H{horizon}" / f"train_H{horizon}.npz",
    )

    val_dataset = GroupedLoadDataset(
        grouped_npz_path=grouped_dir / "val_grouped.npz",
        window_npz_path=windows_dir / f"H{horizon}" / f"val_H{horizon}.npz",
    )

    test_dataset = GroupedLoadDataset(
        grouped_npz_path=grouped_dir / "test_grouped.npz",
        window_npz_path=windows_dir / f"H{horizon}" / f"test_H{horizon}.npz",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    print("\n========== Dataset Size ==========")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# ============================================================
# Metrics
# ============================================================

def inverse_minmax_load(
    x_norm: np.ndarray,
    normalization_params_path: str | Path,
) -> np.ndarray:
    """
    Inverse min-max normalization for load_mw.

    load_norm = (load - min) / (max - min)
    load = load_norm * (max - min) + min
    """
    normalization_params_path = Path(normalization_params_path)

    if not normalization_params_path.exists():
        raise FileNotFoundError(f"Normalization parameter file not found: {normalization_params_path}")

    with open(normalization_params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    if "load_mw" not in params:
        raise KeyError("normalization_params.json must contain key 'load_mw'.")

    load_min = float(params["load_mw"]["min"])
    load_max = float(params["load_mw"]["max"])

    return x_norm * (load_max - load_min) + load_min


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, MAPE, and R2.

    y_true and y_pred shape:
        [N, H]
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    eps = 1e-8

    error = y_pred - y_true

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    mape = np.mean(np.abs(error) / np.maximum(np.abs(y_true), eps)) * 100.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot < eps:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
    }


def compute_metrics_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute metrics for each forecasting step.
    """
    horizon = y_true.shape[1]
    rows = []

    for h in range(horizon):
        metrics = compute_metrics(y_true[:, h:h + 1], y_pred[:, h:h + 1])
        metrics["step"] = h + 1
        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================
# Train / validate / test
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    losses = []

    for high, mid, low, y in dataloader:
        high = high.to(device)
        mid = mid.to(device)
        low = low.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(high, mid, low)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    losses = []

    for high, mid, low, y in dataloader:
        high = high.to(device)
        mid = mid.to(device)
        low = low.to(device)
        y = y.to(device)

        pred = model(high, mid, low)

        loss = criterion(pred, y)
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    preds = []
    trues = []

    for high, mid, low, y in dataloader:
        high = high.to(device)
        mid = mid.to(device)
        low = low.to(device)

        pred = model(high, mid, low)

        preds.append(pred.cpu().numpy())
        trues.append(y.numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    return y_true, y_pred


class EarlyStopping:
    """
    Early stopping based on validation loss.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True

        self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str | Path,
    device: torch.device,
    learning_rate: float,
    max_epochs: int,
    early_stopping_patience: int,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=0.0,
    )

    best_model_path = output_dir / "best_model.pt"

    log_rows = []

    print("\n========== Training ==========")

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = evaluate_loss(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        improved = early_stopping.step(val_loss)

        if improved:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_model_path,
            )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "improved": improved,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.8f} | "
            f"val_loss={val_loss:.8f} | "
            f"best_val={early_stopping.best_loss:.8f}"
        )

        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    elapsed = time.time() - start_time

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "training_log.csv", index=False, encoding="utf-8-sig")

    print(f"\nTraining finished. Elapsed time: {elapsed:.2f} s")
    print(f"Best model saved to: {best_model_path}")

    return best_model_path


# ============================================================
# Save prediction results
# ============================================================

def save_test_results(
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    normalization_params_path: str | Path,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = inverse_minmax_load(y_true_norm, normalization_params_path)
    y_pred = inverse_minmax_load(y_pred_norm, normalization_params_path)

    overall_metrics = compute_metrics(y_true, y_pred)
    metrics_df = pd.DataFrame([overall_metrics])
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")

    step_metrics_df = compute_metrics_by_horizon(y_true, y_pred)
    step_metrics_df.to_csv(output_dir / "test_metrics_by_step.csv", index=False, encoding="utf-8-sig")

    # Save predictions.
    rows = []
    horizon = y_true.shape[1]

    for i in range(y_true.shape[0]):
        row = {"sample_idx": i}

        for h in range(horizon):
            row[f"true_t+{h+1}"] = float(y_true[i, h])
            row[f"pred_t+{h+1}"] = float(y_pred[i, h])
            row[f"error_t+{h+1}"] = float(y_pred[i, h] - y_true[i, h])

        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n========== Test Metrics ==========")
    print(metrics_df)

    print("\n========== Metrics by Forecasting Step ==========")
    print(step_metrics_df)

    print(f"\nSaved: {output_dir / 'test_metrics.csv'}")
    print(f"Saved: {output_dir / 'test_metrics_by_step.csv'}")
    print(f"Saved: {output_dir / 'test_predictions.csv'}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the proposed forecasting model."
    )

    parser.add_argument(
        "--grouped_dir",
        type=str,
        default="outputs/imf_grouping",
        help="Directory containing grouped high/mid/low npz files."
    )

    parser.add_argument(
        "--windows_dir",
        type=str,
        default="outputs/windows",
        help="Directory containing sliding-window label npz files."
    )

    parser.add_argument(
        "--normalization_params",
        type=str,
        default="outputs/normalization_params.json",
        help="Path to normalization parameter JSON file."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/training",
        help="Directory to save training and testing results."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="proposed",
        choices=["proposed", "no_correction", "full_high", "full_low"],
        help="Model name."
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forecasting horizon H."
    )

    parser.add_argument(
        "--input_len",
        type=int,
        default=96,
        help="Input window length L."
    )

    parser.add_argument(
        "--input_channels",
        type=int,
        default=1,
        help="Input channel number."
    )

    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=32,
        help="Hidden channel number."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Initial learning rate."
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs."
    )

    parser.add_argument(
        "--early_stopping",
        type=int,
        default=15,
        help="Early stopping patience."
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
        default=0,
        help="DataLoader worker number."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training device."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_random_seed(args.random_seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_dir = Path(args.output_dir) / f"H{args.horizon}" / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== Configuration ==========")
    print(f"Model: {args.model_name}")
    print(f"Horizon: {args.horizon}")
    print(f"Input length: {args.input_len}")
    print(f"Device: {device}")
    print(f"Random seed: {args.random_seed}")

    # Save configuration.
    config_dict = vars(args)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

    # Build dataloaders.
    train_loader, val_loader, test_loader = build_dataloaders(
        grouped_dir=args.grouped_dir,
        windows_dir=args.windows_dir,
        horizon=args.horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Build model.
    model = build_model(
        model_name=args.model_name,
        input_len=args.input_len,
        output_horizon=args.horizon,
        input_channels=args.input_channels,
        hidden_channels=args.hidden_channels,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Train.
    best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=device,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping,
    )

    # Load best checkpoint.
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Predict on test set.
    y_true_norm, y_pred_norm = predict(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    # Save metrics and predictions.
    save_test_results(
        y_true_norm=y_true_norm,
        y_pred_norm=y_pred_norm,
        normalization_params_path=args.normalization_params,
        output_dir=output_dir,
    )

    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()