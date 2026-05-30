# models.py
# -*- coding: utf-8 -*-
"""
Model definitions for the proposed frequency-aware hybrid forecasting framework.

Main modules:
    1. MSGatedTCN
    2. AdaptiveDLinear
    3. GatedFusion
    4. CorrectionNetwork
    5. ProposedModel

Input format:
    high_group: Tensor, shape [B, L] or [B, L, 1]
    mid_group:  Tensor, shape [B, L] or [B, L, 1]
    low_group:  Tensor, shape [B, L] or [B, L, 1]

Output format:
    final_pred: Tensor, shape [B, H]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility
# ============================================================

def ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Convert input to [B, L, C].

    Accepted:
        [B, L] -> [B, L, 1]
        [B, L, C] -> unchanged
    """
    if x.dim() == 2:
        return x.unsqueeze(-1)

    if x.dim() == 3:
        return x

    raise ValueError(f"Input tensor must be 2D or 3D, got shape {tuple(x.shape)}")


class Chomp1d(nn.Module):
    """
    Remove extra padding introduced by causal convolution.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


# ============================================================
# Multi-scale gated TCN
# ============================================================

class MultiScaleGatedTCNBlock(nn.Module):
    """
    One multi-scale gated TCN block.

    Input:
        x: [B, C_in, L]

    Output:
        out: [B, C_out, L]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (2, 3, 5),
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.branches = nn.ModuleList()

        for kernel_size in kernel_sizes:
            padding = (kernel_size - 1) * dilation

            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                ),
                Chomp1d(padding),
            )

            self.branches.append(branch)

        fused_channels = out_channels * len(kernel_sizes)

        self.feature_proj = nn.Conv1d(fused_channels, out_channels, kernel_size=1)
        self.gate_proj = nn.Conv1d(fused_channels, out_channels, kernel_size=1)

        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        multi_scale_feature = torch.cat(branch_outputs, dim=1)

        feature = torch.tanh(self.feature_proj(multi_scale_feature))
        gate = torch.sigmoid(self.gate_proj(multi_scale_feature))

        gated_feature = feature * gate
        gated_feature = self.dropout(gated_feature)

        residual = self.residual_proj(x)

        out = gated_feature + residual
        out = self.norm(out)

        return out


class MSGatedTCN(nn.Module):
    """
    Multi-scale gated TCN module.

    Input:
        x: [B, L] or [B, L, C]

    Output:
        pred: [B, H]
        hidden: [B, hidden_channels]
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,
        output_horizon: int = 12,
        kernel_sizes: tuple[int, ...] = (2, 3, 5),
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_ch = input_channels

        for dilation in dilations:
            layers.append(
                MultiScaleGatedTCNBlock(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    kernel_sizes=kernel_sizes,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = hidden_channels

        self.network = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_channels, output_horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = ensure_3d(x)              # [B, L, C]
        x = x.transpose(1, 2)         # [B, C, L]

        hidden_seq = self.network(x)  # [B, hidden, L]

        # Use the last time step as forecasting representation.
        hidden = hidden_seq[:, :, -1]  # [B, hidden]

        pred = self.output_layer(hidden)  # [B, H]

        return pred, hidden


# ============================================================
# Adaptive DLinear
# ============================================================

class MovingAverage(nn.Module):
    """
    Causal moving average for trend extraction.

    Input:
        x: [B, L, C]

    Output:
        trend: [B, L, C]
    """

    def __init__(self, kernel_size: int):
        super().__init__()

        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")

        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)  # [B, L, C]

        # Causal padding on the left side.
        x_perm = x.transpose(1, 2)  # [B, C, L]
        x_pad = F.pad(x_perm, (self.kernel_size - 1, 0), mode="replicate")

        trend = F.avg_pool1d(
            x_pad,
            kernel_size=self.kernel_size,
            stride=1,
        )

        trend = trend.transpose(1, 2)  # [B, L, C]

        return trend


class AdaptiveDLinear(nn.Module):
    """
    Adaptive DLinear module.

    This implementation uses a configurable moving-average window.
    The window size can be selected according to the dominant period in preprocessing
    or set as a fixed default value.

    Input:
        x: [B, L] or [B, L, C]

    Output:
        pred: [B, H]
        hidden: [B, hidden_dim]
    """

    def __init__(
        self,
        input_len: int = 96,
        output_horizon: int = 12,
        input_channels: int = 1,
        moving_avg_kernel: int = 7,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.input_len = input_len
        self.output_horizon = output_horizon
        self.input_channels = input_channels

        self.moving_avg = MovingAverage(kernel_size=moving_avg_kernel)

        self.trend_linear = nn.Linear(input_len * input_channels, output_horizon)
        self.residual_linear = nn.Linear(input_len * input_channels, output_horizon)

        self.hidden_projection = nn.Linear(output_horizon, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = ensure_3d(x)  # [B, L, C]

        if x.size(1) != self.input_len:
            raise ValueError(
                f"Expected input length {self.input_len}, got {x.size(1)}"
            )

        trend = self.moving_avg(x)
        residual = x - trend

        trend_flat = trend.reshape(trend.size(0), -1)
        residual_flat = residual.reshape(residual.size(0), -1)

        trend_pred = self.trend_linear(trend_flat)
        residual_pred = self.residual_linear(residual_flat)

        pred = trend_pred + residual_pred  # [B, H]

        hidden = torch.tanh(self.hidden_projection(pred))  # [B, hidden_dim]

        return pred, hidden


# ============================================================
# Gated fusion for mid-frequency group
# ============================================================

class GatedFusion(nn.Module):
    """
    Gated fusion of MS-gTCN and Adaptive DLinear representations.

    Input:
        pred_tcn: [B, H]
        hidden_tcn: [B, D]
        pred_linear: [B, H]
        hidden_linear: [B, D]

    Output:
        fused_pred: [B, H]
        fused_hidden: [B, D]
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        output_horizon: int = 12,
    ):
        super().__init__()

        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.output_layer = nn.Linear(hidden_dim, output_horizon)

    def forward(
        self,
        pred_tcn: torch.Tensor,
        hidden_tcn: torch.Tensor,
        pred_linear: torch.Tensor,
        hidden_linear: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        fusion_input = torch.cat([hidden_tcn, hidden_linear], dim=-1)
        gate = self.gate_layer(fusion_input)

        fused_hidden = gate * hidden_tcn + (1.0 - gate) * hidden_linear

        # Forecast-level residual fusion.
        fused_pred_from_hidden = self.output_layer(fused_hidden)
        fused_pred_average = 0.5 * pred_tcn + 0.5 * pred_linear

        fused_pred = fused_pred_from_hidden + fused_pred_average

        return fused_pred, fused_hidden


# ============================================================
# Lightweight correction network
# ============================================================

class CorrectionNetwork(nn.Module):
    """
    Lightweight error correction network.

    Input:
        concat of:
            high_pred [B, H]
            mid_pred  [B, H]
            low_pred  [B, H]
            sum_pred  [B, H]

        total input dimension = 4 * H

    Output:
        correction term [B, H]
    """

    def __init__(
        self,
        output_horizon: int = 12,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()

        input_dim = output_horizon * 4

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_horizon),
        )

    def forward(
        self,
        high_pred: torch.Tensor,
        mid_pred: torch.Tensor,
        low_pred: torch.Tensor,
    ) -> torch.Tensor:

        sum_pred = high_pred + mid_pred + low_pred

        correction_input = torch.cat(
            [high_pred, mid_pred, low_pred, sum_pred],
            dim=-1,
        )

        correction = self.network(correction_input)

        return correction


# ============================================================
# Proposed complete model
# ============================================================

class ProposedModel(nn.Module):
    """
    Complete proposed model.

    High-frequency group:
        MS-gTCN

    Mid-frequency group:
        MS-gTCN + Adaptive DLinear + gated fusion

    Low-frequency group:
        Adaptive DLinear

    Final:
        Summation + lightweight correction network
    """

    def __init__(
        self,
        input_len: int = 96,
        output_horizon: int = 12,
        input_channels: int = 1,
        hidden_channels: int = 32,
        kernel_sizes: tuple[int, ...] = (2, 3, 5),
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout_tcn: float = 0.2,
        dlinear_kernel: int = 7,
        correction_hidden_dims: tuple[int, int] = (128, 64),
        correction_dropout: float = 0.1,
        use_correction: bool = True,
    ):
        super().__init__()

        self.input_len = input_len
        self.output_horizon = output_horizon
        self.use_correction = use_correction

        # High-frequency branch.
        self.high_tcn = MSGatedTCN(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_horizon=output_horizon,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            dropout=dropout_tcn,
        )

        # Mid-frequency branch.
        self.mid_tcn = MSGatedTCN(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_horizon=output_horizon,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            dropout=dropout_tcn,
        )

        self.mid_dlinear = AdaptiveDLinear(
            input_len=input_len,
            output_horizon=output_horizon,
            input_channels=input_channels,
            moving_avg_kernel=dlinear_kernel,
            hidden_dim=hidden_channels,
        )

        self.mid_fusion = GatedFusion(
            hidden_dim=hidden_channels,
            output_horizon=output_horizon,
        )

        # Low-frequency branch.
        self.low_dlinear = AdaptiveDLinear(
            input_len=input_len,
            output_horizon=output_horizon,
            input_channels=input_channels,
            moving_avg_kernel=dlinear_kernel,
            hidden_dim=hidden_channels,
        )

        # Correction network.
        self.correction_network = CorrectionNetwork(
            output_horizon=output_horizon,
            hidden_dims=correction_hidden_dims,
            dropout=correction_dropout,
        )

    def forward(
        self,
        high_group: torch.Tensor,
        mid_group: torch.Tensor,
        low_group: torch.Tensor,
        return_components: bool = False,
    ):

        # High-frequency forecasting.
        high_pred, high_hidden = self.high_tcn(high_group)

        # Mid-frequency forecasting.
        mid_tcn_pred, mid_tcn_hidden = self.mid_tcn(mid_group)
        mid_linear_pred, mid_linear_hidden = self.mid_dlinear(mid_group)

        mid_pred, mid_hidden = self.mid_fusion(
            pred_tcn=mid_tcn_pred,
            hidden_tcn=mid_tcn_hidden,
            pred_linear=mid_linear_pred,
            hidden_linear=mid_linear_hidden,
        )

        # Low-frequency forecasting.
        low_pred, low_hidden = self.low_dlinear(low_group)

        # Initial reconstruction.
        sum_pred = high_pred + mid_pred + low_pred

        if self.use_correction:
            correction = self.correction_network(
                high_pred=high_pred,
                mid_pred=mid_pred,
                low_pred=low_pred,
            )
            final_pred = sum_pred + correction
        else:
            correction = torch.zeros_like(sum_pred)
            final_pred = sum_pred

        if return_components:
            return {
                "final_pred": final_pred,
                "sum_pred": sum_pred,
                "high_pred": high_pred,
                "mid_pred": mid_pred,
                "low_pred": low_pred,
                "correction": correction,
                "high_hidden": high_hidden,
                "mid_hidden": mid_hidden,
                "low_hidden": low_hidden,
            }

        return final_pred


# ============================================================
# Ablation variants
# ============================================================

class NoCorrectionModel(ProposedModel):
    """
    Ablation variant: remove lightweight correction network.
    """

    def __init__(self, *args, **kwargs):
        kwargs["use_correction"] = False
        super().__init__(*args, **kwargs)


class FullHighFrequencyModel(nn.Module):
    """
    Ablation variant:
    Use MS-gTCN for all three modal groups.
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,
        output_horizon: int = 12,
        kernel_sizes: tuple[int, ...] = (2, 3, 5),
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
    ):
        super().__init__()

        self.high_tcn = MSGatedTCN(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_horizon=output_horizon,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            dropout=dropout,
        )

        self.mid_tcn = MSGatedTCN(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_horizon=output_horizon,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            dropout=dropout,
        )

        self.low_tcn = MSGatedTCN(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_horizon=output_horizon,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            dropout=dropout,
        )

    def forward(
        self,
        high_group: torch.Tensor,
        mid_group: torch.Tensor,
        low_group: torch.Tensor,
    ) -> torch.Tensor:

        high_pred, _ = self.high_tcn(high_group)
        mid_pred, _ = self.mid_tcn(mid_group)
        low_pred, _ = self.low_tcn(low_group)

        return high_pred + mid_pred + low_pred


class FullLowFrequencyModel(nn.Module):
    """
    Ablation variant:
    Use Adaptive DLinear for all three modal groups.
    """

    def __init__(
        self,
        input_len: int = 96,
        output_horizon: int = 12,
        input_channels: int = 1,
        moving_avg_kernel: int = 7,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.high_dlinear = AdaptiveDLinear(
            input_len=input_len,
            output_horizon=output_horizon,
            input_channels=input_channels,
            moving_avg_kernel=moving_avg_kernel,
            hidden_dim=hidden_dim,
        )

        self.mid_dlinear = AdaptiveDLinear(
            input_len=input_len,
            output_horizon=output_horizon,
            input_channels=input_channels,
            moving_avg_kernel=moving_avg_kernel,
            hidden_dim=hidden_dim,
        )

        self.low_dlinear = AdaptiveDLinear(
            input_len=input_len,
            output_horizon=output_horizon,
            input_channels=input_channels,
            moving_avg_kernel=moving_avg_kernel,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        high_group: torch.Tensor,
        mid_group: torch.Tensor,
        low_group: torch.Tensor,
    ) -> torch.Tensor:

        high_pred, _ = self.high_dlinear(high_group)
        mid_pred, _ = self.mid_dlinear(mid_group)
        low_pred, _ = self.low_dlinear(low_group)

        return high_pred + mid_pred + low_pred


# ============================================================
# Model factory
# ============================================================

def build_model(
    model_name: str,
    input_len: int = 96,
    output_horizon: int = 12,
    input_channels: int = 1,
    hidden_channels: int = 32,
) -> nn.Module:
    """
    Build model by name.

    Supported:
        proposed
        no_correction
        full_high
        full_low
    """
    model_name = model_name.lower()

    if model_name == "proposed":
        return ProposedModel(
            input_len=input_len,
            output_horizon=output_horizon,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
        )

    if model_name == "no_correction":
        return NoCorrectionModel(
            input_len=input_len,
            output_horizon=output_horizon,
            input_channels=input_channels,
            hidden_channels=hidden_channels,
        )

    if model_name == "full_high":
        return FullHighFrequencyModel(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_horizon=output_horizon,
        )

    if model_name == "full_low":
        return FullLowFrequencyModel(
            input_len=input_len,
            output_horizon=output_horizon,
            input_channels=input_channels,
            hidden_dim=hidden_channels,
        )

    raise ValueError(
        f"Unsupported model_name: {model_name}. "
        f"Available options: proposed, no_correction, full_high, full_low"
    )


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    batch_size = 8
    input_len = 96
    horizon = 12

    high = torch.randn(batch_size, input_len)
    mid = torch.randn(batch_size, input_len)
    low = torch.randn(batch_size, input_len)

    model = ProposedModel(
        input_len=input_len,
        output_horizon=horizon,
        input_channels=1,
        hidden_channels=32,
    )

    outputs = model(high, mid, low, return_components=True)

    print("final_pred:", outputs["final_pred"].shape)
    print("high_pred:", outputs["high_pred"].shape)
    print("mid_pred:", outputs["mid_pred"].shape)
    print("low_pred:", outputs["low_pred"].shape)
    print("correction:", outputs["correction"].shape)