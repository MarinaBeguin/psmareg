"""Train and validate a simple PSMAReg paired registration model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.paired_dataset import PSMARegPairDataset
from models.losses import image_mse_loss, smoothness_loss, multiclass_dice_loss, lncc_loss
from models.network import SmallRegistrationUNet
from utils.warping import warp_image_3d

TensorBatch = dict[str, torch.Tensor | list[str]]
Metrics = dict[str, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--val_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_pet", type=float, default=0.2)
    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--lambda_ct_dice", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def _get_tensor(batch: TensorBatch, key: str, device: torch.device) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor for key '{key}', got {type(value)}")
    return value.to(device)


def _compute_batch_metrics(
    model: nn.Module,
    batch: TensorBatch,
    device: torch.device,
    lambda_pet: float = 0.2,
    lambda_smooth: float = 0.01,
    lambda_ct_dice: float = 0.5,
) -> tuple[torch.Tensor, Metrics]:
    fixed = _get_tensor(batch, "fixed", device)
    moving = _get_tensor(batch, "moving", device)

    fixed_ct = _get_tensor(batch, "fixed_ct", device)
    fixed_pet = _get_tensor(batch, "fixed_pet", device)
    moving_ct = _get_tensor(batch, "moving_ct", device)
    moving_pet = _get_tensor(batch, "moving_pet", device)

    displacement = model(fixed, moving)

    warped_ct = warp_image_3d(moving_ct, displacement)
    warped_pet = warp_image_3d(moving_pet, displacement)

    loss_ct_dice = torch.zeros((), device=device)

    if "fixed_ct_label" in batch and "moving_ct_label" in batch:
        fixed_ct_label = batch["fixed_ct_label"].to(device).squeeze(1)
        moving_ct_label = batch["moving_ct_label"].to(device).squeeze(1)

        fixed_ct_fg = (fixed_ct_label > 0).float().unsqueeze(1)
        moving_ct_fg = (moving_ct_label > 0).float().unsqueeze(1)

        warped_moving_ct_fg = warp_image_3d(
            moving_ct_fg,
            displacement,
            interpolation_mode="bilinear",
        )

        loss_ct_dice = dice_loss(
            warped_moving_ct_fg,
            fixed_ct_fg,
        )

    loss_ct = lncc_loss(warped_ct, fixed_ct)
    loss_pet = image_mse_loss(warped_pet, fixed_pet)
    loss_smooth = smoothness_loss(displacement)

    loss = (
        loss_ct
        + lambda_pet * loss_pet
        + lambda_smooth * loss_smooth
        + lambda_ct_dice * loss_ct_dice
    )

    lncc_before_ct = lncc_loss(moving_ct, fixed_ct)
    mse_before_pet = image_mse_loss(moving_pet, fixed_pet)

    metrics = {
        "loss": float(loss.detach().cpu()),
        "loss_ct": float(loss_ct.detach().cpu()),
        "loss_pet": float(loss_pet.detach().cpu()),
        "loss_ct_dice": float(loss_ct_dice.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "lncc_before_ct": float(lncc_before_ct.detach().cpu()),
        "lncc_after_ct": float(loss_ct.detach().cpu()),
        "mse_before_pet": float(mse_before_pet.detach().cpu()),
        "mse_after_pet": float(loss_pet.detach().cpu()),
        "mean_disp": float(displacement.abs().mean().detach().cpu()),
        "max_disp": float(displacement.abs().max().detach().cpu()),
        "disp_std": float(displacement.std().detach().cpu()),
    }

    return loss, metrics


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute binary soft Dice loss."""
    intersection = (pred * target).sum()
    denominator = pred.sum() + target.sum()
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def _average_metrics(metrics_list: list[Metrics]) -> Metrics:
    if not metrics_list:
        raise ValueError("Cannot average an empty metrics list.")

    averaged: Metrics = {}
    keys = metrics_list[0].keys()

    for key in keys:
        averaged[key] = sum(metrics[key] for metrics in metrics_list) / len(metrics_list)

    return averaged


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader[TensorBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_pet: float,
    lambda_smooth: float,
    lambda_ct_dice: float,
) -> Metrics:
    model.train()
    metrics_list: list[Metrics] = []

    for batch in loader:
        loss, metrics = _compute_batch_metrics(
            model=model,
            batch=batch,
            device=device,
            lambda_pet=lambda_pet,
            lambda_smooth=lambda_smooth,
            lambda_ct_dice=lambda_ct_dice,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics_list.append(metrics)

    return _average_metrics(metrics_list)


@torch.no_grad()
def _run_validation_epoch(
    model: nn.Module,
    loader: DataLoader[TensorBatch],
    device: torch.device,
    lambda_pet: float,
    lambda_smooth: float,
    lambda_ct_dice: float,
) -> Metrics:
    model.eval()
    metrics_list: list[Metrics] = []

    for batch in loader:
        _, metrics = _compute_batch_metrics(
            model=model,
            batch=batch,
            device=device,
            lambda_pet=lambda_pet,
            lambda_smooth=lambda_smooth,
            lambda_ct_dice=lambda_ct_dice,
        )
        metrics_list.append(metrics)

    return _average_metrics(metrics_list)


def _save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_metrics: Metrics,
    val_metrics: Metrics,
    best_val_loss: float,
) -> None:
    print(f"--> Saving best epoch to {checkpoint_path}...")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )


def _format_metrics(prefix: str, metrics: Metrics) -> str:
    return (
        f"{prefix}_loss={metrics['loss']:.6f} | "
        f"{prefix}_ct={metrics['loss_ct']:.6f} | "
        f"{prefix}_pet={metrics['loss_pet']:.6f} | "
        f"{prefix}_smooth={metrics['loss_smooth']:.6f} | "
        f"{prefix}_ct_dice={metrics['loss_ct_dice']:.6f} | "
        f"{prefix}_ct_before={metrics['lncc_before_ct']:.6f} | "
        f"{prefix}_ct_after={metrics['lncc_after_ct']:.6f} | "
        f"{prefix}_pet_before={metrics['mse_before_pet']:.6f} | "
        f"{prefix}_pet_after={metrics['mse_after_pet']:.6f} | "
        f"{prefix}_mean_disp={metrics['mean_disp']:.4f} | "
        f"{prefix}_max_disp={metrics['max_disp']:.4f} | "
        f"{prefix}_disp_std={metrics['disp_std']:.4f}"
    )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PSMARegPairDataset(args.train_manifest)
    val_dataset = PSMARegPairDataset(args.val_manifest)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SmallRegistrationUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_metrics = _run_train_epoch(
            model=model,
            loader=cast(DataLoader[TensorBatch], train_loader),
            optimizer=optimizer,
            device=device,
            lambda_pet=args.lambda_pet,
            lambda_smooth=args.lambda_smooth,
            lambda_ct_dice=args.lambda_ct_dice,
        )

        val_metrics = _run_validation_epoch(
            model=model,
            loader=cast(DataLoader[TensorBatch], val_loader),
            device=device,
            lambda_pet=args.lambda_pet,
            lambda_smooth=args.lambda_smooth,
            lambda_ct_dice=args.lambda_ct_dice,
        )

        current_val_loss = val_metrics["loss"]

        is_best = current_val_loss < best_val_loss
        if is_best:
            best_val_loss = current_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        _save_checkpoint(
            checkpoint_path=args.output_dir / "latest.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val_loss=best_val_loss,
        )

        if is_best:
            _save_checkpoint(
                checkpoint_path=args.output_dir / "best_val.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_loss=best_val_loss,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"{_format_metrics('train', train_metrics)} | "
            f"{_format_metrics('val', val_metrics)} | "
            f"best_val_loss={best_val_loss:.6f} | "
            f"patience={epochs_without_improvement}/{args.patience}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch:03d}.")
            break


if __name__ == "__main__":
    main()
