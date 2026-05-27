import os
import json
import time
import random
import csv
from contextlib import nullcontext

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.helpers.dataloader import TusDataset
from src.modelos.ResUnet3D import ResUNet3D_HQ

try:
    from torch.amp import GradScaler
except Exception:
    from torch.cuda.amp import GradScaler


# =========================================================
# Reproducibility
# =========================================================
def seed_all(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Numerical / geometric helpers
# =========================================================
def grad3d(x: torch.Tensor):
    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dz = F.pad(dz, (0, 1, 0, 0, 0, 0))

    return dx, dy, dz


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    return (x * mask).sum() / (mask.sum() + eps)


def make_focus_roi(target: torch.Tensor, frac: float, min_thr: float, dilate_ks: int):
    """
    Creates a focus ROI from the ground-truth pressure field.

    target: [B, 1, D, H, W]
    """
    peak = target.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    thr = torch.maximum(
        peak * frac,
        torch.full_like(peak, min_thr),
    )

    roi = (target >= thr).float()

    if dilate_ks is not None and dilate_ks > 1:
        roi = F.max_pool3d(
            roi,
            kernel_size=dilate_ks,
            stride=1,
            padding=dilate_ks // 2,
        )
        roi = (roi > 0).float()

    return roi, peak


# =========================================================
# V2 intermediate loss
# =========================================================
class FocusAwareTUSLoss_V2(nn.Module):
    """
    Intermediate loss for the ablation study.

    This version is intentionally simpler than the final loss.

    It includes:
    1. Global weighted L1 loss
    2. Focus ROI relative MSE
    3. Focus gradient relative loss

    It does NOT include:
    - peak relative loss
    - location loss
    - peak value at GT voxel loss
    - local peak ROI loss
    - adversarial loss
    """

    def __init__(
        self,
        lambda_global=0.8,
        lambda_focus=1.5,
        lambda_grad=0.05,

        wide_frac=0.50,
        wide_min_thr=0.08,
        wide_dilate_ks=7,

        mid_frac=0.60,
        mid_min_thr=0.08,
        mid_dilate_ks=5,

        global_peak_weight=3.0,
        global_peak_gamma=2.0,

        eps=1e-6,
    ):
        super().__init__()

        self.lambda_global = float(lambda_global)
        self.lambda_focus = float(lambda_focus)
        self.lambda_grad = float(lambda_grad)

        self.wide_frac = float(wide_frac)
        self.wide_min_thr = float(wide_min_thr)
        self.wide_dilate_ks = int(wide_dilate_ks)

        self.mid_frac = float(mid_frac)
        self.mid_min_thr = float(mid_min_thr)
        self.mid_dilate_ks = int(mid_dilate_ks)

        self.global_peak_weight = float(global_peak_weight)
        self.global_peak_gamma = float(global_peak_gamma)

        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, epoch: int = 1, return_dict: bool = False):
        pred32 = pred.float()
        target32 = target.float()

        # Keep output physically positive
        pred_pos = pred32.clamp_min(0.0)

        # -------------------------------------------------
        # ROIs from ground truth
        # -------------------------------------------------
        focus_roi_wide, peak = make_focus_roi(
            target32,
            frac=self.wide_frac,
            min_thr=self.wide_min_thr,
            dilate_ks=self.wide_dilate_ks,
        )

        focus_roi_mid, _ = make_focus_roi(
            target32,
            frac=self.mid_frac,
            min_thr=self.mid_min_thr,
            dilate_ks=self.mid_dilate_ks,
        )

        # -------------------------------------------------
        # 1) Global weighted L1
        # -------------------------------------------------
        rel = (target32 / peak).clamp(0.0, 1.0)
        w_global = 1.0 + self.global_peak_weight * rel.pow(self.global_peak_gamma)

        loss_global = (torch.abs(pred_pos - target32) * w_global).mean()

        # -------------------------------------------------
        # 2) Focus ROI relative MSE
        # -------------------------------------------------
        focus_num = masked_mean(
            (pred_pos - target32).pow(2),
            focus_roi_wide,
            eps=self.eps,
        )

        focus_den = masked_mean(
            target32.pow(2),
            focus_roi_wide,
            eps=self.eps,
        ).clamp_min(self.eps)

        loss_focus = focus_num / focus_den

        # -------------------------------------------------
        # 3) Focus gradient relative loss
        # -------------------------------------------------
        pdx, pdy, pdz = grad3d(pred_pos)
        tdx, tdy, tdz = grad3d(target32)

        grad_diff = torch.abs(pdx - tdx) + torch.abs(pdy - tdy) + torch.abs(pdz - tdz)
        grad_ref = torch.abs(tdx) + torch.abs(tdy) + torch.abs(tdz)

        grad_num = masked_mean(
            grad_diff,
            focus_roi_mid,
            eps=self.eps,
        )

        grad_den = masked_mean(
            grad_ref,
            focus_roi_mid,
            eps=self.eps,
        ).clamp_min(self.eps)

        loss_grad = grad_num / grad_den

        # -------------------------------------------------
        # Total V2 loss
        # -------------------------------------------------
        total = (
            self.lambda_global * loss_global
            + self.lambda_focus * loss_focus
            + self.lambda_grad * loss_grad
        )

        if return_dict:
            return total, {
                "total": total.detach(),
                "global": loss_global.detach(),
                "focus": loss_focus.detach(),
                "grad": loss_grad.detach(),
            }

        return total


# =========================================================
# Checkpointing
# =========================================================
def save_ckpt(path, model, optimizer, scaler, epoch, best_val, config):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_val": float(best_val),
        "config": config,
    }
    torch.save(ckpt, path)


# =========================================================
# CSV logging
# =========================================================
def init_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "train_mae",
                "val_mae",
                "train_mse",
                "val_mse",
                "train_global",
                "val_global",
                "train_focus",
                "val_focus",
                "train_grad",
                "val_grad",
                "lr",
                "time_sec",
            ])


def append_csv(
    csv_path,
    epoch,
    train_loss,
    val_loss,
    train_mae,
    val_mae,
    train_mse,
    val_mse,
    train_components,
    val_components,
    lr,
    time_sec,
):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            int(epoch),
            f"{train_loss:.8e}",
            f"{val_loss:.8e}",
            f"{train_mae:.8e}",
            f"{val_mae:.8e}",
            f"{train_mse:.8e}",
            f"{val_mse:.8e}",
            f"{train_components.get('global', float('nan')):.8e}",
            f"{val_components.get('global', float('nan')):.8e}",
            f"{train_components.get('focus', float('nan')):.8e}",
            f"{val_components.get('focus', float('nan')):.8e}",
            f"{train_components.get('grad', float('nan')):.8e}",
            f"{val_components.get('grad', float('nan')):.8e}",
            f"{lr:.8e}",
            f"{time_sec:.2f}",
        ])


# =========================================================
# Metrics
# =========================================================
@torch.no_grad()
def batch_metrics(pred, target):
    pred = pred.float()
    target = target.float()

    mae = torch.mean(torch.abs(pred - target))
    mse = torch.mean((pred - target) ** 2)

    return {
        "mae": float(mae.item()),
        "mse": float(mse.item()),
    }


def average_components(component_sums, n_batches):
    if n_batches == 0:
        return {}

    return {
        k: v / n_batches
        for k, v in component_sums.items()
    }


# =========================================================
# Train loop
# =========================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    epoch,
    use_amp=True,
    grad_clip=1.0,
):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    n_batches = 0

    component_sums = {
        "global": 0.0,
        "focus": 0.0,
        "grad": 0.0,
    }

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (
            use_amp and device == "cuda"
        ) else nullcontext()

        with amp_ctx:
            pred = model(X)
            loss, loss_dict = criterion(pred, y, epoch=epoch, return_dict=True)

        if not torch.isfinite(loss):
            print("Non-finite training loss detected. Batch skipped.")
            continue

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        metrics = batch_metrics(pred.detach(), y.detach())

        total_loss += float(loss.item())
        total_mae += metrics["mae"]
        total_mse += metrics["mse"]

        for k in component_sums.keys():
            component_sums[k] += float(loss_dict[k].item())

        n_batches += 1

    if n_batches == 0:
        return float("inf"), float("inf"), float("inf"), {}

    return (
        total_loss / n_batches,
        total_mae / n_batches,
        total_mse / n_batches,
        average_components(component_sums, n_batches),
    )


# =========================================================
# Validation loop
# =========================================================
@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    criterion,
    device,
    epoch,
    use_amp=True,
):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    n_batches = 0

    component_sums = {
        "global": 0.0,
        "focus": 0.0,
        "grad": 0.0,
    }

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (
            use_amp and device == "cuda"
        ) else nullcontext()

        with amp_ctx:
            pred = model(X)
            loss, loss_dict = criterion(pred, y, epoch=epoch, return_dict=True)

        if not torch.isfinite(loss):
            print("Non-finite validation loss detected. Batch skipped.")
            continue

        metrics = batch_metrics(pred, y)

        total_loss += float(loss.item())
        total_mae += metrics["mae"]
        total_mse += metrics["mse"]

        for k in component_sums.keys():
            component_sums[k] += float(loss_dict[k].item())

        n_batches += 1

    if n_batches == 0:
        return float("inf"), float("inf"), float("inf"), {}

    return (
        total_loss / n_batches,
        total_mae / n_batches,
        total_mse / n_batches,
        average_components(component_sums, n_batches),
    )


# =========================================================
# Dataloader helper
# =========================================================
def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


# =========================================================
# Main
# =========================================================
def main():
    # =========================
    # CONFIG
    # =========================
    SEED = 123

    SAVE_DIR = "checkpoints_unetprimitiva_v2_focus_aware"
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/train"
    VAL_DIR   = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/val"
    TEST_DIR  = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

    BATCH_SIZE = 1
    NUM_WORKERS = 2
    PIN_MEMORY = True

    EPOCHS = 100

    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    BASE = 8
    USE_SE = False
    OUT_POSITIVE = True

    USE_SCHEDULER = True
    PLATEAU_PATIENCE = 12
    PLATEAU_FACTOR = 0.5

    USE_AMP = True
    GRAD_CLIP = 1.0

    CSV_PATH = os.path.join(SAVE_DIR, "training_log.csv")

    # =========================
    # V2 LOSS CONFIG
    # =========================
    LOSS_CONFIG = {
        "lambda_global": 0.8,
        "lambda_focus": 1.5,
        "lambda_grad": 0.05,

        "wide_frac": 0.50,
        "wide_min_thr": 0.08,
        "wide_dilate_ks": 7,

        "mid_frac": 0.60,
        "mid_min_thr": 0.08,
        "mid_dilate_ks": 5,

        "global_peak_weight": 3.0,
        "global_peak_gamma": 2.0,

        "eps": 1e-6,
    }

    # =========================
    # SETUP
    # =========================
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # =========================
    # DATA
    # =========================
    train_ds = TusDataset(TRAIN_DIR)
    val_ds = TusDataset(VAL_DIR)
    test_ds = TusDataset(TEST_DIR)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = make_loader(
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = make_loader(
        dataset=val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # =========================
    # MODEL
    # =========================
    model = ResUNet3D_HQ(
        in_ch=2,
        out_ch=1,
        base=BASE,
        norm_kind="group",
        use_se=USE_SE,
        out_positive=OUT_POSITIVE,
    ).to(device)

    print("Training V2 focus-aware 3D residual U-Net from scratch.")
    print("Losses: global weighted L1 + focus ROI relative MSE + focus gradient relative loss.")
    print("No peak, location, peak_gt, peak_roi, or adversarial losses.")

    # =========================
    # LOSS
    # =========================
    criterion = FocusAwareTUSLoss_V2(**LOSS_CONFIG)

    # =========================
    # OPTIMIZER / SCHEDULER
    # =========================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = None
    if USE_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=PLATEAU_FACTOR,
            patience=PLATEAU_PATIENCE,
        )

    try:
        scaler = GradScaler("cuda", enabled=(device == "cuda" and USE_AMP))
    except TypeError:
        scaler = GradScaler(enabled=(device == "cuda" and USE_AMP))

    # =========================
    # SAVE CONFIG
    # =========================
    config = {
        "experiment_name": "unet_v2_focus_aware",
        "description": (
            "Intermediate 3D residual U-Net trained with a focus-aware composite loss. "
            "This version includes global weighted L1, focus ROI relative MSE, and "
            "focus gradient relative loss. It intentionally excludes peak-relative, "
            "location, peak-at-GT, peak-ROI, and adversarial losses."
        ),
        "seed": SEED,
        "save_dir": SAVE_DIR,
        "train_dir": TRAIN_DIR,
        "val_dir": VAL_DIR,
        "test_dir": TEST_DIR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "base": BASE,
        "use_se": USE_SE,
        "out_positive": OUT_POSITIVE,
        "loss_type": "FocusAwareTUSLoss_V2",
        "loss_config": LOSS_CONFIG,
        "optimizer": "AdamW",
        "scheduler": {
            "use_scheduler": USE_SCHEDULER,
            "type": "ReduceLROnPlateau",
            "patience": PLATEAU_PATIENCE,
            "factor": PLATEAU_FACTOR,
        },
        "use_amp": USE_AMP,
        "grad_clip": GRAD_CLIP,
        "saved_checkpoints": ["best.pth", "last.pth"],
    }

    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    init_csv(CSV_PATH)

    # =========================
    # TRAINING
    # =========================
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_mae, train_mse, train_components = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=USE_AMP,
            grad_clip=GRAD_CLIP,
        )

        val_loss, val_mae, val_mse, val_components = eval_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=USE_AMP,
        )

        prev_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None and np.isfinite(val_loss):
            scheduler.step(val_loss)

        new_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"train_MAE={train_mae:.6f} | "
            f"val_MAE={val_mae:.6f} | "
            f"train_MSE={train_mse:.6e} | "
            f"val_MSE={val_mse:.6e} | "
            f"val_global={val_components.get('global', float('nan')):.6f} | "
            f"val_focus={val_components.get('focus', float('nan')):.6f} | "
            f"val_grad={val_components.get('grad', float('nan')):.6f} | "
            f"lr={new_lr:.2e} | "
            f"time={dt:.1f}s"
        )

        if new_lr < prev_lr:
            print(f"LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        append_csv(
            CSV_PATH,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_mae=train_mae,
            val_mae=val_mae,
            train_mse=train_mse,
            val_mse=val_mse,
            train_components=train_components,
            val_components=val_components,
            lr=new_lr,
            time_sec=dt,
        )

        # -------------------------------------------------
        # Save last checkpoint only
        # -------------------------------------------------
        save_ckpt(
            os.path.join(SAVE_DIR, "last.pth"),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_val=best_val,
            config=config,
        )

        # -------------------------------------------------
        # Save best checkpoint only
        # -------------------------------------------------
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch

            save_ckpt(
                os.path.join(SAVE_DIR, "best.pth"),
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_val=best_val,
                config=config,
            )

            print(f"New best model saved at epoch {epoch:03d} | best_val={best_val:.6f}")

    print("Training done.")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved files in: {SAVE_DIR}")
    print("Expected outputs:")
    print(" - best.pth")
    print(" - last.pth")
    print(" - config.json")
    print(" - training_log.csv")


if __name__ == "__main__":
    main()