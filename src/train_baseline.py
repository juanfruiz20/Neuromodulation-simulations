import os
import json
import time
import random
import csv
from contextlib import nullcontext

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

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
# Fixed subset helper
# =========================================================
def make_fixed_subset(dataset, fraction=0.5, seed=123):
    """
    Creates a fixed random subset of the dataset.

    fraction=0.5 means that only 50% of the original training set
    will be used. The selection is reproducible because it uses a fixed seed.
    """
    n_total = len(dataset)
    n_subset = int(round(n_total * fraction))

    if n_subset <= 0:
        raise RuntimeError("Subset size is 0. Increase TRAIN_FRACTION.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)[:n_subset].tolist()

    return Subset(dataset, indices), indices


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
    use_amp=True,
    grad_clip=1.0,
):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    n_batches = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (
            use_amp and device == "cuda"
        ) else nullcontext()

        with amp_ctx:
            pred = model(X)
            loss = criterion(pred, y)

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
        n_batches += 1

    if n_batches == 0:
        return float("inf"), float("inf"), float("inf")

    return (
        total_loss / n_batches,
        total_mae / n_batches,
        total_mse / n_batches,
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
    use_amp=True,
):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    n_batches = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (
            use_amp and device == "cuda"
        ) else nullcontext()

        with amp_ctx:
            pred = model(X)
            loss = criterion(pred, y)

        if not torch.isfinite(loss):
            print("Non-finite validation loss detected. Batch skipped.")
            continue

        metrics = batch_metrics(pred, y)

        total_loss += float(loss.item())
        total_mae += metrics["mae"]
        total_mse += metrics["mse"]
        n_batches += 1

    if n_batches == 0:
        return float("inf"), float("inf"), float("inf")

    return (
        total_loss / n_batches,
        total_mae / n_batches,
        total_mse / n_batches,
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

    SAVE_DIR = "checkpoints_unet_baseline_halfdata_50epochs"
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/train"
    VAL_DIR   = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/val"
    TEST_DIR  = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

    BATCH_SIZE = 1
    NUM_WORKERS = 2

    EPOCHS = 50
    TRAIN_FRACTION = 0.5

    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    # Primitive / lightweight U-Net configuration
    BASE = 8
    USE_SE = False
    OUT_POSITIVE = True

    # Simple baseline loss
    LOSS_TYPE = "l1"

    USE_SCHEDULER = True
    PLATEAU_PATIENCE = 12
    PLATEAU_FACTOR = 0.5

    USE_AMP = True
    GRAD_CLIP = 1.0

    CSV_PATH = os.path.join(SAVE_DIR, "training_log.csv")

    # =========================
    # SETUP
    # =========================
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = True if device == "cuda" else False

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
    train_ds_full = TusDataset(TRAIN_DIR)
    val_ds = TusDataset(VAL_DIR)
    test_ds = TusDataset(TEST_DIR)

    train_ds, train_indices = make_fixed_subset(
        dataset=train_ds_full,
        fraction=TRAIN_FRACTION,
        seed=SEED,
    )

    print(
        f"Train full: {len(train_ds_full)} | "
        f"Train used: {len(train_ds)} | "
        f"Val: {len(val_ds)} | "
        f"Test: {len(test_ds)}"
    )

    # Save selected subset indices for reproducibility
    with open(os.path.join(SAVE_DIR, "train_subset_indices.json"), "w", encoding="utf-8") as f:
        json.dump(train_indices, f, indent=2)

    # Save selected subset filenames if available
    if hasattr(train_ds_full, "files"):
        train_subset_files = [
            os.path.basename(train_ds_full.files[i])
            for i in train_indices
        ]

        with open(os.path.join(SAVE_DIR, "train_subset_files.json"), "w", encoding="utf-8") as f:
            json.dump(train_subset_files, f, indent=2)

    train_loader = make_loader(
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = make_loader(
        dataset=val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
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

    print("Training primitive 3D residual U-Net baseline from scratch.")
    print("Configuration:")
    print(f"  Train fraction: {TRAIN_FRACTION}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Base channels: {BASE}")
    print(f"  Use SE: {USE_SE}")
    print(f"  Loss: {LOSS_TYPE}")
    print("No focus-aware, peak-aware, gradient, location, or adversarial losses.")

    # =========================
    # LOSS
    # =========================
    if LOSS_TYPE == "l1":
        criterion = nn.L1Loss()
    elif LOSS_TYPE == "mse":
        criterion = nn.MSELoss()
    elif LOSS_TYPE == "smooth_l1":
        criterion = nn.SmoothL1Loss(beta=0.02)
    else:
        raise ValueError(f"Unknown LOSS_TYPE: {LOSS_TYPE}")

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
        "experiment_name": "unet_baseline_halfdata_50epochs",
        "description": (
            "Primitive lightweight 3D residual U-Net baseline trained with only "
            "50% of the training set for 50 epochs. The model uses reduced channel "
            "capacity, no squeeze-and-excitation blocks, and only a voxel-wise L1 "
            "reconstruction loss. No focus-aware, peak-aware, gradient, location, "
            "peak-value, peak-ROI, or adversarial losses are used."
        ),
        "seed": SEED,
        "save_dir": SAVE_DIR,
        "train_dir": TRAIN_DIR,
        "val_dir": VAL_DIR,
        "test_dir": TEST_DIR,
        "train_fraction": TRAIN_FRACTION,
        "train_full_size": len(train_ds_full),
        "train_used_size": len(train_ds),
        "subset_seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "base": BASE,
        "use_se": USE_SE,
        "out_positive": OUT_POSITIVE,
        "loss_type": LOSS_TYPE,
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

        train_loss, train_mae, train_mse = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            use_amp=USE_AMP,
            grad_clip=GRAD_CLIP,
        )

        val_loss, val_mae, val_mse = eval_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
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
            lr=new_lr,
            time_sec=dt,
        )

        # Save best checkpoint
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

        # Save last checkpoint
        save_ckpt(
            os.path.join(SAVE_DIR, "last.pth"),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_val=best_val,
            config=config,
        )

    print("Training done.")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved files in: {SAVE_DIR}")
    print("Expected outputs:")
    print(" - best.pth")
    print(" - last.pth")
    print(" - config.json")
    print(" - training_log.csv")
    print(" - train_subset_indices.json")
    print(" - train_subset_files.json, if TusDataset exposes .files")


if __name__ == "__main__":
    main()