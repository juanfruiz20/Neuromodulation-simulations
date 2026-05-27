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

try:
    from torch.amp import GradScaler
except Exception:
    from torch.cuda.amp import GradScaler


# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_all(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# BASIC RESIDUAL SE 3D U-NET
# =========================================================
def make_group_norm(num_channels, preferred_groups=4):
    """
    Creates a valid GroupNorm layer.
    GroupNorm is useful for 3D volumes with batch_size=1.
    """
    groups = min(preferred_groups, num_channels)

    while groups > 1 and (num_channels % groups != 0):
        groups -= 1

    return nn.GroupNorm(groups, num_channels)


class SEBlock3D(nn.Module):
    """
    Simple Squeeze-and-Excitation block for 3D feature maps.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.pool(x)
        weights = self.fc(weights)
        return x * weights


class ResidualSEBlock3D(nn.Module):
    """
    Simple residual block with GroupNorm and optional SE.

    Conv3D -> GroupNorm -> ReLU -> Conv3D -> GroupNorm -> SE -> skip -> ReLU
    """
    def __init__(self, in_ch, out_ch, norm_groups=4, use_se=True):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = make_group_norm(out_ch, norm_groups)

        self.conv2 = nn.Conv3d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = make_group_norm(out_ch, norm_groups)

        if in_ch != out_ch:
            self.skip = nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=1,
                bias=False,
            )
        else:
            self.skip = nn.Identity()

        self.se = SEBlock3D(out_ch, reduction=8) if use_se else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.se(out)

        out = out + identity
        out = self.act(out)

        return out


class BasicResSEUNet3D(nn.Module):
    """
    Intermediate baseline architecture.

    This model is stronger than a plain BasicUNet3D because it keeps:
    - residual blocks
    - SE blocks
    - GroupNorm

    But it is simpler than ResUNet3D_HQ because it uses:
    - a reduced channel capacity
    - a simple 3-level encoder-decoder
    - simple residual-SE blocks
    - no advanced HQ blocks
    - no adversarial component
    """

    def __init__(
        self,
        in_ch=2,
        out_ch=1,
        base=8,
        norm_groups=4,
        use_se=True,
        out_positive=True,
        out_activation="relu",
    ):
        super().__init__()

        self.out_positive = out_positive
        self.out_activation = out_activation

        # Encoder
        self.enc1 = ResidualSEBlock3D(
            in_ch=in_ch,
            out_ch=base,
            norm_groups=norm_groups,
            use_se=use_se,
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ResidualSEBlock3D(
            in_ch=base,
            out_ch=base * 2,
            norm_groups=norm_groups,
            use_se=use_se,
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ResidualSEBlock3D(
            in_ch=base * 2,
            out_ch=base * 4,
            norm_groups=norm_groups,
            use_se=use_se,
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResidualSEBlock3D(
            in_ch=base * 4,
            out_ch=base * 8,
            norm_groups=norm_groups,
            use_se=use_se,
        )

        # Decoder
        self.up3 = nn.ConvTranspose3d(
            base * 8,
            base * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = ResidualSEBlock3D(
            in_ch=base * 8,
            out_ch=base * 4,
            norm_groups=norm_groups,
            use_se=use_se,
        )

        self.up2 = nn.ConvTranspose3d(
            base * 4,
            base * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ResidualSEBlock3D(
            in_ch=base * 4,
            out_ch=base * 2,
            norm_groups=norm_groups,
            use_se=use_se,
        )

        self.up1 = nn.ConvTranspose3d(
            base * 2,
            base,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = ResidualSEBlock3D(
            in_ch=base * 2,
            out_ch=base,
            norm_groups=norm_groups,
            use_se=use_se,
        )

        self.out_conv = nn.Conv3d(base, out_ch, kernel_size=1)

    def apply_output_activation(self, out):
        if not self.out_positive:
            return out

        if self.out_activation == "relu":
            return F.relu(out)

        if self.out_activation == "softplus":
            return F.softplus(out)

        raise ValueError(f"Unknown out_activation: {self.out_activation}")

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        out = self.apply_output_activation(out)

        return out


# =========================================================
# CHECKPOINTING
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
# CSV LOGGING
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
# METRICS
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
# TRAIN LOOP
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
# VALIDATION LOOP
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
# DATALOADER HELPER
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
# MAIN
# =========================================================
def main():
    # =========================
    # CONFIG
    # =========================
    SEED = 123

    SAVE_DIR = "checkpoints_basicresseunet3d_fulldata_100epochs"
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/train"
    VAL_DIR   = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/val"
    TEST_DIR  = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

    BATCH_SIZE = 1
    NUM_WORKERS = 2

    EPOCHS = 100
    SAVE_EPOCH_50 = True

    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    # Intermediate baseline architecture
    BASE = 8
    NORM_GROUPS = 4
    USE_SE = True
    OUT_POSITIVE = True
    OUT_ACTIVATION = "relu"  # "relu" recommended to avoid positive background bias

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
    train_ds = TusDataset(TRAIN_DIR)
    val_ds = TusDataset(VAL_DIR)
    test_ds = TusDataset(TEST_DIR)

    print(
        f"Train: {len(train_ds)} | "
        f"Val: {len(val_ds)} | "
        f"Test: {len(test_ds)}"
    )

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
    model = BasicResSEUNet3D(
        in_ch=2,
        out_ch=1,
        base=BASE,
        norm_groups=NORM_GROUPS,
        use_se=USE_SE,
        out_positive=OUT_POSITIVE,
        out_activation=OUT_ACTIVATION,
    ).to(device)

    print("Training BasicResSEUNet3D intermediate baseline from scratch.")
    print("Configuration:")
    print(f"  Model type: BasicResSEUNet3D")
    print(f"  Train fraction: 1.0")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Save epoch 50: {SAVE_EPOCH_50}")
    print(f"  Base channels: {BASE}")
    print(f"  GroupNorm groups: {NORM_GROUPS}")
    print(f"  Use SE: {USE_SE}")
    print(f"  Out positive: {OUT_POSITIVE}")
    print(f"  Out activation: {OUT_ACTIVATION}")
    print(f"  Loss: {LOSS_TYPE}")
    print("No ResUNet3D_HQ, no adversarial loss.")
    print("No focus-aware, peak-aware, gradient, or location losses.")

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
        "experiment_name": "basicresseunet3d_fulldata_100epochs",
        "description": (
            "Intermediate 3D U-Net baseline trained with the full training dataset "
            "for 100 epochs. The model is a simplified encoder-decoder U-Net with "
            "skip connections, residual blocks, squeeze-and-excitation modules, "
            "GroupNorm, reduced channel capacity, and a positive output constraint. "
            "It does not use ResUNet3D_HQ, adversarial loss, focus-aware loss, "
            "peak-aware loss, gradient loss, location loss, peak-value loss, or "
            "peak-ROI loss."
        ),
        "model_type": "BasicResSEUNet3D",
        "seed": SEED,
        "save_dir": SAVE_DIR,
        "train_dir": TRAIN_DIR,
        "val_dir": VAL_DIR,
        "test_dir": TEST_DIR,
        "train_fraction": 1.0,
        "train_full_size": len(train_ds),
        "train_used_size": len(train_ds),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "save_epoch_50": SAVE_EPOCH_50,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "base": BASE,
        "norm_kind": "group",
        "norm_groups": NORM_GROUPS,
        "use_se": USE_SE,
        "out_positive": OUT_POSITIVE,
        "out_activation": OUT_ACTIVATION,
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
        "saved_checkpoints": ["best.pth", "last.pth", "epoch_050.pth"],
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

        # Save fixed checkpoint at epoch 50
        if SAVE_EPOCH_50 and epoch == 50:
            save_ckpt(
                os.path.join(SAVE_DIR, "epoch_050.pth"),
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_val=best_val,
                config=config,
            )

            print("Saved checkpoint: epoch_050.pth")

    print("Training done.")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved files in: {SAVE_DIR}")
    print("Expected outputs:")
    print(" - best.pth")
    print(" - last.pth")
    print(" - epoch_050.pth")
    print(" - config.json")
    print(" - training_log.csv")


if __name__ == "__main__":
    main()