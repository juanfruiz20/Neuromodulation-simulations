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
# NORMALIZATION HELPER
# =========================================================
def norm3d(ch: int, kind: str = "group", groups: int = 8):
    if kind == "instance":
        return nn.InstanceNorm3d(ch, affine=True)

    return nn.GroupNorm(
        num_groups=min(groups, ch),
        num_channels=ch,
    )


# =========================================================
# SQUEEZE-EXCITATION
# =========================================================
class SEBlock3D(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()

        r = max(1, ch // reduction)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(ch, r, kernel_size=1)
        self.fc2 = nn.Conv3d(r, ch, kernel_size=1)

    def forward(self, x):
        s = self.pool(x)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))

        return x * s


# =========================================================
# RESIDUAL BLOCK 3D
# =========================================================
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.n1 = norm3d(out_ch, kind=norm_kind)

        self.conv2 = nn.Conv3d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.n2 = norm3d(out_ch, kind=norm_kind)

        self.act = nn.SiLU(inplace=True)

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=1,
                bias=False,
            )

        self.se = SEBlock3D(out_ch, reduction=8) if use_se else nn.Identity()

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)

        x = self.act(self.n1(self.conv1(x)))
        x = self.n2(self.conv2(x))
        x = self.se(x)

        x = self.act(x + identity)

        return x


# =========================================================
# DOWN / UP BLOCKS
# =========================================================
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()

        self.down = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.n = norm3d(out_ch, kind=norm_kind)
        self.act = nn.SiLU(inplace=True)

        self.block = ResBlock3D(
            out_ch,
            out_ch,
            norm_kind=norm_kind,
            use_se=use_se,
        )

    def forward(self, x):
        x = self.act(self.n(self.down(x)))
        x = self.block(x)

        return x


class UpConcat(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=False,
        )

        self.conv_up = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=1,
            bias=False,
        )

        self.block = ResBlock3D(
            out_ch + skip_ch,
            out_ch,
            norm_kind=norm_kind,
            use_se=use_se,
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv_up(x)

        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(
                x,
                size=skip.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        x = torch.cat([skip, x], dim=1)
        x = self.block(x)

        return x


# =========================================================
# REDUCED ResUNet3D_HQ: 3 LEVELS INSTEAD OF 4
# =========================================================
class ResUNet3D_HQ_3L(nn.Module):
    """
    Reduced-depth version of ResUNet3D_HQ.

    Differences vs original ResUNet3D_HQ:
    - Original: 4 downsampling levels, bottleneck at 8^3, max channels base*32.
    - This version: 3 downsampling levels, bottleneck at 16^3, max channels base*16.

    Keeps:
    - ResBlock3D
    - SE blocks
    - GroupNorm / InstanceNorm option
    - SiLU activation
    - trilinear upsampling + 1x1 conv
    - U-Net skip connections
    - Softplus positive output
    """

    def __init__(
        self,
        in_ch=2,
        out_ch=1,
        base=16,
        norm_kind="group",
        use_se=True,
        out_positive=True,
    ):
        super().__init__()

        self.out_positive = out_positive
        self.out_act = nn.Softplus() if out_positive else nn.Identity()

        # -------------------------
        # Encoder
        # -------------------------
        self.stem = ResBlock3D(
            in_ch,
            base,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # 128^3

        self.d1 = Down(
            base,
            base * 2,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # 64^3

        self.d2 = Down(
            base * 2,
            base * 4,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # 32^3

        self.d3 = Down(
            base * 4,
            base * 8,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # 16^3

        # -------------------------
        # Bottleneck
        # -------------------------
        self.mid = ResBlock3D(
            base * 8,
            base * 16,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # 16^3

        # -------------------------
        # Decoder
        # -------------------------
        self.u3 = UpConcat(
            base * 16,
            base * 4,
            base * 4,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # -> 32^3

        self.u2 = UpConcat(
            base * 4,
            base * 2,
            base * 2,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # -> 64^3

        self.u1 = UpConcat(
            base * 2,
            base,
            base,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # -> 128^3

        # -------------------------
        # Head
        # -------------------------
        self.head = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        s0 = self.stem(x)   # base, 128^3
        s1 = self.d1(s0)    # 2b, 64^3
        s2 = self.d2(s1)    # 4b, 32^3
        s3 = self.d3(s2)    # 8b, 16^3

        # Bottleneck
        m = self.mid(s3)    # 16b, 16^3

        # Decoder
        x = self.u3(m, s2)  # 4b, 32^3
        x = self.u2(x, s1)  # 2b, 64^3
        x = self.u1(x, s0)  # b, 128^3

        x = self.head(x)
        x = self.out_act(x)

        return x


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

    SAVE_DIR = "checkpoints_resunet3d_hq_3L_l1_fulldata_100epochs"
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

    # Reduced-depth ResUNet3D_HQ config
    BASE = 16
    NORM_KIND = "group"
    USE_SE = True
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
    model = ResUNet3D_HQ_3L(
        in_ch=2,
        out_ch=1,
        base=BASE,
        norm_kind=NORM_KIND,
        use_se=USE_SE,
        out_positive=OUT_POSITIVE,
    ).to(device)

    print("Training reduced-depth ResUNet3D_HQ_3L baseline from scratch.")
    print("Configuration:")
    print(f"  Model type: ResUNet3D_HQ_3L")
    print(f"  Train fraction: 1.0")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Save epoch 50: {SAVE_EPOCH_50}")
    print(f"  Base channels: {BASE}")
    print(f"  Norm kind: {NORM_KIND}")
    print(f"  Use SE: {USE_SE}")
    print(f"  Out positive: {OUT_POSITIVE}")
    print(f"  Loss: {LOSS_TYPE}")
    print("No adversarial loss.")
    print("No focus-aware, peak-aware, gradient, location, or tube-aware losses.")

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
        "experiment_name": "resunet3d_hq_3L_l1_fulldata_100epochs",
        "description": (
            "Reduced-depth ResUNet3D_HQ baseline trained with the full training "
            "dataset for 100 epochs using only a voxel-wise L1 reconstruction loss. "
            "Compared with the full ResUNet3D_HQ, this model removes the deepest "
            "encoder-decoder level, reducing the number of downsampling stages from "
            "four to three while preserving residual blocks, SE blocks, GroupNorm, "
            "SiLU activations, trilinear upsampling, U-Net skip connections and "
            "Softplus positive output."
        ),
        "model_type": "ResUNet3D_HQ_3L",
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
        "norm_kind": NORM_KIND,
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