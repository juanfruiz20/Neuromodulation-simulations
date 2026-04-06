import os
import json
import time
import numpy as np
from contextlib import nullcontext

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Data_loaderV2 import TusDataset
from Unet3D_V2 import ResUNet3D_HQ
from utils import seed_all
from losses import StableFocusAwareTUSLoss
from callbacks import VisualCallback
from checkpointing import save_ckpt, load_ckpt, init_csv, append_csv

# AMP GradScaler compatibility
try:
    from torch.amp import GradScaler
except Exception:
    from torch.cuda.amp import GradScaler


def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    epoch: int,
                    accum_steps=1, grad_clip=2.0, use_amp=True):
    """Train for one epoch with gradient accumulation."""
    model.train()

    total = 0.0
    n_batches = 0
    accum_counter = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (X, y) in enumerate(loader, start=1):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (
            use_amp and device == "cuda") else nullcontext()

        with amp_ctx:
            pred = model(X)

        loss = criterion(pred, y, epoch=epoch) / accum_steps

        if not torch.isfinite(loss):
            print(
                f"⚠️ Non-finite loss in train, epoch {epoch}, step {step}. Batch skipped.")
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0
            continue

        if device == "cuda" and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_counter += 1

        if accum_counter >= accum_steps:
            if grad_clip is not None and grad_clip > 0:
                if device == "cuda" and use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(grad_clip))

            if device == "cuda" and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0

        total += float(loss.item()) * accum_steps
        n_batches += 1

    if accum_counter > 0:
        if grad_clip is not None and grad_clip > 0:
            if device == "cuda" and use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(grad_clip))

        if device == "cuda" and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

    return total / max(1, n_batches)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, epoch: int, use_amp=True):
    """Evaluate for one epoch."""
    model.eval()

    total = 0.0
    n_batches = 0

    comp_sums = {
        "loss_global": 0.0,
        "loss_focus": 0.0,
        "loss_grad": 0.0,
        "loss_peak": 0.0,
        "loss_loc": 0.0,
        "r_peak": 0.0,
        "r_loc": 0.0,
        "beta_peak": 0.0,
        "beta_loc": 0.0,
        "lambda_peak_eff": 0.0,
        "lambda_loc_eff": 0.0,
    }

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (
            use_amp and device == "cuda") else nullcontext()
        with amp_ctx:
            pred = model(X)

        loss, comp = criterion(pred, y, epoch=epoch, return_dict=True)

        if not torch.isfinite(loss):
            print(f"⚠️ Non-finite loss in val, epoch {epoch}. Batch skipped.")
            continue

        total += float(loss.item())
        n_batches += 1

        for k in comp_sums.keys():
            comp_sums[k] += float(comp[k])

    if n_batches == 0:
        avg_total = float("inf")
        avg_comp = {k: float("inf") for k in comp_sums.keys()}
    else:
        avg_total = total / n_batches
        avg_comp = {k: v / n_batches for k, v in comp_sums.items()}

    return avg_total, avg_comp


def main():
    # =========================
    # CONFIG
    # =========================
    SEED = 123
    SAVE_DIR = "checkpoints_unet_V2"
    os.makedirs(SAVE_DIR, exist_ok=True)

    BATCH_SIZE = 1
    ACCUM_STEPS = 2
    NUM_WORKERS = 2
    PIN_MEMORY = True

    EPOCHS = 100
    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    BASE = 16
    USE_SE = True
    OUT_POSITIVE = True

    USE_SCHEDULER = True
    PLATEAU_PATIENCE = 6
    PLATEAU_FACTOR = 0.5

    RESUME = False
    RESUME_PATH = os.path.join(SAVE_DIR, "last.pth")

    TRAIN_RATIO = 0.85
    VAL_RATIO = 0.10

    SAVE_EVERY_N_EPOCHS = 10
    SPECIFIC_SAVE_EPOCHS = {70, 80, 90, 100}

    CSV_PATH = os.path.join(SAVE_DIR, "training_log.csv")

    VISUAL_EVERY_N_EPOCHS = 10
    VISUAL_FIXED_INDICES = (0, 5, 10)
    VISUAL_SAVE_RAW_NPZ = True

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
    TRAIN_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/train"
    VAL_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/val"
    TEST_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

    train_ds = TusDataset(TRAIN_DIR)
    val_ds = TusDataset(VAL_DIR)
    test_ds = TusDataset(TEST_DIR)

    print(
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None
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
        out_positive=OUT_POSITIVE
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)

    try:
        scaler = GradScaler("cuda", enabled=(device == "cuda"))
    except TypeError:
        scaler = GradScaler(enabled=(device == "cuda"))

    criterion = StableFocusAwareTUSLoss(
        lambda_global=1.0,
        lambda_focus=2.0,
        lambda_peak=0.50,
        lambda_loc=0.25,
        lambda_grad=0.05,

        focus_frac=0.50,
        focus_min_thr=0.08,
        focus_dilate_ks=7,
        peak_frac=0.85,

        global_peak_weight=4.0,
        global_peak_gamma=2.0,

        peak_warmup_start=10,
        peak_warmup_end=30,
        loc_warmup_start=20,
        loc_warmup_end=45,

        beta_peak_start=8.0,
        beta_peak_end=25.0,
        beta_loc_start=8.0,
        beta_loc_end=35.0,

        eps=1e-6,
    )

    visual_cb = VisualCallback(
        save_dir=SAVE_DIR,
        val_dataset=val_ds,
        device=device,
        every_n_epochs=VISUAL_EVERY_N_EPOCHS,
        fixed_indices=VISUAL_FIXED_INDICES,
        use_amp=True,
        save_raw_npz=VISUAL_SAVE_RAW_NPZ,
    )

    scheduler = None
    if USE_SCHEDULER:
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=PLATEAU_FACTOR,
                patience=PLATEAU_PATIENCE,

            )
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=PLATEAU_FACTOR,
                patience=PLATEAU_PATIENCE
            )

    # =========================
    # CONFIG & RESUME
    # =========================
    start_epoch = 0
    best_val = float("inf")
    stats = None
    config = {
        "SEED": SEED,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "BATCH_SIZE": BATCH_SIZE,
        "ACCUM_STEPS": ACCUM_STEPS,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "BASE": BASE,
        "USE_SE": USE_SE,
        "OUT_POSITIVE": OUT_POSITIVE,
        "SAVE_EVERY_N_EPOCHS": SAVE_EVERY_N_EPOCHS,
        "SPECIFIC_SAVE_EPOCHS": sorted(list(SPECIFIC_SAVE_EPOCHS)),
        "VISUAL_EVERY_N_EPOCHS": VISUAL_EVERY_N_EPOCHS,
        "VISUAL_FIXED_INDICES": list(VISUAL_FIXED_INDICES),
        "VISUAL_SAVE_RAW_NPZ": VISUAL_SAVE_RAW_NPZ,
        "LOSS": {
            "lambda_global": 1.0,
            "lambda_focus": 2.0,
            "lambda_peak": 0.50,
            "lambda_loc": 0.25,
            "lambda_grad": 0.05,

            "focus_frac": 0.50,
            "focus_min_thr": 0.08,
            "focus_dilate_ks": 7,
            "peak_frac": 0.85,

            "global_peak_weight": 4.0,
            "global_peak_gamma": 2.0,

            "peak_warmup_start": 10,
            "peak_warmup_end": 30,
            "loc_warmup_start": 20,
            "loc_warmup_end": 45,

            "beta_peak_start": 8.0,
            "beta_peak_end": 25.0,
            "beta_loc_start": 8.0,
            "beta_loc_end": 35.0,
        }
    }

    if RESUME and os.path.exists(RESUME_PATH):
        print(f"🔄 Resuming from: {RESUME_PATH}")
        start_epoch, best_val, stats_ckpt, _ = load_ckpt(
            RESUME_PATH, model, optimizer, scaler, device=device
        )
        if stats_ckpt is not None:
            stats = stats_ckpt
        print(f"   start_epoch={start_epoch} | best_val={best_val:.6f}")

    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    init_csv(CSV_PATH)

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        t0 = time.time()

        tr = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            epoch=epoch,
            accum_steps=ACCUM_STEPS,
            grad_clip=2.0,
            use_amp=True
        )

        va, va_comp = eval_one_epoch(
            model, val_loader, criterion, device,
            epoch=epoch,
            use_amp=True
        )

        prev_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None and np.isfinite(va):
            scheduler.step(va)
        new_lr = optimizer.param_groups[0]["lr"]

        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d} | "
            f"train={tr:.6f} | val={va:.6f} | "
            f"glob={va_comp['loss_global']:.4f} | "
            f"focus={va_comp['loss_focus']:.4f} | "
            f"peak={va_comp['loss_peak']:.4f} | "
            f"loc={va_comp['loss_loc']:.4f} | "
            f"lr={new_lr:.2e} | time={dt:.1f}s"
        )

        if new_lr < prev_lr:
            print(f"🔻 LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        append_csv(CSV_PATH, epoch, tr, va, va_comp, new_lr, dt)

        # Save last checkpoint
        save_ckpt(
            os.path.join(SAVE_DIR, "last.pth"),
            model, optimizer, scaler, epoch, best_val, stats, config
        )

        # Save best checkpoint
        if va < best_val:
            best_val = va
            save_ckpt(
                os.path.join(SAVE_DIR, "best.pth"),
                model, optimizer, scaler, epoch, best_val, stats, config
            )
            print(f"✅ New best val: {best_val:.6f}")

        # Periodic checkpoint saving
        if (SAVE_EVERY_N_EPOCHS >= 50 and epoch % SAVE_EVERY_N_EPOCHS == 0) or (epoch in SPECIFIC_SAVE_EPOCHS):
            save_ckpt(
                os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.pth"),
                model, optimizer, scaler, epoch, best_val, stats, config
            )
            print(f"💾 Saved checkpoint: epoch_{epoch:03d}.pth")

        # Visual callback
        visual_cb(model, epoch)

    print("✅ Training done. Best val:", best_val)


if __name__ == "__main__":
    main()
