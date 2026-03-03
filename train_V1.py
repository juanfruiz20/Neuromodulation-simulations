import os
import json
import math
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Data_loader import list_npz_files, split_files, compute_stats, TusDataset
from Model_Unet3D import ResUNet3D_HQ


# ----------------------------
# Reproducibilidad
# ----------------------------
def seed_all(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Grad loss (para NO suavizar bordes del haz)
# ----------------------------
def grad3d(x: torch.Tensor):
    """
    x: [B, 1, D, H, W]
    retorna grad magnitudes aproximadas (dx, dy, dz) con diferencias finitas.
    """
    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    # Pad para volver a [B,1,D,H,W]
    dx = torch.nn.functional.pad(dx, (0, 0, 0, 0, 0, 1))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1, 0, 0))
    dz = torch.nn.functional.pad(dz, (0, 1, 0, 0, 0, 0))
    return dx, dy, dz


class BeamPreservingLoss(nn.Module):
    """
    Loss base: L1
    Extra opcional: L1 sobre gradientes (preserva bordes/transiciones del haz)
    Extra opcional: peso en voxels con señal (target > thr)
    """

    def __init__(self, use_grad=True, lambda_grad=0.05, use_signal_weight=True, thr=0.05, w_signal=3.0):
        super().__init__()
        self.use_grad = use_grad
        self.lambda_grad = float(lambda_grad)
        self.use_signal_weight = use_signal_weight
        self.thr = float(thr)
        self.w_signal = float(w_signal)

        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, pred, target):
        # pred/target: [B,1,128,128,128]
        base = self.l1(pred, target)

        if self.use_signal_weight:
            # Más peso donde hay campo acústico relevante
            w = 1.0 + self.w_signal * (target > self.thr).float()
            base = base * w

        loss_l1 = base.mean()

        if not self.use_grad:
            return loss_l1

        pdx, pdy, pdz = grad3d(pred)
        tdx, tdy, tdz = grad3d(target)

        loss_grad = (
            torch.abs(pdx - tdx).mean()
            + torch.abs(pdy - tdy).mean()
            + torch.abs(pdz - tdz).mean()
        )

        return loss_l1 + self.lambda_grad * loss_grad


# ----------------------------
# Train / Val loops
# ----------------------------
def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    accum_steps=1, grad_clip=1.0, use_amp=True):
    model.train()
    total = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, (X, y) in enumerate(loader, start=1):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(use_amp and device == "cuda")):
            pred = model(X)
            loss = criterion(pred, y)
            loss = loss / accum_steps

        if device == "cuda" and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % accum_steps == 0:
            if grad_clip is not None and grad_clip > 0:
                if device == "cuda" and use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip)

            if device == "cuda" and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        # deshacemos el /accum_steps para reportar
        total += float(loss.item()) * accum_steps
    return total / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(X)
        loss = criterion(pred, y)
        total += float(loss.item())
    return total / max(1, len(loader))


# ----------------------------
# Checkpointing robusto
# ----------------------------
def save_ckpt(path, model, optimizer, scaler, epoch, best_val, stats, config):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "stats": stats,
        "config": config,
    }
    torch.save(ckpt, path)


def load_ckpt(path, model, optimizer=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    stats = ckpt.get("stats", None)
    config = ckpt.get("config", None)
    return epoch, best_val, stats, config


# ----------------------------
# MAIN
# ----------------------------
def main():
    # ===== CONFIG =====
    SEED = 123
    DATA_DIR = r"dataset_TUS_dx05_TAClike_ovoidXY_R30_thick2dx"
    SAVE_DIR = "checkpoints_unet_hq"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Con 8GB: prueba BATCH=1 y accum=2 (equivale a batch efectivo 2).
    BATCH_SIZE = 1
    ACCUM_STEPS = 2          # 1 si usas batch=2 real; 2/4 si batch=1
    NUM_WORKERS = 2
    PIN_MEMORY = True

    EPOCHS = 80
    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    # Modelo
    # si te cabe y quieres más calidad: prueba 24 (pero puede OOM)
    BASE = 16
    USE_SE = True
    OUT_POSITIVE = True      # p_max_norm >= 0

    # Loss (L1 sin log) + extra para bordes del haz (opcional)
    USE_GRAD_LOSS = True
    LAMBDA_GRAD = 0.05       # pequeño: 0.02–0.10 típico
    USE_SIGNAL_WEIGHT = True
    SIGNAL_THR = 0.05
    W_SIGNAL = 3.0

    # Scheduler simple: ReduceLROnPlateau (robusto)
    USE_SCHEDULER = True
    PLATEAU_PATIENCE = 6
    PLATEAU_FACTOR = 0.5

    # Checkpoints
    RESUME = True
    RESUME_PATH = os.path.join(SAVE_DIR, "last.pth")  # reanuda si existe

    # Split
    TRAIN_RATIO = 0.85
    VAL_RATIO = 0.10

    # ===== SETUP =====
    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        # opcional: TF32 (si tu GPU lo soporta). No rompe y puede acelerar.
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ===== DATA =====
    files = list_npz_files(DATA_DIR)
    train_files, val_files, test_files = split_files(
        files, seed=SEED, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)
    print(
        f"Total sims: {len(files)} | train {len(train_files)} | val {len(val_files)} | test {len(test_files)}")

    # Stats SOLO train
    stats = compute_stats(train_files, max_samples=128, seed=0)

    train_ds = TusDataset(train_files, stats=stats, normalize=True)
    val_ds = TusDataset(val_files,   stats=stats, normalize=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    # ===== MODEL =====
    model = ResUNet3D_HQ(
        in_ch=4, out_ch=1, base=BASE,
        norm_kind="group", use_se=USE_SE,
        out_positive=OUT_POSITIVE
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    criterion = BeamPreservingLoss(
        use_grad=USE_GRAD_LOSS,
        lambda_grad=LAMBDA_GRAD,
        use_signal_weight=USE_SIGNAL_WEIGHT,
        thr=SIGNAL_THR,
        w_signal=W_SIGNAL
    )

    scheduler = None
    if USE_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, verbose=True
        )

    # ===== RESUME =====
    start_epoch = 0
    best_val = float("inf")

    config = {
        "SEED": SEED,
        "DATA_DIR": DATA_DIR,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "BATCH_SIZE": BATCH_SIZE,
        "ACCUM_STEPS": ACCUM_STEPS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "EPOCHS": EPOCHS,
        "BASE": BASE,
        "USE_SE": USE_SE,
        "OUT_POSITIVE": OUT_POSITIVE,
        "USE_GRAD_LOSS": USE_GRAD_LOSS,
        "LAMBDA_GRAD": LAMBDA_GRAD,
        "USE_SIGNAL_WEIGHT": USE_SIGNAL_WEIGHT,
        "SIGNAL_THR": SIGNAL_THR,
        "W_SIGNAL": W_SIGNAL,
    }

    if RESUME and os.path.exists(RESUME_PATH):
        print(f"🔄 Resuming from: {RESUME_PATH}")
        start_epoch, best_val, stats_ckpt, _ = load_ckpt(
            RESUME_PATH, model, optimizer, scaler, device=device)
        if stats_ckpt is not None:
            stats = stats_ckpt  # por si reanudas exactamente mismo normalizado
        print(f"   start_epoch={start_epoch} | best_val={best_val:.6f}")

    # Guardar config para trazabilidad
    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ===== TRAIN =====
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            accum_steps=ACCUM_STEPS, grad_clip=1.0, use_amp=True
        )
        va = eval_one_epoch(model, val_loader, criterion, device)

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d} | train={tr:.6f} | val={va:.6f} | lr={lr_now:.2e} | time={dt:.1f}s")

        # Scheduler
        if scheduler is not None:
            scheduler.step(va)

        # Save last
        save_ckpt(os.path.join(SAVE_DIR, "last.pth"), model,
                  optimizer, scaler, epoch, best_val, stats, config)

        # Save best
        if va < best_val:
            best_val = va
            save_ckpt(os.path.join(SAVE_DIR, "best.pth"), model,
                      optimizer, scaler, epoch, best_val, stats, config)
            print(f"✅ New best val: {best_val:.6f}")

    print("✅ Training done. Best val:", best_val)


if __name__ == "__main__":
    main()
