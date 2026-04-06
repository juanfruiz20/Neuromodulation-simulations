import os
import csv
import torch


def save_ckpt(path, model, optimizer, scaler, epoch, best_val, stats, config):
    """Save checkpoint with model, optimizer, and metadata."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_val": float(best_val),
        "stats": stats,
        "config": config,
    }
    torch.save(ckpt, path)


def load_ckpt(path, model, optimizer=None, scaler=None, device="cpu"):
    """Load checkpoint and restore state."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    stats = ckpt.get("stats", None)
    config = ckpt.get("config", None)
    return epoch, best_val, stats, config


def init_csv(csv_path):
    """Initialize CSV log file with headers."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_loss_global",
                "val_loss_focus",
                "val_loss_grad",
                "val_loss_peak",
                "val_loss_loc",
                "r_peak",
                "r_loc",
                "beta_peak",
                "beta_loc",
                "lambda_peak_eff",
                "lambda_loc_eff",
                "lr",
                "time_sec",
            ])


def append_csv(csv_path, epoch, train_loss, val_loss, val_comp, lr, time_sec):
    """Append training metrics to CSV log."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.8f}",
            f"{val_loss:.8f}",
            f"{val_comp['loss_global']:.8f}",
            f"{val_comp['loss_focus']:.8f}",
            f"{val_comp['loss_grad']:.8f}",
            f"{val_comp['loss_peak']:.8f}",
            f"{val_comp['loss_loc']:.8f}",
            f"{val_comp['r_peak']:.4f}",
            f"{val_comp['r_loc']:.4f}",
            f"{val_comp['beta_peak']:.4f}",
            f"{val_comp['beta_loc']:.4f}",
            f"{val_comp['lambda_peak_eff']:.6f}",
            f"{val_comp['lambda_loc_eff']:.6f}",
            f"{lr:.8e}",
            f"{time_sec:.2f}",
        ])
