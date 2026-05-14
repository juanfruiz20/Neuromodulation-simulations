import os
import csv
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


def init_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "train_g_total",
                "train_g_recon",
                "train_g_adv",
                "train_d_total",
                "train_d_real_mean",
                "train_d_fake_mean",
                "val_g_total",
                "val_g_recon",
                "val_g_adv",
                "val_d_total",
                "val_d_real_mean",
                "val_d_fake_mean",
                "val_loss_global",
                "val_loss_focus",
                "val_loss_grad",
                "val_loss_tube",
                "val_loss_profile",
                "val_peak_rel_err",
                "val_peak_loc_err_mm",
                "val_dice50",
                "val_mse_brain",
                "lambda_adv",
                "lr_G",
                "lr_D",
                "time_sec",
            ])


def append_csv(
    csv_path,
    epoch,
    train_stats,
    val_stats,
    extra_val_stats,
    lambda_adv,
    lr_G,
    lr_D,
    time_sec
):
    def _pick(d, k):
        if d is None:
            return float("nan")
        return d.get(k, float("nan"))

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            epoch,
            f"{train_stats['g_total']:.8f}",
            f"{train_stats['g_recon']:.8f}",
            f"{train_stats['g_adv']:.8f}",
            f"{train_stats['d_total']:.8f}",
            f"{train_stats['d_real_mean']:.8f}",
            f"{train_stats['d_fake_mean']:.8f}",
            f"{val_stats['g_total']:.8f}",
            f"{val_stats['g_recon']:.8f}",
            f"{val_stats['g_adv']:.8f}",
            f"{val_stats['d_total']:.8f}",
            f"{val_stats['d_real_mean']:.8f}",
            f"{val_stats['d_fake_mean']:.8f}",
            f"{val_stats['loss_global']:.8f}",
            f"{val_stats['loss_focus']:.8f}",
            f"{val_stats['loss_grad']:.8f}",
            f"{val_stats['loss_tube']:.8f}",
            f"{val_stats['loss_profile']:.8f}",
            f"{_pick(extra_val_stats, 'peak_rel_err'):.8f}",
            f"{_pick(extra_val_stats, 'peak_loc_err_mm'):.8f}",
            f"{_pick(extra_val_stats, 'dice50'):.8f}",
            f"{_pick(extra_val_stats, 'mse_brain'):.8f}",
            f"{lambda_adv:.8f}",
            f"{lr_G:.8e}",
            f"{lr_D:.8e}",
            f"{time_sec:.2f}",
        ])


def log_epoch_to_tensorboard(
    writer: SummaryWriter,
    epoch: int,
    train_stats: Dict[str, float],
    val_stats: Dict[str, float],
    lambda_adv: float,
    lr_G: float,
    lr_D: float
):
    writer.add_scalar("loss/train_G_total", train_stats["g_total"], epoch)
    writer.add_scalar("loss/train_G_recon", train_stats["g_recon"], epoch)
    writer.add_scalar("loss/train_G_adv", train_stats["g_adv"], epoch)
    writer.add_scalar("loss/train_D_total", train_stats["d_total"], epoch)

    writer.add_scalar("loss/val_G_total", val_stats["g_total"], epoch)
    writer.add_scalar("loss/val_G_recon", val_stats["g_recon"], epoch)
    writer.add_scalar("loss/val_G_adv", val_stats["g_adv"], epoch)
    writer.add_scalar("loss/val_D_total", val_stats["d_total"], epoch)

    writer.add_scalar("recon/val_global", val_stats["loss_global"], epoch)
    writer.add_scalar("recon/val_focus", val_stats["loss_focus"], epoch)
    writer.add_scalar("recon/val_grad", val_stats["loss_grad"], epoch)
    writer.add_scalar("recon/val_tube", val_stats["loss_tube"], epoch)
    writer.add_scalar("recon/val_profile", val_stats["loss_profile"], epoch)

    writer.add_scalar(
        "scores/train_D_real_mean",
        train_stats["d_real_mean"],
        epoch
    )

    writer.add_scalar(
        "scores/train_D_fake_mean",
        train_stats["d_fake_mean"],
        epoch
    )

    writer.add_scalar(
        "scores/val_D_real_mean",
        val_stats["d_real_mean"],
        epoch
    )

    writer.add_scalar(
        "scores/val_D_fake_mean",
        val_stats["d_fake_mean"],
        epoch
    )

    writer.add_scalar("schedule/lambda_adv", lambda_adv, epoch)
    writer.add_scalar("optim/lr_G", lr_G, epoch)
    writer.add_scalar("optim/lr_D", lr_D, epoch)

    writer.flush()


def log_extra_metrics_to_tensorboard(
    writer,
    epoch: int,
    stats: Dict[str, float],
    split: str = "val_extra"
):
    for k, v in stats.items():
        writer.add_scalar(f"{split}/{k}", v, epoch)

    writer.flush()