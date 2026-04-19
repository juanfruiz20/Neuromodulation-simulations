import os
import json
import time
import random
import csv
from contextlib import nullcontext
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataloader import TusDataset
from src.modelos.ResUnet3D import ResUNet3D_HQ
from src.modelos.Discriminator3D import PatchDiscriminator3D

try:
    from torch.amp.grad_scaler import GradScaler
except Exception:
    from torch.cuda.amp import GradScaler


# =========================================================
# Reproducibilidad
# =========================================================
def seed_all(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Helpers básicos
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


def ramp_linear(epoch: int, start: int, end: int) -> float:
    if epoch < start:
        return 0.0
    if epoch >= end:
        return 1.0
    return float(epoch - start) / float(max(1, end - start))


def make_focus_roi(target: torch.Tensor, frac: float, min_thr: float, dilate_ks: int):
    peak = target.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    thr = torch.maximum(peak * frac, torch.full_like(peak, min_thr))
    roi = (target >= thr).float()

    if dilate_ks is not None and dilate_ks > 1:
        roi = F.max_pool3d(
            roi,
            kernel_size=dilate_ks,
            stride=1,
            padding=dilate_ks // 2
        )
        roi = (roi > 0).float()

    return roi, peak


# =========================================================
# Helpers geométricos para tube ROI
# =========================================================
def get_gt_peak_indices(target: torch.Tensor):
    """
    target: [B,1,D,H,W]
    retorna z_idx, y_idx, x_idx cada uno [B]
    """
    B, C, D, H, W = target.shape
    flat = target.view(B, -1)
    idx = flat.argmax(dim=1)

    yz = H * W
    z_idx = idx // yz
    rem = idx % yz
    y_idx = rem // W
    x_idx = rem % W
    return z_idx.long(), y_idx.long(), x_idx.long()


def get_source_center_from_mask(source_mask: torch.Tensor):
    """
    source_mask: [B,1,D,H,W]
    devuelve centro de masa aproximado [B,3] en coords voxel (z,y,x)
    """
    B, C, D, H, W = source_mask.shape
    device = source_mask.device

    z = torch.arange(D, device=device, dtype=torch.float32).view(1, 1, D, 1, 1)
    y = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, 1, H, 1)
    x = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, 1, W)

    w = (source_mask > 0.5).float()

    mass = w.sum(dim=(2, 3, 4), keepdim=True)
    has_mass = (mass > 0).view(B)

    zc = (w * z).sum(dim=(2, 3, 4), keepdim=False) / (mass.view(B, 1) + 1e-6)
    yc = (w * y).sum(dim=(2, 3, 4), keepdim=False) / (mass.view(B, 1) + 1e-6)
    xc = (w * x).sum(dim=(2, 3, 4), keepdim=False) / (mass.view(B, 1) + 1e-6)

    centers = torch.cat([zc, yc, xc], dim=1)

    if not bool(has_mass.all()):
        flat = source_mask.view(B, -1)
        idx = flat.argmax(dim=1)

        yz = H * W
        z_idx = (idx // yz).float()
        rem = idx % yz
        y_idx = (rem // W).float()
        x_idx = (rem % W).float()

        fallback = torch.stack([z_idx, y_idx, x_idx], dim=1)
        centers = torch.where(has_mass[:, None], centers, fallback)

    return centers


def build_tube_roi_and_tcoord(source_mask: torch.Tensor,
                              target: torch.Tensor,
                              radius_vox: int = 3):
    """
    Construye un tubo entre centro de source_mask y pico GT.
    source_mask: [B,1,D,H,W]
    target:      [B,1,D,H,W]

    returns:
      tube_roi: [B,1,D,H,W]
      t_coord:  [B,1,D,H,W] en [0,1]
      src_center: [B,3]
      peak_center: [B,3]
    """
    B, C, D, H, W = target.shape
    device = target.device
    dtype = torch.float32

    src_center = get_source_center_from_mask(
        source_mask).to(device=device, dtype=dtype)

    z_idx, y_idx, x_idx = get_gt_peak_indices(target)
    peak_center = torch.stack([
        z_idx.float(), y_idx.float(), x_idx.float()
    ], dim=1).to(device=device, dtype=dtype)

    z = torch.arange(D, device=device, dtype=dtype).view(1, D, 1, 1)
    y = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
    x = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)

    zz = z.expand(B, D, H, W)
    yy = y.expand(B, D, H, W)
    xx = x.expand(B, D, H, W)

    p = torch.stack([zz, yy, xx], dim=-1)

    a = src_center.view(B, 1, 1, 1, 3)
    b = peak_center.view(B, 1, 1, 1, 3)

    ab = b - a
    ap = p - a

    ab2 = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    t_raw = (ap * ab).sum(dim=-1, keepdim=True) / ab2
    t = t_raw.clamp(0.0, 1.0)

    closest = a + t * ab
    dist2 = ((p - closest) ** 2).sum(dim=-1, keepdim=True)

    tube_roi = (dist2 <= float(radius_vox ** 2)).float()
    tube_roi = tube_roi.permute(0, 4, 1, 2, 3).contiguous()
    t_coord = t.permute(0, 4, 1, 2, 3).contiguous()

    return tube_roi, t_coord, src_center, peak_center


def compute_profile_loss(pred: torch.Tensor,
                         target: torch.Tensor,
                         tube_roi: torch.Tensor,
                         t_coord: torch.Tensor,
                         num_bins: int = 16,
                         eps: float = 1e-6):
    """
    Compara el perfil axial dentro del tubo:
    divide el segmento en bins según t_coord y compara promedios.
    """
    device = pred.device
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device)

    losses = []
    for i in range(num_bins):
        if i < num_bins - 1:
            slab = (t_coord >= edges[i]) & (t_coord < edges[i + 1])
        else:
            slab = (t_coord >= edges[i]) & (t_coord <= edges[i + 1])

        slab = slab.float() * tube_roi

        pred_m = masked_mean(pred, slab, eps=eps)
        tgt_m = masked_mean(target, slab, eps=eps)

        l = torch.abs(pred_m - tgt_m) / (torch.abs(tgt_m) + eps)
        losses.append(l)

    return torch.stack(losses).mean()


# =========================================================
# Loss Tube-Aware
# =========================================================
class StableTubeAwareTUSLoss(nn.Module):
    def __init__(
        self,
        lambda_global=0.8,
        lambda_focus=1.5,
        lambda_grad=0.08,
        lambda_tube=1.0,
        lambda_profile=0.6,

        focus_frac=0.50,
        focus_min_thr=0.08,
        focus_dilate_ks=7,

        tube_radius_vox=3,
        profile_bins=16,

        global_peak_weight=4.0,
        global_peak_gamma=2.0,

        tube_warmup_start=3,
        tube_warmup_end=12,
        profile_warmup_start=5,
        profile_warmup_end=15,

        eps=1e-6,
    ):
        super().__init__()

        self.lambda_global = float(lambda_global)
        self.lambda_focus = float(lambda_focus)
        self.lambda_grad = float(lambda_grad)
        self.lambda_tube = float(lambda_tube)
        self.lambda_profile = float(lambda_profile)

        self.focus_frac = float(focus_frac)
        self.focus_min_thr = float(focus_min_thr)
        self.focus_dilate_ks = int(focus_dilate_ks)

        self.tube_radius_vox = int(tube_radius_vox)
        self.profile_bins = int(profile_bins)

        self.global_peak_weight = float(global_peak_weight)
        self.global_peak_gamma = float(global_peak_gamma)

        self.tube_warmup_start = int(tube_warmup_start)
        self.tube_warmup_end = int(tube_warmup_end)
        self.profile_warmup_start = int(profile_warmup_start)
        self.profile_warmup_end = int(profile_warmup_end)

        self.eps = float(eps)

    def get_schedule(self, epoch: int):
        r_tube = ramp_linear(epoch, self.tube_warmup_start,
                             self.tube_warmup_end)
        r_profile = ramp_linear(
            epoch, self.profile_warmup_start, self.profile_warmup_end)

        return {
            "r_tube": r_tube,
            "r_profile": r_profile,
            "lambda_tube_eff": self.lambda_tube * r_tube,
            "lambda_profile_eff": self.lambda_profile * r_profile,
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor, source_mask: torch.Tensor,
                epoch: int = 1, return_dict: bool = False):
        pred32 = pred.float()
        target32 = target.float()
        pred_pos = pred32.clamp_min(0.0)

        sched = self.get_schedule(epoch)

        focus_roi, peak = make_focus_roi(
            target32,
            frac=self.focus_frac,
            min_thr=self.focus_min_thr,
            dilate_ks=self.focus_dilate_ks
        )

        rel = (target32 / peak).clamp(0.0, 1.0)
        w_global = 1.0 + self.global_peak_weight * \
            rel.pow(self.global_peak_gamma)
        loss_global = (torch.abs(pred_pos - target32) * w_global).mean()

        focus_num = masked_mean(
            (pred_pos - target32).pow(2), focus_roi, eps=self.eps)
        focus_den = masked_mean(target32.pow(
            2), focus_roi, eps=self.eps).clamp_min(self.eps)
        loss_focus = focus_num / focus_den

        pdx, pdy, pdz = grad3d(pred_pos)
        tdx, tdy, tdz = grad3d(target32)

        grad_diff = torch.abs(pdx - tdx) + \
            torch.abs(pdy - tdy) + torch.abs(pdz - tdz)
        grad_ref = torch.abs(tdx) + torch.abs(tdy) + torch.abs(tdz)

        grad_num = masked_mean(grad_diff, focus_roi, eps=self.eps)
        grad_den = masked_mean(grad_ref, focus_roi,
                               eps=self.eps).clamp_min(self.eps)
        loss_grad = grad_num / grad_den

        tube_roi, t_coord, _, _ = build_tube_roi_and_tcoord(
            source_mask=source_mask,
            target=target32,
            radius_vox=self.tube_radius_vox
        )

        if sched["lambda_tube_eff"] > 0.0:
            tube_num = masked_mean(
                (pred_pos - target32).pow(2), tube_roi, eps=self.eps)
            tube_den = masked_mean(target32.pow(
                2), tube_roi, eps=self.eps).clamp_min(self.eps)
            loss_tube = tube_num / tube_den
        else:
            loss_tube = torch.zeros(
                (), device=pred.device, dtype=torch.float32)

        if sched["lambda_profile_eff"] > 0.0:
            loss_profile = compute_profile_loss(
                pred=pred_pos,
                target=target32,
                tube_roi=tube_roi,
                t_coord=t_coord,
                num_bins=self.profile_bins,
                eps=self.eps
            )
        else:
            loss_profile = torch.zeros(
                (), device=pred.device, dtype=torch.float32)

        total = (
            self.lambda_global * loss_global
            + self.lambda_focus * loss_focus
            + self.lambda_grad * loss_grad
            + sched["lambda_tube_eff"] * loss_tube
            + sched["lambda_profile_eff"] * loss_profile
        )

        if return_dict:
            comp = {
                "loss_total_recon": float(total.detach().item()),
                "loss_global": float(loss_global.detach().item()),
                "loss_focus": float(loss_focus.detach().item()),
                "loss_grad": float(loss_grad.detach().item()),
                "loss_tube": float(loss_tube.detach().item()),
                "loss_profile": float(loss_profile.detach().item()),
                "r_tube": float(sched["r_tube"]),
                "r_profile": float(sched["r_profile"]),
                "lambda_tube_eff": float(sched["lambda_tube_eff"]),
                "lambda_profile_eff": float(sched["lambda_profile_eff"]),
            }
            return total, comp

        return total


# =========================================================
# Métricas extra en VALIDATION
# =========================================================
def make_brain_mask_from_skull(skull_mask_3d: np.ndarray) -> np.ndarray:
    skull = skull_mask_3d > 0.5
    filled = binary_fill_holes(skull)
    brain = np.logical_and(filled, np.logical_not(skull))
    return brain.astype(np.float32)


def get_peak_idx_np(vol: np.ndarray):
    return np.unravel_index(np.argmax(vol), vol.shape)


def dice_coeff(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-8) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    inter = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()

    if denom == 0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


@torch.no_grad()
def eval_extra_metrics(G, loader, device, dx_mm=0.5, use_amp=True):
    G.eval()

    sums = {
        "peak_rel_err": 0.0,
        "peak_loc_err_mm": 0.0,
        "dice50": 0.0,
        "mse_brain": 0.0,
    }
    n = 0

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_amp and device == "cuda")
        else nullcontext()
    )

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with amp_ctx:
            pred = G(X).clamp_min(0.0)

        pred_np = pred.detach().float().cpu().squeeze().numpy()
        gt_np = y.detach().float().cpu().squeeze().numpy()

        # Asume X = [source_mask, skull_mask]
        x_np = X.detach().float().cpu().squeeze(0).numpy()
        if x_np.ndim == 4 and x_np.shape[0] >= 2:
            skull_np = x_np[1]
            brain_mask = make_brain_mask_from_skull(skull_np)
        else:
            brain_mask = np.ones_like(gt_np, dtype=np.float32)

        gt_peak_idx = get_peak_idx_np(gt_np)
        pr_peak_idx = get_peak_idx_np(pred_np)

        gt_peak = float(gt_np[gt_peak_idx])
        pr_peak = float(pred_np[pr_peak_idx])

        peak_rel_err = abs(pr_peak - gt_peak) / (abs(gt_peak) + 1e-8)

        peak_loc_err_vox = float(np.sqrt(
            (pr_peak_idx[0] - gt_peak_idx[0]) ** 2 +
            (pr_peak_idx[1] - gt_peak_idx[1]) ** 2 +
            (pr_peak_idx[2] - gt_peak_idx[2]) ** 2
        ))
        peak_loc_err_mm = peak_loc_err_vox * float(dx_mm)

        gt_thr50 = 0.50 * gt_peak
        pr_thr50 = 0.50 * float(pred_np.max())

        gt_mask50 = (gt_np >= gt_thr50) & (brain_mask > 0)
        pr_mask50 = (pred_np >= pr_thr50) & (brain_mask > 0)

        dice50 = dice_coeff(pr_mask50, gt_mask50)

        brain_sel = brain_mask > 0
        mse_brain = float(((pred_np - gt_np) ** 2)
                          [brain_sel].mean()) if brain_sel.any() else float("nan")

        sums["peak_rel_err"] += peak_rel_err
        sums["peak_loc_err_mm"] += peak_loc_err_mm
        sums["dice50"] += dice50
        sums["mse_brain"] += mse_brain
        n += 1

    if n == 0:
        return {k: float("inf") for k in sums.keys()}

    return {k: v / n for k, v in sums.items()}


# =========================================================
# Visual Callback
# =========================================================
class VisualCallback:
    def __init__(
        self,
        save_dir,
        val_dataset,
        device,
        every_n_epochs=10,
        fixed_indices=(0, 5, 10),
        use_amp=True,
        save_raw_npz=False,
        writer: SummaryWriter = None,
    ):
        self.device = device
        self.every_n_epochs = int(every_n_epochs)
        self.use_amp = bool(use_amp)
        self.save_raw_npz = bool(save_raw_npz)
        self.writer = writer

        self.out_dir = os.path.join(save_dir, "visuals")
        os.makedirs(self.out_dir, exist_ok=True)

        n_val = len(val_dataset)
        valid_indices = [idx for idx in fixed_indices if 0 <= idx < n_val]
        if len(valid_indices) == 0:
            valid_indices = list(range(min(3, n_val)))

        self.samples = []
        for idx in valid_indices:
            x, y = val_dataset[idx]

            if not torch.is_tensor(x):
                x = torch.from_numpy(x)
            if not torch.is_tensor(y):
                y = torch.from_numpy(y)

            self.samples.append((int(idx), x.clone().cpu(), y.clone().cpu()))

        print(
            f"VisualCallback activo con casos fijos: {[s[0] for s in self.samples]}")

    def should_run(self, epoch: int) -> bool:
        return self.every_n_epochs > 0 and epoch % self.every_n_epochs == 0

    @staticmethod
    def _to_numpy_3d(t: torch.Tensor):
        return t.detach().float().cpu().squeeze().numpy()

    @staticmethod
    def _norm_2d(a: np.ndarray):
        a = a.astype(np.float32)
        mn, mx = float(a.min()), float(a.max())
        if mx - mn < 1e-8:
            return np.zeros_like(a, dtype=np.float32)
        return (a - mn) / (mx - mn + 1e-8)

    @classmethod
    def _make_overlay(cls, gt2d: np.ndarray, pred2d: np.ndarray):
        g = cls._norm_2d(gt2d)
        p = cls._norm_2d(pred2d)
        rgb = np.stack([p, g, np.zeros_like(g)], axis=-1)
        return np.clip(rgb, 0.0, 1.0)

    @staticmethod
    def _peak_idx(vol: np.ndarray):
        return np.unravel_index(np.argmax(vol), vol.shape)

    def _build_case_figure(self, pred: np.ndarray, gt: np.ndarray, epoch: int, case_idx: int):
        pred = np.clip(pred, 0.0, None)
        err = np.abs(pred - gt)

        z_gt, y_gt, x_gt = self._peak_idx(gt)
        z_pr, y_pr, x_pr = self._peak_idx(pred)

        peak_err_vox = np.sqrt(
            (z_pr - z_gt) ** 2 +
            (y_pr - y_gt) ** 2 +
            (x_pr - x_gt) ** 2
        )

        gt_ax = gt[z_gt, :, :]
        pr_ax = pred[z_gt, :, :]
        er_ax = err[z_gt, :, :]

        gt_co = gt[:, y_gt, :]
        pr_co = pred[:, y_gt, :]
        er_co = err[:, y_gt, :]

        gt_sa = gt[:, :, x_gt]
        pr_sa = pred[:, :, x_gt]
        er_sa = err[:, :, x_gt]

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        rows = [
            ("Axial", gt_ax, pr_ax, er_ax),
            ("Coronal", gt_co, pr_co, er_co),
            ("Sagittal", gt_sa, pr_sa, er_sa),
        ]

        vmax_main = max(float(gt.max()), float(pred.max()), 1e-8)
        vmax_err = max(float(err.max()), 1e-8)

        for r, (name, g, p, e) in enumerate(rows):
            axes[r, 0].imshow(g, cmap="jet", origin="lower",
                              vmin=0, vmax=vmax_main)
            axes[r, 0].set_title(f"{name} | GT")
            axes[r, 0].axis("off")

            axes[r, 1].imshow(p, cmap="jet", origin="lower",
                              vmin=0, vmax=vmax_main)
            axes[r, 1].set_title(f"{name} | Pred")
            axes[r, 1].axis("off")

            axes[r, 2].imshow(e, cmap="magma", origin="lower",
                              vmin=0, vmax=vmax_err)
            axes[r, 2].set_title(f"{name} | |Error|")
            axes[r, 2].axis("off")

            overlay = self._make_overlay(g, p)
            axes[r, 3].imshow(overlay, origin="lower")
            axes[r, 3].set_title(f"{name} | Overlay")
            axes[r, 3].axis("off")

        gt_peak = float(gt.max())
        pr_peak = float(pred.max())
        peak_rel_err = abs(pr_peak - gt_peak) / (abs(gt_peak) + 1e-8)

        fig.suptitle(
            f"Epoch {epoch:03d} | Case {case_idx} | "
            f"GT peak idx=({z_gt},{y_gt},{x_gt}) | "
            f"Pred peak idx=({z_pr},{y_pr},{x_pr}) | "
            f"PeakErr={peak_err_vox:.2f} vox | "
            f"PeakRelErr={peak_rel_err:.4f}",
            fontsize=12
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        raw_payload = {
            "pred": pred.astype(np.float32),
            "gt": gt.astype(np.float32),
            "peak_gt": np.array([z_gt, y_gt, x_gt], dtype=np.int32),
            "peak_pred": np.array([z_pr, y_pr, x_pr], dtype=np.int32),
            "peak_err_vox": np.array([peak_err_vox], dtype=np.float32),
            "peak_rel_err": np.array([peak_rel_err], dtype=np.float32),
        }
        return fig, raw_payload

    @torch.no_grad()
    def __call__(self, model, epoch: int):
        if not self.should_run(epoch):
            return

        was_training = model.training
        model.eval()

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (self.use_amp and self.device == "cuda")
            else nullcontext()
        )

        print(
            f"Guardando visualizaciones de validación en epoch {epoch:03d}...")

        for case_idx, x_cpu, y_cpu in self.samples:
            X = x_cpu.unsqueeze(0).to(self.device, non_blocking=True)

            with amp_ctx:
                pred = model(X)

            pred_np = self._to_numpy_3d(pred.clamp_min(0.0))
            gt_np = self._to_numpy_3d(y_cpu)

            fig, raw_payload = self._build_case_figure(
                pred_np, gt_np, epoch, case_idx)

            case_dir = os.path.join(self.out_dir, f"case_{case_idx:03d}")
            os.makedirs(case_dir, exist_ok=True)

            out_png = os.path.join(case_dir, f"epoch_{epoch:03d}.png")
            fig.savefig(out_png, dpi=160, bbox_inches="tight")

            if self.writer is not None:
                self.writer.add_figure(
                    f"visuals/case_{case_idx:03d}", fig, global_step=epoch)

            plt.close(fig)

            if self.save_raw_npz:
                out_npz = os.path.join(case_dir, f"epoch_{epoch:03d}.npz")
                np.savez_compressed(out_npz, **raw_payload)

        if was_training:
            model.train()


# =========================================================
# Schedules adversariales
# =========================================================
def adv_weight_schedule(epoch: int,
                        start_epoch: int = 1,
                        ramp_end_epoch: int = 15,
                        start_value: float = 1e-3,
                        end_value: float = 1e-2) -> float:
    if epoch < start_epoch:
        return 0.0
    if epoch >= ramp_end_epoch:
        return float(end_value)

    alpha = float(epoch - start_epoch) / \
        float(max(1, ramp_end_epoch - start_epoch))
    return float((1.0 - alpha) * start_value + alpha * end_value)


# =========================================================
# Utils train
# =========================================================
def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def save_ckpt(path, G, D, optim_G, optim_D, scaler_G, scaler_D, epoch, best_val, config):
    ckpt = {
        "G": G.state_dict(),
        "D": D.state_dict(),
        "optim_G": optim_G.state_dict() if optim_G is not None else None,
        "optim_D": optim_D.state_dict() if optim_D is not None else None,
        "scaler_G": scaler_G.state_dict() if scaler_G is not None else None,
        "scaler_D": scaler_D.state_dict() if scaler_D is not None else None,
        "epoch": int(epoch),
        "best_val": float(best_val),
        "config": config,
    }
    torch.save(ckpt, path)


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


def append_csv(csv_path, epoch, train_stats, val_stats, extra_val_stats, lambda_adv, lr_G, lr_D, time_sec):
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


def log_epoch_to_tensorboard(writer: SummaryWriter,
                             epoch: int,
                             train_stats: Dict[str, float],
                             val_stats: Dict[str, float],
                             lambda_adv: float,
                             lr_G: float,
                             lr_D: float):
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

    writer.add_scalar("scores/train_D_real_mean",
                      train_stats["d_real_mean"], epoch)
    writer.add_scalar("scores/train_D_fake_mean",
                      train_stats["d_fake_mean"], epoch)
    writer.add_scalar("scores/val_D_real_mean",
                      val_stats["d_real_mean"], epoch)
    writer.add_scalar("scores/val_D_fake_mean",
                      val_stats["d_fake_mean"], epoch)

    writer.add_scalar("schedule/lambda_adv", lambda_adv, epoch)
    writer.add_scalar("optim/lr_G", lr_G, epoch)
    writer.add_scalar("optim/lr_D", lr_D, epoch)

    writer.flush()


def log_extra_metrics_to_tensorboard(writer, epoch: int, stats: Dict[str, float], split: str = "val_extra"):
    for k, v in stats.items():
        writer.add_scalar(f"{split}/{k}", v, epoch)
    writer.flush()


# =========================================================
# Train / Eval loops
# =========================================================
def train_one_epoch(G, D,
                    loader,
                    optim_G, optim_D,
                    scaler_G, scaler_D,
                    recon_criterion, adv_criterion,
                    device, epoch: int,
                    lambda_adv: float,
                    grad_clip_G: float = 2.0,
                    grad_clip_D: float = 1.0,
                    use_amp: bool = True) -> Dict[str, float]:
    G.train()
    D.train()

    sums = {
        "g_total": 0.0,
        "g_recon": 0.0,
        "g_adv": 0.0,
        "d_total": 0.0,
        "d_real_mean": 0.0,
        "d_fake_mean": 0.0,
    }
    n_batches = 0

    for step, (X, y) in enumerate(loader, start=1):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        source_mask = X[:, 0:1]

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (use_amp and device == "cuda")
            else nullcontext()
        )

        # -------------------------
        # A) Train D
        # -------------------------
        optim_D.zero_grad(set_to_none=True)

        with torch.no_grad():
            with amp_ctx:
                y_hat_det = G(X)

        with amp_ctx:
            pred_real = D(X, y)
            pred_fake = D(X, y_hat_det)

            loss_d_real = adv_criterion(pred_real, torch.ones_like(pred_real))
            loss_d_fake = adv_criterion(pred_fake, torch.zeros_like(pred_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

        if not torch.isfinite(loss_d):
            print(
                f"[WARN] Non-finite loss_D at epoch={epoch}, step={step}. Batch skipped.")
            continue

        if device == "cuda" and use_amp:
            scaler_D.scale(loss_d).backward()
            scaler_D.unscale_(optim_D)
            if grad_clip_D is not None and grad_clip_D > 0:
                torch.nn.utils.clip_grad_norm_(
                    D.parameters(), max_norm=float(grad_clip_D))
            scaler_D.step(optim_D)
            scaler_D.update()
        else:
            loss_d.backward()
            if grad_clip_D is not None and grad_clip_D > 0:
                torch.nn.utils.clip_grad_norm_(
                    D.parameters(), max_norm=float(grad_clip_D))
            optim_D.step()

        # -------------------------
        # B) Train G
        # -------------------------
        optim_G.zero_grad(set_to_none=True)

        with amp_ctx:
            y_hat = G(X)

            loss_recon, _ = recon_criterion(
                y_hat, y,
                source_mask=source_mask,
                epoch=epoch,
                return_dict=True
            )

            pred_fake_for_g = D(X, y_hat)
            loss_adv_g = adv_criterion(
                pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g = loss_recon + lambda_adv * loss_adv_g

        if not torch.isfinite(loss_g):
            print(
                f"[WARN] Non-finite loss_G at epoch={epoch}, step={step}. Batch skipped.")
            continue

        if device == "cuda" and use_amp:
            scaler_G.scale(loss_g).backward()
            scaler_G.unscale_(optim_G)
            if grad_clip_G is not None and grad_clip_G > 0:
                torch.nn.utils.clip_grad_norm_(
                    G.parameters(), max_norm=float(grad_clip_G))
            scaler_G.step(optim_G)
            scaler_G.update()
        else:
            loss_g.backward()
            if grad_clip_G is not None and grad_clip_G > 0:
                torch.nn.utils.clip_grad_norm_(
                    G.parameters(), max_norm=float(grad_clip_G))
            optim_G.step()

        sums["g_total"] += float(loss_g.item())
        sums["g_recon"] += float(loss_recon.item())
        sums["g_adv"] += float(loss_adv_g.item())
        sums["d_total"] += float(loss_d.item())
        sums["d_real_mean"] += float(pred_real.detach().mean().item())
        sums["d_fake_mean"] += float(pred_fake.detach().mean().item())
        n_batches += 1

    if n_batches == 0:
        return {k: float("inf") for k in sums.keys()}

    return {k: v / n_batches for k, v in sums.items()}


@torch.no_grad()
def eval_one_epoch(G, D,
                   loader,
                   recon_criterion, adv_criterion,
                   device, epoch: int,
                   lambda_adv: float,
                   use_amp: bool = True) -> Dict[str, float]:
    G.eval()
    D.eval()

    sums = {
        "g_total": 0.0,
        "g_recon": 0.0,
        "g_adv": 0.0,
        "d_total": 0.0,
        "d_real_mean": 0.0,
        "d_fake_mean": 0.0,
        "loss_global": 0.0,
        "loss_focus": 0.0,
        "loss_grad": 0.0,
        "loss_tube": 0.0,
        "loss_profile": 0.0,
    }
    n_batches = 0

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_amp and device == "cuda")
        else nullcontext()
    )

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        source_mask = X[:, 0:1]

        with amp_ctx:
            y_hat = G(X)

            loss_recon, recon_comp = recon_criterion(
                y_hat, y,
                source_mask=source_mask,
                epoch=epoch,
                return_dict=True
            )

            pred_fake_for_g = D(X, y_hat)
            loss_adv_g = adv_criterion(
                pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g = loss_recon + lambda_adv * loss_adv_g

            pred_real = D(X, y)
            pred_fake = D(X, y_hat.detach())
            loss_d_real = adv_criterion(pred_real, torch.ones_like(pred_real))
            loss_d_fake = adv_criterion(pred_fake, torch.zeros_like(pred_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

        if (not torch.isfinite(loss_g)) or (not torch.isfinite(loss_d)):
            print(
                f"[WARN] Non-finite validation loss at epoch={epoch}. Batch skipped.")
            continue

        sums["g_total"] += float(loss_g.item())
        sums["g_recon"] += float(loss_recon.item())
        sums["g_adv"] += float(loss_adv_g.item())
        sums["d_total"] += float(loss_d.item())
        sums["d_real_mean"] += float(pred_real.detach().mean().item())
        sums["d_fake_mean"] += float(pred_fake.detach().mean().item())
        sums["loss_global"] += float(recon_comp["loss_global"])
        sums["loss_focus"] += float(recon_comp["loss_focus"])
        sums["loss_grad"] += float(recon_comp["loss_grad"])
        sums["loss_tube"] += float(recon_comp["loss_tube"])
        sums["loss_profile"] += float(recon_comp["loss_profile"])
        n_batches += 1

    if n_batches == 0:
        return {k: float("inf") for k in sums.keys()}

    return {k: v / n_batches for k, v in sums.items()}


# =========================================================
# MAIN
# =========================================================
def main():
    # =========================
    # CONFIG
    # =========================
    SEED = 123

    SAVE_DIR = "checkpoints_cgan_exp01_fromscratch_tb"
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/train"
    VAL_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/val"
    # no se usa durante train
    TEST_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

    CSV_PATH = os.path.join(SAVE_DIR, "training_log.csv")
    TB_DIR = os.path.join(SAVE_DIR, "tensorboard")

    BATCH_SIZE = 1
    NUM_WORKERS = 2
    PIN_MEMORY = True

    EPOCHS = 100
    LR_G = 1e-4
    LR_D = 5e-5
    WEIGHT_DECAY_G = 1e-4
    WEIGHT_DECAY_D = 0.0

    BETAS_G = (0.5, 0.999)
    BETAS_D = (0.5, 0.999)

    BASE_G = 16
    BASE_D = 16

    USE_SE = True
    OUT_POSITIVE = True
    USE_AMP = True

    SAVE_EVERY_N_EPOCHS = 10
    SPECIFIC_SAVE_EPOCHS = {10, 20, 30, 40, 50, 60, 80, 100}

    # adversarial schedule
    ADV_START_EPOCH = 1
    ADV_RAMP_END_EPOCH = 15
    ADV_START_VALUE = 1e-3
    ADV_END_VALUE = 1e-2

    # metrics / visuals cadence
    EXTRA_METRICS_EVERY_N_EPOCHS = 10
    VISUAL_EVERY_N_EPOCHS = 10
    VISUAL_FIXED_INDICES = (0, 5, 10)
    VISUAL_SAVE_RAW_NPZ = True

    # voxel size para peak_loc_err_mm
    DX_MM = 0.5

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

    writer = SummaryWriter(log_dir=TB_DIR)

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
    # MODELS (FROM SCRATCH)
    # =========================
    G = ResUNet3D_HQ(
        in_ch=2,
        out_ch=1,
        base=BASE_G,
        norm_kind="group",
        use_se=USE_SE,
        out_positive=OUT_POSITIVE
    ).to(device)

    D = PatchDiscriminator3D(
        in_ch=3,   # 2 canales de X + 1 canal del campo
        base=BASE_D
    ).to(device)

    # =========================
    # OPTIMS / SCALERS
    # =========================
    optim_G = optim.Adam(
        G.parameters(),
        lr=LR_G,
        betas=BETAS_G,
        weight_decay=WEIGHT_DECAY_G
    )

    optim_D = optim.Adam(
        D.parameters(),
        lr=LR_D,
        betas=BETAS_D,
        weight_decay=WEIGHT_DECAY_D
    )

    try:
        scaler_G = GradScaler("cuda", enabled=(device == "cuda" and USE_AMP))
        scaler_D = GradScaler("cuda", enabled=(device == "cuda" and USE_AMP))
    except TypeError:
        scaler_G = GradScaler(enabled=(device == "cuda" and USE_AMP))
        scaler_D = GradScaler(enabled=(device == "cuda" and USE_AMP))

    # =========================
    # LOSSES
    # =========================
    recon_criterion = StableTubeAwareTUSLoss(
        lambda_global=0.8,
        lambda_focus=1.5,
        lambda_grad=0.08,
        lambda_tube=1.0,
        lambda_profile=0.6,

        focus_frac=0.50,
        focus_min_thr=0.08,
        focus_dilate_ks=7,

        tube_radius_vox=3,
        profile_bins=16,

        global_peak_weight=4.0,
        global_peak_gamma=2.0,

        tube_warmup_start=3,
        tube_warmup_end=12,
        profile_warmup_start=5,
        profile_warmup_end=15,

        eps=1e-6,
    )

    # LSGAN
    adv_criterion = nn.MSELoss()

    # =========================
    # VISUAL CALLBACK (VAL)
    # =========================
    visual_cb = VisualCallback(
        save_dir=SAVE_DIR,
        val_dataset=val_ds,
        device=device,
        every_n_epochs=VISUAL_EVERY_N_EPOCHS,
        fixed_indices=VISUAL_FIXED_INDICES,
        use_amp=USE_AMP,
        save_raw_npz=VISUAL_SAVE_RAW_NPZ,
        writer=writer,
    )

    # =========================
    # CONFIG SAVE
    # =========================
    best_val = float("inf")
    config = {
        "SEED": SEED,
        "EPOCHS": EPOCHS,
        "LR_G": LR_G,
        "LR_D": LR_D,
        "WEIGHT_DECAY_G": WEIGHT_DECAY_G,
        "WEIGHT_DECAY_D": WEIGHT_DECAY_D,
        "BETAS_G": BETAS_G,
        "BETAS_D": BETAS_D,
        "ADV": {
            "ADV_START_EPOCH": ADV_START_EPOCH,
            "ADV_RAMP_END_EPOCH": ADV_RAMP_END_EPOCH,
            "ADV_START_VALUE": ADV_START_VALUE,
            "ADV_END_VALUE": ADV_END_VALUE,
        },
        "LOSS": {
            "lambda_global": 0.8,
            "lambda_focus": 1.5,
            "lambda_grad": 0.08,
            "lambda_tube": 1.0,
            "lambda_profile": 0.6,
            "tube_radius_vox": 3,
            "profile_bins": 16,
            "tube_warmup_start": 3,
            "tube_warmup_end": 12,
            "profile_warmup_start": 5,
            "profile_warmup_end": 15,
        },
        "VISUAL": {
            "VISUAL_EVERY_N_EPOCHS": VISUAL_EVERY_N_EPOCHS,
            "VISUAL_FIXED_INDICES": list(VISUAL_FIXED_INDICES),
            "VISUAL_SAVE_RAW_NPZ": VISUAL_SAVE_RAW_NPZ,
        },
        "EXTRA_METRICS": {
            "EXTRA_METRICS_EVERY_N_EPOCHS": EXTRA_METRICS_EVERY_N_EPOCHS,
            "DX_MM": DX_MM,
        }
    }

    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    init_csv(CSV_PATH)

    # =========================
    # TRAIN
    # =========================
    try:
        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            lambda_adv = adv_weight_schedule(
                epoch=epoch,
                start_epoch=ADV_START_EPOCH,
                ramp_end_epoch=ADV_RAMP_END_EPOCH,
                start_value=ADV_START_VALUE,
                end_value=ADV_END_VALUE
            )

            train_stats = train_one_epoch(
                G=G,
                D=D,
                loader=train_loader,
                optim_G=optim_G,
                optim_D=optim_D,
                scaler_G=scaler_G,
                scaler_D=scaler_D,
                recon_criterion=recon_criterion,
                adv_criterion=adv_criterion,
                device=device,
                epoch=epoch,
                lambda_adv=lambda_adv,
                grad_clip_G=2.0,
                grad_clip_D=1.0,
                use_amp=USE_AMP
            )

            val_stats = eval_one_epoch(
                G=G,
                D=D,
                loader=val_loader,
                recon_criterion=recon_criterion,
                adv_criterion=adv_criterion,
                device=device,
                epoch=epoch,
                lambda_adv=lambda_adv,
                use_amp=USE_AMP
            )

            extra_val_stats = None
            if epoch % EXTRA_METRICS_EVERY_N_EPOCHS == 0:
                extra_val_stats = eval_extra_metrics(
                    G=G,
                    loader=val_loader,
                    device=device,
                    dx_mm=DX_MM,
                    use_amp=USE_AMP,
                )
                log_extra_metrics_to_tensorboard(
                    writer=writer,
                    epoch=epoch,
                    stats=extra_val_stats,
                    split="val_extra"
                )

            dt = time.time() - t0

            print(
                f"Epoch {epoch:03d} | "
                f"train_G={train_stats['g_total']:.6f} | "
                f"train_recon={train_stats['g_recon']:.6f} | "
                f"train_adv={train_stats['g_adv']:.6f} | "
                f"train_D={train_stats['d_total']:.6f} | "
                f"Dreal={train_stats['d_real_mean']:.3f} | "
                f"Dfake={train_stats['d_fake_mean']:.3f} || "
                f"val_G={val_stats['g_total']:.6f} | "
                f"val_recon={val_stats['g_recon']:.6f} | "
                f"val_adv={val_stats['g_adv']:.6f} | "
                f"val_D={val_stats['d_total']:.6f} | "
                f"vDreal={val_stats['d_real_mean']:.3f} | "
                f"vDfake={val_stats['d_fake_mean']:.3f} | "
                f"glob={val_stats['loss_global']:.4f} | "
                f"focus={val_stats['loss_focus']:.4f} | "
                f"grad={val_stats['loss_grad']:.4f} | "
                f"tube={val_stats['loss_tube']:.4f} | "
                f"profile={val_stats['loss_profile']:.4f} | "
                f"lambda_adv={lambda_adv:.5f} | "
                f"lrG={optim_G.param_groups[0]['lr']:.2e} | "
                f"lrD={optim_D.param_groups[0]['lr']:.2e} | "
                f"time={dt:.1f}s"
            )

            if extra_val_stats is not None:
                print(
                    f"[VAL extra] "
                    f"peak_rel={extra_val_stats['peak_rel_err']:.4f} | "
                    f"peak_loc_mm={extra_val_stats['peak_loc_err_mm']:.3f} | "
                    f"dice50={extra_val_stats['dice50']:.4f} | "
                    f"mse_brain={extra_val_stats['mse_brain']:.4f}"
                )

            append_csv(
                CSV_PATH,
                epoch=epoch,
                train_stats=train_stats,
                val_stats=val_stats,
                extra_val_stats=extra_val_stats,
                lambda_adv=lambda_adv,
                lr_G=optim_G.param_groups[0]["lr"],
                lr_D=optim_D.param_groups[0]["lr"],
                time_sec=dt
            )

            log_epoch_to_tensorboard(
                writer=writer,
                epoch=epoch,
                train_stats=train_stats,
                val_stats=val_stats,
                lambda_adv=lambda_adv,
                lr_G=optim_G.param_groups[0]["lr"],
                lr_D=optim_D.param_groups[0]["lr"]
            )

            save_ckpt(
                os.path.join(SAVE_DIR, "last.pth"),
                G, D,
                optim_G, optim_D,
                scaler_G, scaler_D,
                epoch, best_val, config
            )

            if val_stats["g_total"] < best_val:
                best_val = val_stats["g_total"]
                save_ckpt(
                    os.path.join(SAVE_DIR, "best.pth"),
                    G, D,
                    optim_G, optim_D,
                    scaler_G, scaler_D,
                    epoch, best_val, config
                )
                print(f"New best val_G: {best_val:.6f}")

            if (epoch % SAVE_EVERY_N_EPOCHS == 0) or (epoch in SPECIFIC_SAVE_EPOCHS):
                save_ckpt(
                    os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.pth"),
                    G, D,
                    optim_G, optim_D,
                    scaler_G, scaler_D,
                    epoch, best_val, config
                )
                print(f"Saved checkpoint: epoch_{epoch:03d}.pth")

            visual_cb(G, epoch)

        print("Training done. Best val_G:", best_val)

    finally:
        writer.close()


if __name__ == "__main__":
    main()
