import os
import json
import time
import random
import csv
from contextlib import nullcontext

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataloader import TusDataset
from src.modelos.ResUnet3D import ResUNet3D_HQ
try:
    from torch.amp import GradScaler
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
# Helpers b�sicos
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


def interp_linear(alpha: float, start_value: float, end_value: float) -> float:
    alpha = float(max(0.0, min(1.0, alpha)))
    return (1.0 - alpha) * float(start_value) + alpha * float(end_value)


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
# Helpers geom�tricos para tube ROI
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

    mass = w.sum(dim=(2, 3, 4), keepdim=True)  # [B,1,1,1,1]
    has_mass = (mass > 0).view(B)

    zc = (w * z).sum(dim=(2, 3, 4), keepdim=False) / (mass.view(B, 1) + 1e-6)
    yc = (w * y).sum(dim=(2, 3, 4), keepdim=False) / (mass.view(B, 1) + 1e-6)
    xc = (w * x).sum(dim=(2, 3, 4), keepdim=False) / (mass.view(B, 1) + 1e-6)

    centers = torch.cat([zc, yc, xc], dim=1)  # [B,3]

    # fallback si la m�scara est� vac�a: usar argmax
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

    return centers  # [B,3]


def build_tube_roi_and_tcoord(source_mask: torch.Tensor,
                              target: torch.Tensor,
                              radius_vox: int = 3):
    """
    Construye un tubo entre centro de source_mask y pico GT.
    source_mask: [B,1,D,H,W]
    target:      [B,1,D,H,W]

    returns:
      tube_roi: [B,1,D,H,W]
      t_coord:  [B,1,D,H,W]  en [0,1] aprox sobre el segmento
      src_center: [B,3]
      peak_center: [B,3]
    """
    B, C, D, H, W = target.shape
    device = target.device
    dtype = torch.float32

    src_center = get_source_center_from_mask(source_mask).to(device=device, dtype=dtype)

    z_idx, y_idx, x_idx = get_gt_peak_indices(target)
    peak_center = torch.stack([
        z_idx.float(), y_idx.float(), x_idx.float()
    ], dim=1).to(device=device, dtype=dtype)  # [B,3]

    z = torch.arange(D, device=device, dtype=dtype).view(1, D, 1, 1)
    y = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
    x = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)

    zz = z.expand(B, D, H, W)
    yy = y.expand(B, D, H, W)
    xx = x.expand(B, D, H, W)

    p = torch.stack([zz, yy, xx], dim=-1)  # [B,D,H,W,3]

    a = src_center.view(B, 1, 1, 1, 3)
    b = peak_center.view(B, 1, 1, 1, 3)

    ab = b - a
    ap = p - a

    ab2 = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    t_raw = (ap * ab).sum(dim=-1, keepdim=True) / ab2
    t = t_raw.clamp(0.0, 1.0)

    closest = a + t * ab
    dist2 = ((p - closest) ** 2).sum(dim=-1, keepdim=True)

    tube_roi = (dist2 <= float(radius_vox ** 2)).float()  # [B,D,H,W,1]
    tube_roi = tube_roi.permute(0, 4, 1, 2, 3).contiguous()  # [B,1,D,H,W]
    t_coord = t.permute(0, 4, 1, 2, 3).contiguous()          # [B,1,D,H,W]

    return tube_roi, t_coord, src_center, peak_center


def compute_profile_loss(pred: torch.Tensor,
                         target: torch.Tensor,
                         tube_roi: torch.Tensor,
                         t_coord: torch.Tensor,
                         num_bins: int = 16,
                         eps: float = 1e-6):
    """
    Compara el perfil axial dentro del tubo:
    divide el segmento en bins seg�n t_coord y compara promedios.
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

        # ROI focal amplia
        focus_frac=0.50,
        focus_min_thr=0.08,
        focus_dilate_ks=7,

        # tubo
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
        r_tube = ramp_linear(epoch, self.tube_warmup_start, self.tube_warmup_end)
        r_profile = ramp_linear(epoch, self.profile_warmup_start, self.profile_warmup_end)

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

        # -----------------------------------------
        # ROI focal amplia
        # -----------------------------------------
        focus_roi, peak = make_focus_roi(
            target32,
            frac=self.focus_frac,
            min_thr=self.focus_min_thr,
            dilate_ks=self.focus_dilate_ks
        )

        # -----------------------------------------
        # 1) Global weighted L1
        # -----------------------------------------
        rel = (target32 / peak).clamp(0.0, 1.0)
        w_global = 1.0 + self.global_peak_weight * rel.pow(self.global_peak_gamma)
        loss_global = (torch.abs(pred_pos - target32) * w_global).mean()

        # -----------------------------------------
        # 2) Focus relative MSE
        # -----------------------------------------
        focus_num = masked_mean((pred_pos - target32).pow(2), focus_roi, eps=self.eps)
        focus_den = masked_mean(target32.pow(2), focus_roi, eps=self.eps).clamp_min(self.eps)
        loss_focus = focus_num / focus_den

        # -----------------------------------------
        # 3) Grad relative loss en ROI focal
        # -----------------------------------------
        pdx, pdy, pdz = grad3d(pred_pos)
        tdx, tdy, tdz = grad3d(target32)

        grad_diff = torch.abs(pdx - tdx) + torch.abs(pdy - tdy) + torch.abs(pdz - tdz)
        grad_ref = torch.abs(tdx) + torch.abs(tdy) + torch.abs(tdz)

        grad_num = masked_mean(grad_diff, focus_roi, eps=self.eps)
        grad_den = masked_mean(grad_ref, focus_roi, eps=self.eps).clamp_min(self.eps)
        loss_grad = grad_num / grad_den

        # -----------------------------------------
        # 4) Tube ROI
        # -----------------------------------------
        tube_roi, t_coord, src_center, peak_center = build_tube_roi_and_tcoord(
            source_mask=source_mask,
            target=target32,
            radius_vox=self.tube_radius_vox
        )

        if sched["lambda_tube_eff"] > 0.0:
            tube_num = masked_mean((pred_pos - target32).pow(2), tube_roi, eps=self.eps)
            tube_den = masked_mean(target32.pow(2), tube_roi, eps=self.eps).clamp_min(self.eps)
            loss_tube = tube_num / tube_den
        else:
            loss_tube = torch.zeros((), device=pred.device, dtype=torch.float32)

        # -----------------------------------------
        # 5) Axial profile loss dentro del tubo
        # -----------------------------------------
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
            loss_profile = torch.zeros((), device=pred.device, dtype=torch.float32)

        total = (
            self.lambda_global * loss_global
            + self.lambda_focus * loss_focus
            + self.lambda_grad * loss_grad
            + sched["lambda_tube_eff"] * loss_tube
            + sched["lambda_profile_eff"] * loss_profile
        )

        if return_dict:
            comp = {
                "loss_total": float(total.detach().item()),
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
    ):
        self.device = device
        self.every_n_epochs = int(every_n_epochs)
        self.use_amp = bool(use_amp)
        self.save_raw_npz = bool(save_raw_npz)

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

        print(f"=� VisualCallback activo con casos fijos: {[s[0] for s in self.samples]}")

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

    def _save_case_figure(self, pred: np.ndarray, gt: np.ndarray, epoch: int, case_idx: int):
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
            axes[r, 0].imshow(g, cmap="jet", origin="lower", vmin=0, vmax=vmax_main)
            axes[r, 0].set_title(f"{name} | GT")
            axes[r, 0].axis("off")

            axes[r, 1].imshow(p, cmap="jet", origin="lower", vmin=0, vmax=vmax_main)
            axes[r, 1].set_title(f"{name} | Pred")
            axes[r, 1].axis("off")

            axes[r, 2].imshow(e, cmap="magma", origin="lower", vmin=0, vmax=vmax_err)
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

        out_png = os.path.join(self.out_dir, f"epoch_{epoch:03d}_case_{case_idx:03d}.png")
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)

        if self.save_raw_npz:
            out_npz = os.path.join(self.out_dir, f"epoch_{epoch:03d}_case_{case_idx:03d}.npz")
            np.savez_compressed(
                out_npz,
                pred=pred.astype(np.float32),
                gt=gt.astype(np.float32),
                peak_gt=np.array([z_gt, y_gt, x_gt], dtype=np.int32),
                peak_pred=np.array([z_pr, y_pr, x_pr], dtype=np.int32),
                peak_err_vox=np.array([peak_err_vox], dtype=np.float32),
                peak_rel_err=np.array([peak_rel_err], dtype=np.float32),
            )

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

        print(f"=� Guardando visualizaciones de validaci�n en epoch {epoch:03d}...")

        for case_idx, x_cpu, y_cpu in self.samples:
            X = x_cpu.unsqueeze(0).to(self.device, non_blocking=True)

            with amp_ctx:
                pred = model(X)

            pred_np = self._to_numpy_3d(pred.clamp_min(0.0))
            gt_np = self._to_numpy_3d(y_cpu)

            self._save_case_figure(pred_np, gt_np, epoch, case_idx)

        if was_training:
            model.train()


# =========================================================
# Train / Val loops
# =========================================================
def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    epoch: int,
                    grad_clip=2.0, use_amp=True):
    model.train()

    total = 0.0
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (X, y) in enumerate(loader, start=1):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        source_mask = X[:, 0:1]  # asumimos canal 0 = source_mask

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (
            use_amp and device == "cuda") else nullcontext()

        with amp_ctx:
            pred = model(X)
            loss = criterion(pred, y, source_mask=source_mask, epoch=epoch)

        if not torch.isfinite(loss):
            print(f"� Non-finite loss en train, epoch {epoch}, step {step}. Batch saltado.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if device == "cuda" and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        total += float(loss.item())
        n_batches += 1

    return total / max(1, n_batches)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, epoch: int, use_amp=True):
    model.eval()

    total = 0.0
    n_batches = 0

    comp_sums = {
        "loss_global": 0.0,
        "loss_focus": 0.0,
        "loss_grad": 0.0,
        "loss_tube": 0.0,
        "loss_profile": 0.0,
        "r_tube": 0.0,
        "r_profile": 0.0,
        "lambda_tube_eff": 0.0,
        "lambda_profile_eff": 0.0,
    }

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        source_mask = X[:, 0:1]

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (
            use_amp and device == "cuda") else nullcontext()

        with amp_ctx:
            pred = model(X)
            loss, comp = criterion(pred, y, source_mask=source_mask, epoch=epoch, return_dict=True)

        if not torch.isfinite(loss):
            print(f"� Non-finite loss en val, epoch {epoch}. Batch saltado.")
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
                "val_loss_global",
                "val_loss_focus",
                "val_loss_grad",
                "val_loss_tube",
                "val_loss_profile",
                "r_tube",
                "r_profile",
                "lambda_tube_eff",
                "lambda_profile_eff",
                "lr",
                "time_sec",
            ])


def append_csv(csv_path, epoch, train_loss, val_loss, val_comp, lr, time_sec):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.8f}",
            f"{val_loss:.8f}",
            f"{val_comp['loss_global']:.8f}",
            f"{val_comp['loss_focus']:.8f}",
            f"{val_comp['loss_grad']:.8f}",
            f"{val_comp['loss_tube']:.8f}",
            f"{val_comp['loss_profile']:.8f}",
            f"{val_comp['r_tube']:.4f}",
            f"{val_comp['r_profile']:.4f}",
            f"{val_comp['lambda_tube_eff']:.6f}",
            f"{val_comp['lambda_profile_eff']:.6f}",
            f"{lr:.8e}",
            f"{time_sec:.2f}",
        ])


# =========================================================
# MAIN
# =========================================================
def main():
    # =========================
    # CONFIG
    # =========================
    SEED = 123

    SAVE_DIR = "checkpoints_unet_expDexpC2"
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/train"
    VAL_DIR   = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/val"
    TEST_DIR  = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

    OLD_CKPT_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_ExpC/epoch_050.pth"
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    PIN_MEMORY = True

    EPOCHS = 40
    LR = 1e-5
    WEIGHT_DECAY = 1e-4

    BASE = 16
    USE_SE = True
    OUT_POSITIVE = True

    USE_SCHEDULER = True
    PLATEAU_PATIENCE = 6
    PLATEAU_FACTOR = 0.5

    SAVE_EVERY_N_EPOCHS = 10
    SPECIFIC_SAVE_EPOCHS = {10, 20, 30, 40}

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
    train_ds = TusDataset(TRAIN_DIR)
    val_ds   = TusDataset(VAL_DIR)
    test_ds  = TusDataset(TEST_DIR)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

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

    if os.path.exists(OLD_CKPT_PATH):
        print(f"= Fine-tuning from old checkpoint: {OLD_CKPT_PATH}")
        ckpt = torch.load(OLD_CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        raise FileNotFoundError(f"No se encontr� OLD_CKPT_PATH: {OLD_CKPT_PATH}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    try:
        scaler = GradScaler("cuda", enabled=(device == "cuda"))
    except TypeError:
        scaler = GradScaler(enabled=(device == "cuda"))

    criterion = StableTubeAwareTUSLoss(
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
                verbose=True
            )
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=PLATEAU_FACTOR,
                patience=PLATEAU_PATIENCE
            )

    # =========================
    # CONFIG SAVE
    # =========================
    best_val = float("inf")
    config = {
        "SEED": SEED,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "OLD_CKPT_PATH": OLD_CKPT_PATH,
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
        }
    }

    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    init_csv(CSV_PATH)

    # =========================
    # TRAIN
    # =========================
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            epoch=epoch,
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
            f"grad={va_comp['loss_grad']:.4f} | "
            f"tube={va_comp['loss_tube']:.4f} | "
            f"profile={va_comp['loss_profile']:.4f} | "
            f"lr={new_lr:.2e} | time={dt:.1f}s"
        )

        if new_lr < prev_lr:
            print(f"=; LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        append_csv(CSV_PATH, epoch, tr, va, va_comp, new_lr, dt)

        save_ckpt(
            os.path.join(SAVE_DIR, "last.pth"),
            model, optimizer, scaler, epoch, best_val, config
        )

        if va < best_val:
            best_val = va
            save_ckpt(
                os.path.join(SAVE_DIR, "best.pth"),
                model, optimizer, scaler, epoch, best_val, config
            )
            print(f" New best val: {best_val:.6f}")

        if (epoch % SAVE_EVERY_N_EPOCHS == 0) or (epoch in SPECIFIC_SAVE_EPOCHS):
            save_ckpt(
                os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.pth"),
                model, optimizer, scaler, epoch, best_val, config
            )
            print(f"=� Saved checkpoint: epoch_{epoch:03d}.pth")

        visual_cb(model, epoch)

    print(" Training done. Best val:", best_val)


if __name__ == "__main__":
    main()