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
# Helpers num�ricos / geom�tricos
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


def make_peak_mask(target: torch.Tensor, frac: float = 0.97):
    peak = target.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    mask = (target >= peak * frac).float()
    return mask, peak


def approx_max(x: torch.Tensor, mask: torch.Tensor = None, beta: float = 20.0):
    x = x.float()
    B, C, D, H, W = x.shape
    flat = x.view(B, C, -1)

    if mask is not None:
        mask = mask.float().view(B, C, -1)
        neg_val = torch.full_like(flat, -50.0)
        flat = torch.where(mask > 0, flat, neg_val)

    m = flat.max(dim=-1, keepdim=True).values
    out = m + torch.logsumexp(beta * (flat - m), dim=-1, keepdim=True) / beta
    return out  # [B, C, 1]


def soft_argmax3d(x: torch.Tensor, mask: torch.Tensor = None, beta: float = 20.0):
    x = x.float()
    B, C, D, H, W = x.shape
    flat = x.view(B, C, -1)

    if mask is not None:
        mask = mask.float().view(B, C, -1)
        neg_val = torch.full_like(flat, -50.0)
        flat = torch.where(mask > 0, flat, neg_val)

    m = flat.max(dim=-1, keepdim=True).values
    logits = beta * (flat - m)
    prob = torch.softmax(logits, dim=-1)

    zz = torch.linspace(-1.0, 1.0, D, device=x.device, dtype=torch.float32)
    yy = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=torch.float32)
    xx = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(zz, yy, xx, indexing="ij")

    coords = torch.stack([zz, yy, xx], dim=0).view(1, 1, 3, -1)
    exp_coords = (prob.unsqueeze(2) * coords).sum(dim=-1)
    return exp_coords.squeeze(1)  # [B, 3]


def get_gt_peak_indices(target: torch.Tensor):
    """
    target: [B, 1, D, H, W]
    returns: z_idx, y_idx, x_idx each [B]
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


def gather_at_indices(vol: torch.Tensor, z_idx: torch.Tensor, y_idx: torch.Tensor, x_idx: torch.Tensor):
    """
    vol: [B, 1, D, H, W]
    returns: [B, 1]
    """
    B = vol.shape[0]
    vals = vol[
        torch.arange(B, device=vol.device),
        torch.zeros(B, dtype=torch.long, device=vol.device),
        z_idx, y_idx, x_idx
    ]
    return vals.view(B, 1)


def make_spherical_peak_roi_like(target: torch.Tensor, radius_vox: int = 2):
    """
    ROI esf�rica alrededor del pico GT.
    target: [B, 1, D, H, W]
    returns:
      roi: [B, 1, D, H, W] float
      peak_idx: tuple(z_idx, y_idx, x_idx)
    """
    B, C, D, H, W = target.shape
    device = target.device

    z_idx, y_idx, x_idx = get_gt_peak_indices(target)

    z = torch.arange(D, device=device).view(1, D, 1, 1)
    y = torch.arange(H, device=device).view(1, 1, H, 1)
    x = torch.arange(W, device=device).view(1, 1, 1, W)

    zz = z.expand(B, D, H, W)
    yy = y.expand(B, D, H, W)
    xx = x.expand(B, D, H, W)

    zc = z_idx.view(B, 1, 1, 1)
    yc = y_idx.view(B, 1, 1, 1)
    xc = x_idx.view(B, 1, 1, 1)

    dist2 = (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2
    roi = (dist2 <= (radius_vox ** 2)).float().unsqueeze(1)  # [B,1,D,H,W]

    return roi, (z_idx, y_idx, x_idx)


# =========================================================
# Loss ExpC PeakStrong
# =========================================================
class StableFocusAwareTUSLoss_ExpC_peakStrong(nn.Module):
    def __init__(
        self,
        lambda_global=0.8,
        lambda_focus=2.0,
        lambda_peak=1.0,
        lambda_loc=0.7,
        lambda_grad=0.08,
        lambda_peak_gt=0.35,
        lambda_peak_roi=0.45,

        # ROI amplia para forma focal
        wide_frac=0.50,
        wide_min_thr=0.08,
        wide_dilate_ks=7,

        # ROI media para gradientes
        mid_frac=0.60,
        mid_min_thr=0.08,
        mid_dilate_ks=5,

        # ROI muy estrecha para localizaci�n
        tight_frac=0.80,
        tight_min_thr=0.10,
        tight_dilate_ks=1,

        # m�scara muy estricta para approx max
        peak_frac=0.95,

        # ROI esf�rica fija alrededor del pico GT
        peak_roi_radius_vox=2,

        global_peak_weight=4.0,
        global_peak_gamma=2.0,

        peak_warmup_start=8,
        peak_warmup_end=18,
        loc_warmup_start=10,
        loc_warmup_end=22,
        peak_local_warmup_start=12,
        peak_local_warmup_end=24,

        beta_peak_start=12.0,
        beta_peak_end=50.0,
        beta_loc_start=12.0,
        beta_loc_end=80.0,

        eps=1e-6,
    ):
        super().__init__()

        self.lambda_global = float(lambda_global)
        self.lambda_focus = float(lambda_focus)
        self.lambda_peak = float(lambda_peak)
        self.lambda_loc = float(lambda_loc)
        self.lambda_grad = float(lambda_grad)
        self.lambda_peak_gt = float(lambda_peak_gt)
        self.lambda_peak_roi = float(lambda_peak_roi)

        self.wide_frac = float(wide_frac)
        self.wide_min_thr = float(wide_min_thr)
        self.wide_dilate_ks = int(wide_dilate_ks)

        self.mid_frac = float(mid_frac)
        self.mid_min_thr = float(mid_min_thr)
        self.mid_dilate_ks = int(mid_dilate_ks)

        self.tight_frac = float(tight_frac)
        self.tight_min_thr = float(tight_min_thr)
        self.tight_dilate_ks = int(tight_dilate_ks)

        self.peak_frac = float(peak_frac)
        self.peak_roi_radius_vox = int(peak_roi_radius_vox)

        self.global_peak_weight = float(global_peak_weight)
        self.global_peak_gamma = float(global_peak_gamma)

        self.peak_warmup_start = int(peak_warmup_start)
        self.peak_warmup_end = int(peak_warmup_end)
        self.loc_warmup_start = int(loc_warmup_start)
        self.loc_warmup_end = int(loc_warmup_end)
        self.peak_local_warmup_start = int(peak_local_warmup_start)
        self.peak_local_warmup_end = int(peak_local_warmup_end)

        self.beta_peak_start = float(beta_peak_start)
        self.beta_peak_end = float(beta_peak_end)
        self.beta_loc_start = float(beta_loc_start)
        self.beta_loc_end = float(beta_loc_end)

        self.eps = float(eps)

    def get_schedule(self, epoch: int):
        r_peak = ramp_linear(epoch, self.peak_warmup_start, self.peak_warmup_end)
        r_loc = ramp_linear(epoch, self.loc_warmup_start, self.loc_warmup_end)
        r_peak_local = ramp_linear(epoch, self.peak_local_warmup_start, self.peak_local_warmup_end)

        beta_peak = interp_linear(r_peak, self.beta_peak_start, self.beta_peak_end)
        beta_loc = interp_linear(r_loc, self.beta_loc_start, self.beta_loc_end)

        return {
            "r_peak": r_peak,
            "r_loc": r_loc,
            "r_peak_local": r_peak_local,
            "beta_peak": beta_peak,
            "beta_loc": beta_loc,
            "lambda_peak_eff": self.lambda_peak * r_peak,
            "lambda_loc_eff": self.lambda_loc * r_loc,
            "lambda_peak_gt_eff": self.lambda_peak_gt * r_peak_local,
            "lambda_peak_roi_eff": self.lambda_peak_roi * r_peak_local,
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor, epoch: int = 1, return_dict: bool = False):
        pred32 = pred.float()
        target32 = target.float()
        pred_pos = pred32.clamp_min(0.0)

        sched = self.get_schedule(epoch)

        # ROIs por threshold
        focus_roi_wide, peak = make_focus_roi(
            target32,
            frac=self.wide_frac,
            min_thr=self.wide_min_thr,
            dilate_ks=self.wide_dilate_ks
        )

        focus_roi_mid, _ = make_focus_roi(
            target32,
            frac=self.mid_frac,
            min_thr=self.mid_min_thr,
            dilate_ks=self.mid_dilate_ks
        )

        focus_roi_tight, _ = make_focus_roi(
            target32,
            frac=self.tight_frac,
            min_thr=self.tight_min_thr,
            dilate_ks=self.tight_dilate_ks
        )

        peak_mask, _ = make_peak_mask(target32, frac=self.peak_frac)

        # ROI esf�rica fija alrededor del pico GT
        peak_roi_sphere, (z_idx, y_idx, x_idx) = make_spherical_peak_roi_like(
            target32,
            radius_vox=self.peak_roi_radius_vox
        )

        # -------------------------------------------------
        # 1) Global weighted L1
        # -------------------------------------------------
        rel = (target32 / peak).clamp(0.0, 1.0)
        w_global = 1.0 + self.global_peak_weight * rel.pow(self.global_peak_gamma)
        loss_global = (torch.abs(pred_pos - target32) * w_global).mean()

        # -------------------------------------------------
        # 2) Focus ROI relative MSE (ROI amplia)
        # -------------------------------------------------
        focus_num = masked_mean((pred_pos - target32).pow(2), focus_roi_wide, eps=self.eps)
        focus_den = masked_mean(target32.pow(2), focus_roi_wide, eps=self.eps).clamp_min(self.eps)
        loss_focus = focus_num / focus_den

        # -------------------------------------------------
        # 3) Focus grad relative loss (ROI media)
        # -------------------------------------------------
        pdx, pdy, pdz = grad3d(pred_pos)
        tdx, tdy, tdz = grad3d(target32)

        grad_diff = torch.abs(pdx - tdx) + torch.abs(pdy - tdy) + torch.abs(pdz - tdz)
        grad_ref = torch.abs(tdx) + torch.abs(tdy) + torch.abs(tdz)

        grad_num = masked_mean(grad_diff, focus_roi_mid, eps=self.eps)
        grad_den = masked_mean(grad_ref, focus_roi_mid, eps=self.eps).clamp_min(self.eps)
        loss_grad = grad_num / grad_den

        # -------------------------------------------------
        # 4) Peak relative loss via approx max (muy estricta)
        # -------------------------------------------------
        if sched["lambda_peak_eff"] > 0.0:
            pred_peak = approx_max(pred_pos, mask=peak_mask, beta=sched["beta_peak"]).view(-1, 1)
            targ_peak = approx_max(target32, mask=peak_mask, beta=sched["beta_peak"]).view(-1, 1)
            loss_peak = (torch.abs(pred_peak - targ_peak) / (torch.abs(targ_peak) + self.eps)).mean()
        else:
            loss_peak = torch.zeros((), device=pred.device, dtype=torch.float32)

        # -------------------------------------------------
        # 5) Location loss (ROI muy estrecha)
        # -------------------------------------------------
        if sched["lambda_loc_eff"] > 0.0:
            pred_loc = soft_argmax3d(pred_pos, mask=focus_roi_tight, beta=sched["beta_loc"])
            targ_loc = soft_argmax3d(target32, mask=focus_roi_tight, beta=sched["beta_loc"])
            loss_loc = F.l1_loss(pred_loc, targ_loc)
        else:
            loss_loc = torch.zeros((), device=pred.device, dtype=torch.float32)

        # -------------------------------------------------
        # 6) Peak value at GT-peak voxel
        #     penaliza directamente el valor en la ubicaci�n GT
        # -------------------------------------------------
        if sched["lambda_peak_gt_eff"] > 0.0:
            pred_peak_at_gt = gather_at_indices(pred_pos, z_idx, y_idx, x_idx)
            targ_peak_at_gt = gather_at_indices(target32, z_idx, y_idx, x_idx)
            loss_peak_gt = (torch.abs(pred_peak_at_gt - targ_peak_at_gt) / (torch.abs(targ_peak_at_gt) + self.eps)).mean()
        else:
            loss_peak_gt = torch.zeros((), device=pred.device, dtype=torch.float32)

        # -------------------------------------------------
        # 7) Local shape around GT peak (ROI sphere)
        # -------------------------------------------------
        if sched["lambda_peak_roi_eff"] > 0.0:
            peak_roi_num = masked_mean((pred_pos - target32).pow(2), peak_roi_sphere, eps=self.eps)
            peak_roi_den = masked_mean(target32.pow(2), peak_roi_sphere, eps=self.eps).clamp_min(self.eps)
            loss_peak_roi = peak_roi_num / peak_roi_den
        else:
            loss_peak_roi = torch.zeros((), device=pred.device, dtype=torch.float32)

        total = (
            self.lambda_global * loss_global
            + self.lambda_focus * loss_focus
            + self.lambda_grad * loss_grad
            + sched["lambda_peak_eff"] * loss_peak
            + sched["lambda_loc_eff"] * loss_loc
            + sched["lambda_peak_gt_eff"] * loss_peak_gt
            + sched["lambda_peak_roi_eff"] * loss_peak_roi
        )

        if return_dict:
            comp = {
                "loss_total": float(total.detach().item()),
                "loss_global": float(loss_global.detach().item()),
                "loss_focus": float(loss_focus.detach().item()),
                "loss_grad": float(loss_grad.detach().item()),
                "loss_peak": float(loss_peak.detach().item()),
                "loss_loc": float(loss_loc.detach().item()),
                "loss_peak_gt": float(loss_peak_gt.detach().item()),
                "loss_peak_roi": float(loss_peak_roi.detach().item()),
                "r_peak": float(sched["r_peak"]),
                "r_loc": float(sched["r_loc"]),
                "r_peak_local": float(sched["r_peak_local"]),
                "beta_peak": float(sched["beta_peak"]),
                "beta_loc": float(sched["beta_loc"]),
                "lambda_peak_eff": float(sched["lambda_peak_eff"]),
                "lambda_loc_eff": float(sched["lambda_loc_eff"]),
                "lambda_peak_gt_eff": float(sched["lambda_peak_gt_eff"]),
                "lambda_peak_roi_eff": float(sched["lambda_peak_roi_eff"]),
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

        print(f"=� VisualCallback activo con casos fijos: {[s[0] for s in self.samples]}")

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

        print(f"=� Guardando visualizaciones de validaci�n en epoch {epoch:03d}...")

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

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (
            use_amp and device == "cuda") else nullcontext()

        with amp_ctx:
            pred = model(X)
            loss = criterion(pred, y, epoch=epoch)

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
        "loss_peak": 0.0,
        "loss_loc": 0.0,
        "loss_peak_gt": 0.0,
        "loss_peak_roi": 0.0,
        "r_peak": 0.0,
        "r_loc": 0.0,
        "r_peak_local": 0.0,
        "beta_peak": 0.0,
        "beta_loc": 0.0,
        "lambda_peak_eff": 0.0,
        "lambda_loc_eff": 0.0,
        "lambda_peak_gt_eff": 0.0,
        "lambda_peak_roi_eff": 0.0,
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


def load_ckpt(path, model, optimizer=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    config = ckpt.get("config", None)
    return epoch, best_val, config


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
                "val_loss_peak",
                "val_loss_loc",
                "val_loss_peak_gt",
                "val_loss_peak_roi",
                "r_peak",
                "r_loc",
                "r_peak_local",
                "beta_peak",
                "beta_loc",
                "lambda_peak_eff",
                "lambda_loc_eff",
                "lambda_peak_gt_eff",
                "lambda_peak_roi_eff",
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
            f"{val_comp['loss_peak']:.8f}",
            f"{val_comp['loss_loc']:.8f}",
            f"{val_comp['loss_peak_gt']:.8f}",
            f"{val_comp['loss_peak_roi']:.8f}",
            f"{val_comp['r_peak']:.4f}",
            f"{val_comp['r_loc']:.4f}",
            f"{val_comp['r_peak_local']:.4f}",
            f"{val_comp['beta_peak']:.4f}",
            f"{val_comp['beta_loc']:.4f}",
            f"{val_comp['lambda_peak_eff']:.6f}",
            f"{val_comp['lambda_loc_eff']:.6f}",
            f"{val_comp['lambda_peak_gt_eff']:.6f}",
            f"{val_comp['lambda_peak_roi_eff']:.6f}",
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

    SAVE_DIR = "checkpoints_unet_ExpC_V2"
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/train"
    VAL_DIR   = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/val"
    TEST_DIR  = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

    # checkpoint del modelo antiguo global-best
    OLD_CKPT_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_V2/epoch_100.pth"

    BATCH_SIZE = 1
    NUM_WORKERS = 2
    PIN_MEMORY = True

    EPOCHS = 60
    LR = 2e-5
    WEIGHT_DECAY = 1e-4

    BASE = 16
    USE_SE = True
    OUT_POSITIVE = True

    USE_SCHEDULER = True
    PLATEAU_PATIENCE = 6
    PLATEAU_FACTOR = 0.5

    SAVE_EVERY_N_EPOCHS = 10
    SPECIFIC_SAVE_EPOCHS = {20, 30, 40, 50, 60}

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

    # Fine-tuning desde el modelo antiguo
    if os.path.exists(OLD_CKPT_PATH):
        print(f"= Fine-tuning from old checkpoint: {OLD_CKPT_PATH}")
        ckpt = torch.load(OLD_CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        raise FileNotFoundError(f"No se encontr� OLD_CKPT_PATH: {OLD_CKPT_PATH}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    try:
        scaler = GradScaler("cuda", enabled=(device == "cuda"))
    except TypeError:
        scaler = GradScaler(enabled=(device == "cuda"))

    criterion = StableFocusAwareTUSLoss_ExpC_peakStrong(
        lambda_global=0.8,
        lambda_focus=2.0,
        lambda_peak=1.2,
        lambda_loc=0.9,
        lambda_grad=0.08,
        lambda_peak_gt=0.6,
        lambda_peak_roi=0.8,

        wide_frac=0.50,
        wide_min_thr=0.08,
        wide_dilate_ks=7,

        mid_frac=0.60,
        mid_min_thr=0.08,
        mid_dilate_ks=5,

        tight_frac=0.85,
        tight_min_thr=0.10,
        tight_dilate_ks=1,

        peak_frac=0.97,
        peak_roi_radius_vox=2,

        global_peak_weight=4.0,
        global_peak_gamma=2.0,

        peak_warmup_start=4,
        peak_warmup_end=12,
        loc_warmup_start=4,
        loc_warmup_end=14,
        peak_local_warmup_start=4,
        peak_local_warmup_end=12,

        beta_peak_start=12.0,
        beta_peak_end=50.0,
        beta_loc_start=12.0,
        beta_loc_end=80.0,

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
            "lambda_focus": 2.0,
            "lambda_peak": 1.2,
            "lambda_loc": 0.9,
            "lambda_grad": 0.08,
            "lambda_peak_gt": 0.6,
            "lambda_peak_roi": 0.8,
            "wide_frac": 0.50,
            "mid_frac": 0.60,
            "tight_frac": 0.85,
            "peak_frac": 0.97,
            "peak_roi_radius_vox": 2,
            "beta_peak_end": 50.0,
            "beta_loc_end": 80.0,
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
            f"peak={va_comp['loss_peak']:.4f} | "
            f"loc={va_comp['loss_loc']:.4f} | "
            f"peakGT={va_comp['loss_peak_gt']:.4f} | "
            f"peakROI={va_comp['loss_peak_roi']:.4f} | "
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

    print("Training done. Best val:", best_val)


if __name__ == "__main__":
    main()