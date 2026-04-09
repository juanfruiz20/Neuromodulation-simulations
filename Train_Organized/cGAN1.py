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

from Data_loaderV2 import TusDataset
from Unet3D_V2 import ResUNet3D_HQ

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
# Helpers
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
        roi = F.max_pool3d(roi, kernel_size=dilate_ks,
                           stride=1, padding=dilate_ks // 2)
        roi = (roi > 0).float()

    return roi, peak


def make_peak_mask(target: torch.Tensor, frac: float = 0.92):
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
    return out


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
    return exp_coords.squeeze(1)


# =========================================================
# Loss física base (ExpB)
# =========================================================
class StableFocusAwareTUSLoss_ExpB(nn.Module):
    def __init__(
        self,
        lambda_global=0.8,
        lambda_focus=2.0,
        lambda_peak=0.8,
        lambda_loc=0.6,
        lambda_grad=0.08,

        wide_frac=0.50,
        wide_min_thr=0.08,
        wide_dilate_ks=7,

        mid_frac=0.60,
        mid_min_thr=0.08,
        mid_dilate_ks=5,

        tight_frac=0.75,
        tight_min_thr=0.10,
        tight_dilate_ks=3,

        peak_frac=0.92,

        global_peak_weight=4.0,
        global_peak_gamma=2.0,

        peak_warmup_start=5,
        peak_warmup_end=18,
        loc_warmup_start=8,
        loc_warmup_end=20,

        beta_peak_start=10.0,
        beta_peak_end=30.0,
        beta_loc_start=10.0,
        beta_loc_end=50.0,

        eps=1e-6,
    ):
        super().__init__()

        self.lambda_global = float(lambda_global)
        self.lambda_focus = float(lambda_focus)
        self.lambda_peak = float(lambda_peak)
        self.lambda_loc = float(lambda_loc)
        self.lambda_grad = float(lambda_grad)

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

        self.global_peak_weight = float(global_peak_weight)
        self.global_peak_gamma = float(global_peak_gamma)

        self.peak_warmup_start = int(peak_warmup_start)
        self.peak_warmup_end = int(peak_warmup_end)
        self.loc_warmup_start = int(loc_warmup_start)
        self.loc_warmup_end = int(loc_warmup_end)

        self.beta_peak_start = float(beta_peak_start)
        self.beta_peak_end = float(beta_peak_end)
        self.beta_loc_start = float(beta_loc_start)
        self.beta_loc_end = float(beta_loc_end)

        self.eps = float(eps)

    def get_schedule(self, epoch: int):
        r_peak = ramp_linear(epoch, self.peak_warmup_start,
                             self.peak_warmup_end)
        r_loc = ramp_linear(epoch, self.loc_warmup_start, self.loc_warmup_end)

        beta_peak = interp_linear(
            r_peak, self.beta_peak_start, self.beta_peak_end)
        beta_loc = interp_linear(r_loc, self.beta_loc_start, self.beta_loc_end)

        return {
            "r_peak": r_peak,
            "r_loc": r_loc,
            "beta_peak": beta_peak,
            "beta_loc": beta_loc,
            "lambda_peak_eff": self.lambda_peak * r_peak,
            "lambda_loc_eff": self.lambda_loc * r_loc,
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor, epoch: int = 1, return_dict: bool = False):
        pred32 = pred.float()
        target32 = target.float()
        pred_pos = pred32.clamp_min(0.0)

        sched = self.get_schedule(epoch)

        focus_roi_wide, peak = make_focus_roi(
            target32, frac=self.wide_frac, min_thr=self.wide_min_thr, dilate_ks=self.wide_dilate_ks
        )
        focus_roi_mid, _ = make_focus_roi(
            target32, frac=self.mid_frac, min_thr=self.mid_min_thr, dilate_ks=self.mid_dilate_ks
        )
        focus_roi_tight, _ = make_focus_roi(
            target32, frac=self.tight_frac, min_thr=self.tight_min_thr, dilate_ks=self.tight_dilate_ks
        )
        peak_mask, _ = make_peak_mask(target32, frac=self.peak_frac)

        rel = (target32 / peak).clamp(0.0, 1.0)
        w_global = 1.0 + self.global_peak_weight * \
            rel.pow(self.global_peak_gamma)
        loss_global = (torch.abs(pred_pos - target32) * w_global).mean()

        focus_num = masked_mean(
            (pred_pos - target32).pow(2), focus_roi_wide, eps=self.eps)
        focus_den = masked_mean(target32.pow(
            2), focus_roi_wide, eps=self.eps).clamp_min(self.eps)
        loss_focus = focus_num / focus_den

        pdx, pdy, pdz = grad3d(pred_pos)
        tdx, tdy, tdz = grad3d(target32)

        grad_diff = torch.abs(pdx - tdx) + \
            torch.abs(pdy - tdy) + torch.abs(pdz - tdz)
        grad_ref = torch.abs(tdx) + torch.abs(tdy) + torch.abs(tdz)

        grad_num = masked_mean(grad_diff, focus_roi_mid, eps=self.eps)
        grad_den = masked_mean(grad_ref, focus_roi_mid,
                               eps=self.eps).clamp_min(self.eps)
        loss_grad = grad_num / grad_den

        if sched["lambda_peak_eff"] > 0.0:
            pred_peak = approx_max(
                pred_pos, mask=peak_mask, beta=sched["beta_peak"]).view(-1, 1)
            targ_peak = approx_max(
                target32, mask=peak_mask, beta=sched["beta_peak"]).view(-1, 1)
            loss_peak = (torch.abs(pred_peak - targ_peak) /
                         (torch.abs(targ_peak) + self.eps)).mean()
        else:
            loss_peak = torch.zeros(
                (), device=pred.device, dtype=torch.float32)

        if sched["lambda_loc_eff"] > 0.0:
            pred_loc = soft_argmax3d(
                pred_pos, mask=focus_roi_tight, beta=sched["beta_loc"])
            targ_loc = soft_argmax3d(
                target32, mask=focus_roi_tight, beta=sched["beta_loc"])
            loss_loc = F.l1_loss(pred_loc, targ_loc)
        else:
            loss_loc = torch.zeros((), device=pred.device, dtype=torch.float32)

        total = (
            self.lambda_global * loss_global
            + self.lambda_focus * loss_focus
            + self.lambda_grad * loss_grad
            + sched["lambda_peak_eff"] * loss_peak
            + sched["lambda_loc_eff"] * loss_loc
        )

        if return_dict:
            comp = {
                "loss_total": float(total.detach().item()),
                "loss_global": float(loss_global.detach().item()),
                "loss_focus": float(loss_focus.detach().item()),
                "loss_grad": float(loss_grad.detach().item()),
                "loss_peak": float(loss_peak.detach().item()),
                "loss_loc": float(loss_loc.detach().item()),
                "r_peak": float(sched["r_peak"]),
                "r_loc": float(sched["r_loc"]),
                "beta_peak": float(sched["beta_peak"]),
                "beta_loc": float(sched["beta_loc"]),
                "lambda_peak_eff": float(sched["lambda_peak_eff"]),
                "lambda_loc_eff": float(sched["lambda_loc_eff"]),
            }
            return total, comp

        return total


# =========================================================
# Discriminador 3D PatchGAN
# =========================================================
class PatchDiscriminator3D(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()

        def block(cin, cout, k=4, s=2, p=1, norm=True):
            layers = [nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p)]
            if norm:
                layers.append(nn.InstanceNorm3d(cout, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_ch, base, norm=False),
            block(base, base * 2),
            block(base * 2, base * 4),
            block(base * 4, base * 8, s=1),
            nn.Conv3d(base * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)


def d_loss_lsgan(d_real, d_fake):
    loss_real = torch.mean((d_real - 1.0) ** 2)
    loss_fake = torch.mean((d_fake - 0.0) ** 2)
    return 0.5 * (loss_real + loss_fake)


def g_loss_lsgan(d_fake):
    return torch.mean((d_fake - 1.0) ** 2)


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

        print(
            f"🖼️ VisualCallback activo con casos fijos: {[s[0] for s in self.samples]}")

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

        peak_err_vox = np.sqrt((z_pr - z_gt) ** 2 +
                               (y_pr - y_gt) ** 2 + (x_pr - x_gt) ** 2)

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
        rows = [("Axial", gt_ax, pr_ax, er_ax), ("Coronal", gt_co,
                                                 pr_co, er_co), ("Sagittal", gt_sa, pr_sa, er_sa)]

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
            f"GT peak idx=({z_gt},{y_gt},{x_gt}) | Pred peak idx=({z_pr},{y_pr},{x_pr}) | "
            f"PeakErr={peak_err_vox:.2f} vox | PeakRelErr={peak_rel_err:.4f}",
            fontsize=12
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_png = os.path.join(
            self.out_dir, f"epoch_{epoch:03d}_case_{case_idx:03d}.png")
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)

        if self.save_raw_npz:
            out_npz = os.path.join(
                self.out_dir, f"epoch_{epoch:03d}_case_{case_idx:03d}.npz")
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
    def __call__(self, G, epoch: int):
        if not self.should_run(epoch):
            return

        was_training = G.training
        G.eval()

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (self.use_amp and self.device == "cuda")
            else nullcontext()
        )

        print(
            f"🖼️ Guardando visualizaciones de validación en epoch {epoch:03d}...")

        for case_idx, x_cpu, y_cpu in self.samples:
            X = x_cpu.unsqueeze(0).to(self.device, non_blocking=True)
            with amp_ctx:
                pred = G(X)

            pred_np = self._to_numpy_3d(pred.clamp_min(0.0))
            gt_np = self._to_numpy_3d(y_cpu)

            self._save_case_figure(pred_np, gt_np, epoch, case_idx)

        if was_training:
            G.train()


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
                "train_g_total",
                "train_g_phys",
                "train_g_adv",
                "train_d",
                "val_g_phys",
                "val_loss_global",
                "val_loss_focus",
                "val_loss_grad",
                "val_loss_peak",
                "val_loss_loc",
                "lambda_adv",
                "lr_g",
                "lr_d",
                "time_sec",
            ])


def append_csv(csv_path, epoch, tr_stats, val_phys, val_comp, lambda_adv, lr_g, lr_d, time_sec):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{tr_stats['g_total']:.8f}",
            f"{tr_stats['g_phys']:.8f}",
            f"{tr_stats['g_adv']:.8f}",
            f"{tr_stats['d_total']:.8f}",
            f"{val_phys:.8f}",
            f"{val_comp['loss_global']:.8f}",
            f"{val_comp['loss_focus']:.8f}",
            f"{val_comp['loss_grad']:.8f}",
            f"{val_comp['loss_peak']:.8f}",
            f"{val_comp['loss_loc']:.8f}",
            f"{lambda_adv:.6f}",
            f"{lr_g:.8e}",
            f"{lr_d:.8e}",
            f"{time_sec:.2f}",
        ])


# =========================================================
# Schedules GAN
# =========================================================
def get_lambda_adv(epoch, adv_start_epoch=6, adv_full_epoch=20, adv_max=0.01):
    alpha = ramp_linear(epoch, adv_start_epoch, adv_full_epoch)
    return interp_linear(alpha, 0.0, adv_max)


# =========================================================
# Train / Val loops
# =========================================================
def train_one_epoch_cgan(
    G, D, loader,
    opt_G, opt_D,
    scaler_G, scaler_D,
    criterion_phys,
    device,
    epoch,
    lambda_adv=0.01,
    grad_clip_G=2.0,
    grad_clip_D=2.0,
    use_amp=True,
    d_update_every=2,
):
    G.train()
    D.train()

    sums = {"g_total": 0.0, "g_phys": 0.0, "g_adv": 0.0, "d_total": 0.0}
    n_batches = 0
    n_d_updates = 0

    for batch_idx, (X, y) in enumerate(loader, start=1):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (use_amp and device == "cuda")
            else nullcontext()
        )

        # =====================================================
        # 1) Update D only if:
        #    - adversarial branch is active
        #    - this batch is scheduled for D update
        # =====================================================
        do_d_update = (lambda_adv > 0.0) and ((batch_idx % d_update_every) == 0)

        if do_d_update:
            opt_D.zero_grad(set_to_none=True)

            with amp_ctx:
                y_hat_detached = G(X).detach()
                d_real = D(X, y)
                d_fake = D(X, y_hat_detached)
                loss_D = d_loss_lsgan(d_real, d_fake)

            if torch.isfinite(loss_D):
                if device == "cuda" and use_amp:
                    scaler_D.scale(loss_D).backward()
                    scaler_D.unscale_(opt_D)
                    if grad_clip_D is not None and grad_clip_D > 0:
                        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=float(grad_clip_D))
                    scaler_D.step(opt_D)
                    scaler_D.update()
                else:
                    loss_D.backward()
                    if grad_clip_D is not None and grad_clip_D > 0:
                        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=float(grad_clip_D))
                    opt_D.step()

                d_loss_value = float(loss_D.item())
                n_d_updates += 1

            else:
                print(f"⚠️ Non-finite D loss en epoch {epoch}, batch {batch_idx}. Batch saltado.")
                opt_D.zero_grad(set_to_none=True)
                continue
        else:
            d_loss_value = 0.0

        # =====================================================
        # 2) Update G every batch
        # =====================================================
        opt_G.zero_grad(set_to_none=True)

        with amp_ctx:
            y_hat = G(X)
            loss_phys = criterion_phys(y_hat, y, epoch=epoch)

            if lambda_adv > 0.0:
                d_fake_for_g = D(X, y_hat)
                loss_adv = g_loss_lsgan(d_fake_for_g)
            else:
                loss_adv = torch.zeros((), device=device, dtype=torch.float32)

            loss_G = loss_phys + lambda_adv * loss_adv

        if torch.isfinite(loss_G):
            if device == "cuda" and use_amp:
                scaler_G.scale(loss_G).backward()
                scaler_G.unscale_(opt_G)
                if grad_clip_G is not None and grad_clip_G > 0:
                    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=float(grad_clip_G))
                scaler_G.step(opt_G)
                scaler_G.update()
            else:
                loss_G.backward()
                if grad_clip_G is not None and grad_clip_G > 0:
                    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=float(grad_clip_G))
                opt_G.step()
        else:
            print(f"⚠️ Non-finite G loss en epoch {epoch}, batch {batch_idx}. Batch saltado.")
            opt_G.zero_grad(set_to_none=True)
            continue

        sums["g_total"] += float(loss_G.item())
        sums["g_phys"] += float(loss_phys.item())
        sums["g_adv"] += float(loss_adv.item())
        sums["d_total"] += d_loss_value
        n_batches += 1

    if n_batches == 0:
        return {k: float("inf") for k in sums.keys()}

    out = {
        "g_total": sums["g_total"] / max(1, n_batches),
        "g_phys": sums["g_phys"] / max(1, n_batches),
        "g_adv": sums["g_adv"] / max(1, n_batches),
        # D se promedia solo sobre las veces que realmente se actualizó
        "d_total": (sums["d_total"] / max(1, n_d_updates)) if n_d_updates > 0 else 0.0,
    }
    return out

@torch.no_grad()
def eval_generator(G, loader, criterion_phys, device, epoch, use_amp=True):
    G.eval()

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
            y_hat = G(X)
            loss, comp = criterion_phys(
                y_hat, y, epoch=epoch, return_dict=True)

        if not torch.isfinite(loss):
            print(f"⚠️ Non-finite val G loss en epoch {epoch}. Batch saltado.")
            continue

        total += float(loss.item())
        n_batches += 1
        for k in comp_sums.keys():
            comp_sums[k] += float(comp[k])

    if n_batches == 0:
        return float("inf"), {k: float("inf") for k in comp_sums.keys()}

    avg_total = total / n_batches
    avg_comp = {k: v / n_batches for k, v in comp_sums.items()}
    return avg_total, avg_comp


# =========================================================
# MAIN
# =========================================================
def main():
    SEED = 123
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # -------------------------
    # CONFIG
    # -------------------------
    SAVE_DIR = "checkpoints_cgan_tus_v1"
    os.makedirs(SAVE_DIR, exist_ok=True)

    TRAIN_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/train"
    VAL_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/val"
    TEST_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

    BATCH_SIZE = 1
    NUM_WORKERS = 2
    PIN_MEMORY = True

    EPOCHS = 100

    LR_G = 1e-4
    LR_D = 1e-5
    WEIGHT_DECAY_G = 1e-4
    WEIGHT_DECAY_D = 0.0
    D_UPDATE_EVERY = 2
    BASE_G = 16
    USE_SE = True
    OUT_POSITIVE = True
    BASE_D = 32

    USE_SCHEDULER_G = True
    PLATEAU_PATIENCE_G = 6
    PLATEAU_FACTOR_G = 0.5

    WARM_START_G = True
    WARM_START_G_PATH = r"checkpoints_unet_V2/last.pth"

    RESUME_CGAN = False
    RESUME_G_PATH = os.path.join(SAVE_DIR, "last_G.pth")
    RESUME_D_PATH = os.path.join(SAVE_DIR, "last_D.pth")

    SAVE_EVERY_N_EPOCHS = 10
    SPECIFIC_SAVE_EPOCHS = {70, 80, 90, 100}
    CSV_PATH = os.path.join(SAVE_DIR, "training_log.csv")

    VISUAL_EVERY_N_EPOCHS = 10
    VISUAL_FIXED_INDICES = (0, 5, 10)
    VISUAL_SAVE_RAW_NPZ = True

    ADV_START_EPOCH = 10
    ADV_FULL_EPOCH = 30
    ADV_MAX = 0.001

    # -------------------------
    # DATA
    # -------------------------
    train_ds = TusDataset(TRAIN_DIR)
    val_ds = TusDataset(VAL_DIR)
    test_ds = TusDataset(TEST_DIR)

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

    # -------------------------
    # MODELS
    # -------------------------
    G = ResUNet3D_HQ(
        in_ch=2,
        out_ch=1,
        base=BASE_G,
        norm_kind="group",
        use_se=USE_SE,
        out_positive=OUT_POSITIVE
    ).to(device)

    D = PatchDiscriminator3D(in_ch=3, base=BASE_D).to(device)

    opt_G = optim.AdamW(G.parameters(), lr=LR_G,
                        weight_decay=WEIGHT_DECAY_G, betas=(0.5, 0.999))
    opt_D = optim.AdamW(D.parameters(), lr=LR_D,
                        weight_decay=WEIGHT_DECAY_D, betas=(0.5, 0.999))

    try:
        scaler_G = GradScaler("cuda", enabled=(device == "cuda"))
        scaler_D = GradScaler("cuda", enabled=(device == "cuda"))
    except TypeError:
        scaler_G = GradScaler(enabled=(device == "cuda"))
        scaler_D = GradScaler(enabled=(device == "cuda"))

    criterion_phys = StableFocusAwareTUSLoss_ExpB()

    visual_cb = VisualCallback(
        save_dir=SAVE_DIR,
        val_dataset=val_ds,
        device=device,
        every_n_epochs=VISUAL_EVERY_N_EPOCHS,
        fixed_indices=VISUAL_FIXED_INDICES,
        use_amp=True,
        save_raw_npz=VISUAL_SAVE_RAW_NPZ,
    )

    scheduler_G = None
    if USE_SCHEDULER_G:
        try:
            scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
                opt_G, mode="min", factor=PLATEAU_FACTOR_G, patience=PLATEAU_PATIENCE_G, verbose=True
            )
        except TypeError:
            scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
                opt_G, mode="min", factor=PLATEAU_FACTOR_G, patience=PLATEAU_PATIENCE_G
            )

    # -------------------------
    # Warm start / resume
    # -------------------------
    start_epoch = 0
    best_val_g = float("inf")

    if WARM_START_G and os.path.exists(WARM_START_G_PATH) and not RESUME_CGAN:
        print(f"🔄 Warm-start G from: {WARM_START_G_PATH}")
        ckpt = torch.load(WARM_START_G_PATH, map_location=device)
        G.load_state_dict(ckpt["model"], strict=True)

    if RESUME_CGAN:
        if os.path.exists(RESUME_G_PATH):
            start_epoch, best_val_g, _ = load_ckpt(
                RESUME_G_PATH, G, opt_G, scaler_G, device=device)
            print(
                f"🔄 Resume G: epoch={start_epoch}, best_val_g={best_val_g:.6f}")
        if os.path.exists(RESUME_D_PATH):
            _, _, _ = load_ckpt(RESUME_D_PATH, D, opt_D,
                                scaler_D, device=device)
            print("🔄 Resume D loaded.")

    config = {
        "SEED": SEED,
        "EPOCHS": EPOCHS,
        "LR_G": LR_G,
        "LR_D": LR_D,
        "ADV_START_EPOCH": ADV_START_EPOCH,
        "ADV_FULL_EPOCH": ADV_FULL_EPOCH,
        "ADV_MAX": ADV_MAX,
        "WARM_START_G": WARM_START_G,
        "WARM_START_G_PATH": WARM_START_G_PATH,
        "GENERATOR": "ResUNet3D_HQ",
        "DISCRIMINATOR": "PatchDiscriminator3D",
        "D_UPDATE_EVERY": D_UPDATE_EVERY,
    }

    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    init_csv(CSV_PATH)

    # -------------------------
    # TRAIN
    # -------------------------
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        t0 = time.time()

        lambda_adv = get_lambda_adv(
            epoch,
            adv_start_epoch=ADV_START_EPOCH,
            adv_full_epoch=ADV_FULL_EPOCH,
            adv_max=ADV_MAX
        )

        tr_stats = train_one_epoch_cgan(
            G=G, D=D, loader=train_loader,
            opt_G=opt_G, opt_D=opt_D,
            scaler_G=scaler_G, scaler_D=scaler_D,
            criterion_phys=criterion_phys,
            device=device,
            epoch=epoch,
            lambda_adv=lambda_adv,
            grad_clip_G=2.0,
            grad_clip_D=2.0,
            use_amp=True,
            d_update_every=D_UPDATE_EVERY,  
        )

        val_g_phys, val_comp = eval_generator(
            G=G, loader=val_loader, criterion_phys=criterion_phys,
            device=device, epoch=epoch, use_amp=True
        )

        prev_lr_g = opt_G.param_groups[0]["lr"]
        if scheduler_G is not None and np.isfinite(val_g_phys):
            scheduler_G.step(val_g_phys)
        new_lr_g = opt_G.param_groups[0]["lr"]
        new_lr_d = opt_D.param_groups[0]["lr"]

        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d} | "
            f"G_total={tr_stats['g_total']:.6f} | "
            f"G_phys={tr_stats['g_phys']:.6f} | "
            f"G_adv={tr_stats['g_adv']:.6f} | "
            f"D={tr_stats['d_total']:.6f} | "
            f"Val_phys={val_g_phys:.6f} | "
            f"Peak={val_comp['loss_peak']:.4f} | "
            f"Loc={val_comp['loss_loc']:.4f} | "
            f"lambda_adv={lambda_adv:.4f} | "
            f"lrG={new_lr_g:.2e} | lrD={new_lr_d:.2e} | "
            f"time={dt:.1f}s"
        )

        if new_lr_g < prev_lr_g:
            print(f"🔻 LR_G reduced: {prev_lr_g:.2e} -> {new_lr_g:.2e}")

        append_csv(CSV_PATH, epoch, tr_stats, val_g_phys,
                   val_comp, lambda_adv, new_lr_g, new_lr_d, dt)

        save_ckpt(os.path.join(SAVE_DIR, "last_G.pth"), G,
                  opt_G, scaler_G, epoch, best_val_g, config)
        save_ckpt(os.path.join(SAVE_DIR, "last_D.pth"), D,
                  opt_D, scaler_D, epoch, best_val_g, config)

        if val_g_phys < best_val_g:
            best_val_g = val_g_phys
            save_ckpt(os.path.join(SAVE_DIR, "best_G.pth"), G,
                      opt_G, scaler_G, epoch, best_val_g, config)
            save_ckpt(os.path.join(SAVE_DIR, "best_D_at_best_G.pth"),
                      D, opt_D, scaler_D, epoch, best_val_g, config)
            print(f"✅ New best G val phys: {best_val_g:.6f}")

        if (epoch % SAVE_EVERY_N_EPOCHS == 0) or (epoch in SPECIFIC_SAVE_EPOCHS):
            save_ckpt(os.path.join(
                SAVE_DIR, f"epoch_{epoch:03d}_G.pth"), G, opt_G, scaler_G, epoch, best_val_g, config)
            save_ckpt(os.path.join(
                SAVE_DIR, f"epoch_{epoch:03d}_D.pth"), D, opt_D, scaler_D, epoch, best_val_g, config)
            print(f"💾 Saved GAN checkpoints for epoch {epoch:03d}")

        visual_cb(G, epoch)

    print("✅ cGAN training done. Best generator val phys:", best_val_g)


if __name__ == "__main__":
    main()
