from contextlib import nullcontext

import numpy as np
import torch
from scipy.ndimage import binary_fill_holes


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

        if brain_sel.any():
            mse_brain = float(((pred_np - gt_np) ** 2)[brain_sel].mean())
        else:
            mse_brain = float("nan")

        sums["peak_rel_err"] += peak_rel_err
        sums["peak_loc_err_mm"] += peak_loc_err_mm
        sums["dice50"] += dice50
        sums["mse_brain"] += mse_brain

        n += 1

    if n == 0:
        return {k: float("inf") for k in sums.keys()}

    return {k: v / n for k, v in sums.items()}