import random
import numpy as np
import torch
import torch.nn.functional as F


def seed_all(seed: int = 123):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grad3d(x: torch.Tensor):
    """
    Compute 3D gradients using finite differences with padding.

    Args:
        x: [B, 1, D, H, W]

    Returns:
        (dx, dy, dz) with original size padding
    """
    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dz = F.pad(dz, (0, 1, 0, 0, 0, 0))
    return dx, dy, dz


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """Compute mean of x where mask is non-zero."""
    return (x * mask).sum() / (mask.sum() + eps)


def ramp_linear(epoch: int, start: int, end: int) -> float:
    """Linear ramp from 0->1 between start and end epochs."""
    if epoch < start:
        return 0.0
    if epoch >= end:
        return 1.0
    return float(epoch - start) / float(max(1, end - start))


def interp_linear(alpha: float, start_value: float, end_value: float) -> float:
    """Linear interpolation between start_value and end_value."""
    alpha = float(max(0.0, min(1.0, alpha)))
    return (1.0 - alpha) * float(start_value) + alpha * float(end_value)


def make_focus_roi(target: torch.Tensor,
                   frac: float = 0.50,
                   min_thr: float = 0.08,
                   dilate_ks: int = 7):
    """
    Create focal ROI from target.

    Uses relative threshold to peak + dilation to include neighborhood.
    """
    peak = target.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    thr = torch.maximum(peak * frac, torch.full_like(peak, min_thr))
    roi = (target >= thr).float()

    if dilate_ks is not None and dilate_ks > 1:
        roi = F.max_pool3d(roi, kernel_size=dilate_ks,
                           stride=1, padding=dilate_ks // 2)
        roi = (roi > 0).float()

    return roi, peak


def make_peak_mask(target: torch.Tensor, frac: float = 0.85):
    """Create tight mask around peak voxels."""
    peak = target.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    mask = (target >= peak * frac).float()
    return mask, peak


def approx_max(x: torch.Tensor, mask: torch.Tensor = None, beta: float = 20.0):
    """
    Differentiable approximation of max using stable logsumexp.
    Computed in float32 for stability.
    """
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
    """
    Compute normalized coordinates [-1,1] of the max center using softmax.

    Args:
        x: [B, 1, D, H, W]
        mask: Optional mask
        beta: Temperature for softmax

    Returns:
        [B, 3] coordinates in [-1,1] range
    """
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
