import torch

from src.losses.basic_ops import masked_mean


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


def build_tube_roi_and_tcoord(
    source_mask: torch.Tensor,
    target: torch.Tensor,
    radius_vox: int = 3
):
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

    src_center = get_source_center_from_mask(source_mask).to(
        device=device,
        dtype=dtype
    )

    z_idx, y_idx, x_idx = get_gt_peak_indices(target)

    peak_center = torch.stack(
        [z_idx.float(), y_idx.float(), x_idx.float()],
        dim=1
    ).to(device=device, dtype=dtype)

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


def compute_profile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tube_roi: torch.Tensor,
    t_coord: torch.Tensor,
    num_bins: int = 16,
    eps: float = 1e-6
):
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

        loss = torch.abs(pred_m - tgt_m) / (torch.abs(tgt_m) + eps)
        losses.append(loss)

    return torch.stack(losses).mean()