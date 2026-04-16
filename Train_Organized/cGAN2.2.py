import os
import csv
import glob
import time
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_fill_holes

from Unet3D_V2 import ResUNet3D_HQ


# =========================================================
# SSIM opcional
# =========================================================
try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except ImportError:
    try:
        from skimage.measure import compare_ssim as skimage_ssim
        HAS_SKIMAGE = True
    except ImportError:
        HAS_SKIMAGE = False
        skimage_ssim = None


# =========================================================
# CONFIG
# =========================================================
TRAIN_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/train"
VAL_DIR   = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/val"

# UNet base congelada
BASE_CKPT_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_expDexpB/epoch_030.pth"

# Checkpoint desde el que quieres reanudar el cGAN freeze ya entrenado
RESUME_CKPT_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_lowres_freeze2_expC/best_lowres_freeze_cgan.pth"

# Carpeta nueva para el fine-tuning final
OUT_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_lowres_freeze2_expC_finetune"

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

EXPECTED_SHAPE = (128, 128, 128)
LOW_RES_SHAPE = (48, 48, 48)

DX_MM = 1.0
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

BATCH_SIZE = 1

# �pocas EXTRA de fine-tuning
EPOCHS = 30

# LR m�s bajos para fine-tuning
LR_G = 1e-4
LR_D = 5e-5
BETAS = (0.5, 0.999)
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0

# Residual global peque�o
ALPHA_RESID = 0.05

# Warmup sin adversarial
ADV_WARMUP_EPOCHS = 3

# Loss weights base
LAMBDA_GLOBAL = 10.0
LAMBDA_BRAIN = 10.0
LAMBDA_FREEZE = 8.0
LAMBDA_ADV = 0.02

# Freeze base
FREEZE_RADIUS_VOX = 5


# =========================================================
# UTILIDADES
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_ssim_safe(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    if not HAS_SKIMAGE:
        return float("nan")

    if min(a.shape) < 7 or min(b.shape) < 7:
        return float("nan")

    try:
        return float(skimage_ssim(a, b, data_range=data_range))
    except Exception:
        return float("nan")


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-8) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    inter = np.logical_and(mask_a, mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()
    return float((2.0 * inter) / (denom + eps))


def crop_to_mask_bbox(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return vol
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    return vol[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]


def get_peak_index_masked(vol: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    masked = np.where(mask > 0, vol, -np.inf)
    flat_idx = int(np.argmax(masked))
    return np.unravel_index(flat_idx, vol.shape)


def reconstruct_brain_mask(mask_skull: np.ndarray, is_water_only: bool) -> np.ndarray:
    skull = mask_skull > 0.5

    if is_water_only or skull.sum() == 0:
        return np.zeros_like(mask_skull, dtype=np.float32)

    filled = binary_fill_holes(skull)
    brain = np.logical_and(filled, np.logical_not(skull))
    return brain.astype(np.float32)


def downsample_to_lowres(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=LOW_RES_SHAPE, mode="trilinear", align_corners=False)


def upsample_to_fullres(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=EXPECTED_SHAPE, mode="trilinear", align_corners=False)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = []
    for b in range(pred.shape[0]):
        m = mask[b, 0] > 0.5
        if m.sum() == 0:
            vals.append(torch.abs(pred[b:b+1] - target[b:b+1]).mean())
        else:
            vals.append(torch.abs(pred[b, 0][m] - target[b, 0][m]).mean())
    return torch.stack(vals).mean()


def make_freeze_focus_roi(
    y_gt: torch.Tensor,
    brain_mask: torch.Tensor,
    radius_vox: int = 5
) -> torch.Tensor:
    """
    Crea una esfera binaria [B,1,D,H,W] alrededor del pico GT dentro del cerebro.
    Si no hay cerebro, ROI = 0.
    """
    B, _, D, H, W = y_gt.shape
    roi = torch.zeros_like(y_gt)

    zz = torch.arange(D, device=y_gt.device).view(D, 1, 1)
    yy = torch.arange(H, device=y_gt.device).view(1, H, 1)
    xx = torch.arange(W, device=y_gt.device).view(1, 1, W)

    for b in range(B):
        brain = brain_mask[b, 0] > 0.5
        if brain.sum() == 0:
            continue

        gt_np = y_gt[b, 0].detach().cpu().numpy()
        brain_np = brain.detach().cpu().numpy()
        pz, py, px = get_peak_index_masked(gt_np, brain_np)

        dist2 = (zz - pz) ** 2 + (yy - py) ** 2 + (xx - px) ** 2
        sphere = dist2 <= (radius_vox ** 2)

        sphere = sphere & brain
        roi[b, 0] = sphere.float()

    return roi


def freeze_focus_loss(y_fake: torch.Tensor, y_base: torch.Tensor, freeze_roi: torch.Tensor) -> torch.Tensor:
    """
    Penaliza cambios respecto a y_base solo dentro de la ROI focal.
    """
    vals = []
    for b in range(y_fake.shape[0]):
        m = freeze_roi[b, 0] > 0.5
        if m.sum() == 0:
            vals.append(torch.tensor(0.0, device=y_fake.device, dtype=y_fake.dtype))
        else:
            vals.append(torch.abs(y_fake[b, 0][m] - y_base[b, 0][m]).mean())
    return torch.stack(vals).mean()


def get_lambda_freeze(epoch: int) -> float:
    if epoch <= 10:
        return 8.0
    elif epoch <= 20:
        return 6.0
    else:
        return 4.0


def get_freeze_radius(epoch: int) -> int:
    if epoch <= 10:
        return 5
    elif epoch <= 20:
        return 4
    else:
        return 4


# =========================================================
# DATASET
# =========================================================
class TusGanDataset(Dataset):
    def __init__(self, data_dir: str, expected_shape=(128, 128, 128)):
        super().__init__()
        self.data_dir = data_dir
        self.expected_shape = tuple(expected_shape)
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if len(self.files) == 0:
            raise RuntimeError(f"No se encontraron archivos .npz en: {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]

        with np.load(path) as d:
            src = d["source_mask"].astype(np.float32)
            skull = d["mask_skull"].astype(np.float32)
            y = d["p_max_norm"].astype(np.float32)
            is_water_only = bool(np.array(d["is_water_only"]).item())

        for name, arr in [
            ("source_mask", src),
            ("mask_skull", skull),
            ("p_max_norm", y),
        ]:
            if arr.shape != self.expected_shape:
                raise RuntimeError(
                    f"Shape inv�lida en {os.path.basename(path)} | "
                    f"{name}: {arr.shape} | esperado: {self.expected_shape}"
                )

        brain = reconstruct_brain_mask(skull, is_water_only)

        # Mismo orden que el UNet base
        X = np.stack([src, skull], axis=0)   # [2, D, H, W]
        y = y[np.newaxis, ...]               # [1, D, H, W]
        brain = brain[np.newaxis, ...]       # [1, D, H, W]

        return {
            "X": torch.from_numpy(X),
            "y": torch.from_numpy(y),
            "brain_mask": torch.from_numpy(brain),
            "is_water_only": is_water_only,
            "file_name": os.path.basename(path),
        }


# =========================================================
# MODELOS
# =========================================================
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=not norm)
        ]
        if norm:
            num_groups = 8 if out_ch >= 8 else 1
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        num_groups = 8 if ch >= 8 else 1
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        r = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        x = self.act(x + r)
        return x


class LowResGlobalGenerator3D(nn.Module):
    """
    Generador peque�o que trabaja en baja resoluci�n.
    Entrada: [x_low, y_base_low] -> 3 canales si x tiene 2.
    Salida: delta_low en [-1,1]
    """
    def __init__(self, x_ch: int = 2):
        super().__init__()
        in_ch = x_ch + 1

        self.enc1 = nn.Sequential(
            ConvBlock3D(in_ch, 16, stride=1, norm=True),
            ConvBlock3D(16, 16, stride=1, norm=True),
        )
        self.down1 = nn.Sequential(
            ConvBlock3D(16, 32, stride=2, norm=True),
            ConvBlock3D(32, 32, stride=1, norm=True),
        )
        self.down2 = nn.Sequential(
            ConvBlock3D(32, 64, stride=2, norm=True),
            ConvBlock3D(64, 64, stride=1, norm=True),
        )

        self.mid = nn.Sequential(
            ResidualBlock3D(64),
            ResidualBlock3D(64),
        )

        self.up1 = nn.Sequential(
            ConvBlock3D(64 + 32, 32, stride=1, norm=True),
            ConvBlock3D(32, 32, stride=1, norm=True),
        )
        self.up2 = nn.Sequential(
            ConvBlock3D(32 + 16, 16, stride=1, norm=True),
            ConvBlock3D(16, 16, stride=1, norm=True),
        )

        self.out_conv = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x_low, y_base_low):
        z = torch.cat([x_low, y_base_low], dim=1)

        s1 = self.enc1(z)   # 48^3
        s2 = self.down1(s1) # 24^3
        s3 = self.down2(s2) # 12^3

        m = self.mid(s3)

        u1 = F.interpolate(m, size=s2.shape[-3:], mode="trilinear", align_corners=False)
        u1 = torch.cat([u1, s2], dim=1)
        u1 = self.up1(u1)

        u2 = F.interpolate(u1, size=s1.shape[-3:], mode="trilinear", align_corners=False)
        u2 = torch.cat([u2, s1], dim=1)
        u2 = self.up2(u2)

        delta_low = torch.tanh(self.out_conv(u2))
        return delta_low


class LowResPatchDiscriminator3D(nn.Module):
    """
    Discriminador condicional solo en baja resoluci�n.
    Entrada: concat([x_low, y_low])
    """
    def __init__(self, x_ch: int = 2):
        super().__init__()
        in_ch = x_ch + 1

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x_low, y_low):
        z = torch.cat([x_low, y_low], dim=1)
        return self.net(z)


# =========================================================
# CARGA DEL UNET BASE
# =========================================================
def load_base_unet(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    model = ResUNet3D_HQ(
        in_ch=2,
        out_ch=1,
        base=16,
        norm_kind="group",
        use_se=True,
        out_positive=True,
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, ckpt


# =========================================================
# RESUME DEL cGAN
# =========================================================
def load_resume_lowres_freeze_ckpt(
    ckpt_path: str,
    G: nn.Module,
    D: nn.Module,
    device: str,
):
    ckpt = torch.load(ckpt_path, map_location=device)

    G.load_state_dict(ckpt["generator_lowres"], strict=True)
    D.load_state_dict(ckpt["discriminator_lowres"], strict=True)

    start_epoch = int(ckpt.get("epoch", 0))
    best_score = float(ckpt.get("val_stats", {}).get("selection_score", float("inf")))
    return ckpt, start_epoch, best_score


# =========================================================
# LOSSES GAN
# =========================================================
def discriminator_lsgan_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    loss_real = ((d_real - 1.0) ** 2).mean()
    loss_fake = (d_fake ** 2).mean()
    return 0.5 * (loss_real + loss_fake)


def generator_adv_lsgan_loss(d_fake_for_g: torch.Tensor) -> torch.Tensor:
    return ((d_fake_for_g - 1.0) ** 2).mean()


def build_fake_output(y_base: torch.Tensor, delta_low: torch.Tensor, alpha_resid: float) -> torch.Tensor:
    delta_full = upsample_to_fullres(delta_low)
    y_fake = y_base + alpha_resid * delta_full
    y_fake = y_fake.clamp_min(0.0)
    return y_fake


def compute_generator_losses(
    D: nn.Module,
    X: torch.Tensor,
    y_gt: torch.Tensor,
    y_base: torch.Tensor,
    brain_mask: torch.Tensor,
    freeze_roi: torch.Tensor,
    G: nn.Module,
    alpha_resid: float,
    lambda_adv_current: float,
    lambda_freeze_current: float,
):
    x_low = downsample_to_lowres(X)
    y_base_low = downsample_to_lowres(y_base)
    _ = downsample_to_lowres(y_gt)  # por claridad, aunque no se use expl�citamente abajo

    delta_low = G(x_low, y_base_low)
    y_fake = build_fake_output(y_base, delta_low, alpha_resid)
    y_fake_low = downsample_to_lowres(y_fake)

    loss_l1_global = F.l1_loss(y_fake, y_gt)
    loss_l1_brain = masked_l1_loss(y_fake, y_gt, brain_mask)
    loss_freeze = freeze_focus_loss(y_fake, y_base, freeze_roi)

    d_fake_for_g = D(x_low, y_fake_low)
    loss_adv = generator_adv_lsgan_loss(d_fake_for_g)

    loss_g = (
        LAMBDA_GLOBAL * loss_l1_global +
        LAMBDA_BRAIN * loss_l1_brain +
        lambda_freeze_current * loss_freeze +
        lambda_adv_current * loss_adv
    )

    return {
        "loss_g_total": loss_g,
        "loss_l1_global": loss_l1_global,
        "loss_l1_brain": loss_l1_brain,
        "loss_freeze": loss_freeze,
        "loss_adv": loss_adv,
        "y_fake": y_fake,
        "y_fake_low": y_fake_low,
        "x_low": x_low,
    }


# =========================================================
# M�TRICAS DE VALIDACI�N
# =========================================================
def compute_metrics_one_sample(
    pred: np.ndarray,
    gt: np.ndarray,
    brain_mask: np.ndarray,
    dx_mm: float,
    is_water_only: bool,
) -> Dict[str, float]:

    pred = np.clip(pred.astype(np.float32), 0.0, None)
    gt = gt.astype(np.float32)
    brain_mask = (brain_mask > 0.5)

    mse_global = float(np.mean((pred - gt) ** 2))
    mae_global = float(np.mean(np.abs(pred - gt)))
    ssim_global = compute_ssim_safe(pred, gt, data_range=1.0)

    out = {
        "mse_global": mse_global,
        "mae_global": mae_global,
        "ssim_global": ssim_global,
        "is_water_only": int(is_water_only),
    }

    if is_water_only or brain_mask.sum() == 0:
        out.update({
            "mse_brain": float("nan"),
            "mae_brain": float("nan"),
            "ssim_brain": float("nan"),
            "peak_abs_err_brain": float("nan"),
            "peak_rel_err_brain": float("nan"),
            "peak_loc_err_vox_brain": float("nan"),
            "peak_loc_err_mm_brain": float("nan"),
            "dice_focus_brain_thr50": float("nan"),
        })
        return out

    pred_brain_vals = pred[brain_mask]
    gt_brain_vals = gt[brain_mask]

    mse_brain = float(np.mean((pred_brain_vals - gt_brain_vals) ** 2))
    mae_brain = float(np.mean(np.abs(pred_brain_vals - gt_brain_vals)))

    pred_brain_vol = pred * brain_mask.astype(np.float32)
    gt_brain_vol = gt * brain_mask.astype(np.float32)

    pred_brain_crop = crop_to_mask_bbox(pred_brain_vol, brain_mask)
    gt_brain_crop = crop_to_mask_bbox(gt_brain_vol, brain_mask)
    ssim_brain = compute_ssim_safe(pred_brain_crop, gt_brain_crop, data_range=1.0)

    gt_peak_idx = get_peak_index_masked(gt, brain_mask)
    pred_peak_idx = get_peak_index_masked(pred, brain_mask)

    gt_peak_val_brain = float(gt[gt_peak_idx])
    pred_peak_val_brain = float(pred[pred_peak_idx])

    peak_abs_err_brain = abs(pred_peak_val_brain - gt_peak_val_brain)
    peak_rel_err_brain = peak_abs_err_brain / (abs(gt_peak_val_brain) + 1e-8)

    peak_loc_err_vox_brain = math.sqrt(
        (pred_peak_idx[0] - gt_peak_idx[0]) ** 2 +
        (pred_peak_idx[1] - gt_peak_idx[1]) ** 2 +
        (pred_peak_idx[2] - gt_peak_idx[2]) ** 2
    )
    peak_loc_err_mm_brain = peak_loc_err_vox_brain * dx_mm

    gt_thr50 = 0.50 * gt_peak_val_brain
    pr_thr50 = 0.50 * pred_peak_val_brain

    gt_focus_thr50 = np.logical_and(gt >= gt_thr50, brain_mask)
    pr_focus_thr50 = np.logical_and(pred >= pr_thr50, brain_mask)

    dice_focus_brain_thr50 = dice_score(gt_focus_thr50, pr_focus_thr50)

    out.update({
        "mse_brain": mse_brain,
        "mae_brain": mae_brain,
        "ssim_brain": ssim_brain,
        "peak_abs_err_brain": peak_abs_err_brain,
        "peak_rel_err_brain": peak_rel_err_brain,
        "peak_loc_err_vox_brain": float(peak_loc_err_vox_brain),
        "peak_loc_err_mm_brain": float(peak_loc_err_mm_brain),
        "dice_focus_brain_thr50": dice_focus_brain_thr50,
    })

    return out


def summarize_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys_num = [
        "mse_global",
        "mae_global",
        "ssim_global",
        "mse_brain",
        "mae_brain",
        "ssim_brain",
        "peak_abs_err_brain",
        "peak_rel_err_brain",
        "peak_loc_err_vox_brain",
        "peak_loc_err_mm_brain",
        "dice_focus_brain_thr50",
    ]

    n_samples = len(rows)
    n_water_only = int(sum(int(r["is_water_only"]) for r in rows))
    n_non_water_only = int(n_samples - n_water_only)

    out = {
        "n_samples": n_samples,
        "n_water_only": n_water_only,
        "n_non_water_only": n_non_water_only,
    }

    for k in keys_num:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            out[f"mean_{k}"] = float("nan")
            out[f"std_{k}"] = float("nan")
        else:
            out[f"mean_{k}"] = float(np.mean(vals))
            out[f"std_{k}"] = float(np.std(vals))

    return out


def model_selection_score(summary: Dict[str, float]) -> float:
    def safe(key: str, default: float):
        v = summary.get(key, float("nan"))
        return default if not np.isfinite(v) else float(v)

    peak_loc = safe("mean_peak_loc_err_mm_brain", 1e6)
    peak_rel = safe("mean_peak_rel_err_brain", 1e6)
    mae_brain = safe("mean_mae_brain", 1e6)
    dice50 = safe("mean_dice_focus_brain_thr50", 0.0)
    ssim_brain = safe("mean_ssim_brain", 0.0)

    score = (
        0.45 * peak_loc +
        0.25 * peak_rel +
        0.20 * mae_brain -
        0.05 * dice50 -
        0.05 * ssim_brain
    )
    return float(score)


# =========================================================
# TRAIN / VAL
# =========================================================
def train_one_epoch(
    epoch: int,
    base_model: nn.Module,
    G: nn.Module,
    D: nn.Module,
    loader: DataLoader,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    lambda_freeze_current: float,
    freeze_radius_current: int,
):
    G.train()
    D.train()

    lambda_adv_current = 0.0 if epoch <= ADV_WARMUP_EPOCHS else LAMBDA_ADV

    accum = {
        "loss_d": 0.0,
        "loss_g_total": 0.0,
        "loss_l1_global": 0.0,
        "loss_l1_brain": 0.0,
        "loss_freeze": 0.0,
        "loss_adv": 0.0,
    }

    n_batches = 0
    t0 = time.time()

    for batch in loader:
        X = batch["X"].to(DEVICE, non_blocking=True)
        y = batch["y"].to(DEVICE, non_blocking=True)
        brain = batch["brain_mask"].to(DEVICE, non_blocking=True)

        with torch.no_grad():
            y_base = base_model(X).clamp_min(0.0)

        freeze_roi = make_freeze_focus_roi(
            y_gt=y,
            brain_mask=brain,
            radius_vox=freeze_radius_current,
        )

        # -------------------------
        # 1) Update D
        # -------------------------
        opt_d.zero_grad(set_to_none=True)

        x_low = downsample_to_lowres(X)
        y_base_low = downsample_to_lowres(y_base)
        y_low = downsample_to_lowres(y)

        delta_low_det = G(x_low, y_base_low)
        y_fake_det = build_fake_output(y_base, delta_low_det, ALPHA_RESID)
        y_fake_low_det = downsample_to_lowres(y_fake_det)

        d_real = D(x_low, y_low)
        d_fake = D(x_low, y_fake_low_det.detach())
        loss_d = discriminator_lsgan_loss(d_real, d_fake)

        loss_d.backward()
        if GRAD_CLIP is not None and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(D.parameters(), GRAD_CLIP)
        opt_d.step()

        # -------------------------
        # 2) Update G
        # -------------------------
        opt_g.zero_grad(set_to_none=True)

        g_losses = compute_generator_losses(
            D=D,
            X=X,
            y_gt=y,
            y_base=y_base,
            brain_mask=brain,
            freeze_roi=freeze_roi,
            G=G,
            alpha_resid=ALPHA_RESID,
            lambda_adv_current=lambda_adv_current,
            lambda_freeze_current=lambda_freeze_current,
        )

        loss_g = g_losses["loss_g_total"]
        loss_g.backward()

        if GRAD_CLIP is not None and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(G.parameters(), GRAD_CLIP)

        opt_g.step()

        accum["loss_d"] += float(loss_d.item())
        accum["loss_g_total"] += float(g_losses["loss_g_total"].item())
        accum["loss_l1_global"] += float(g_losses["loss_l1_global"].item())
        accum["loss_l1_brain"] += float(g_losses["loss_l1_brain"].item())
        accum["loss_freeze"] += float(g_losses["loss_freeze"].item())
        accum["loss_adv"] += float(g_losses["loss_adv"].item())

        n_batches += 1

    for k in accum:
        accum[k] /= max(n_batches, 1)

    accum["time_sec"] = time.time() - t0
    accum["lambda_adv_current"] = lambda_adv_current
    accum["lambda_freeze_current"] = lambda_freeze_current
    accum["freeze_radius_current"] = freeze_radius_current
    return accum


@torch.no_grad()
def validate_one_epoch(
    epoch: int,
    base_model: nn.Module,
    G: nn.Module,
    D: nn.Module,
    loader: DataLoader,
    lambda_freeze_current: float,
    freeze_radius_current: int,
):
    G.eval()
    D.eval()

    lambda_adv_current = 0.0 if epoch <= ADV_WARMUP_EPOCHS else LAMBDA_ADV

    accum = {
        "val_loss_g_total": 0.0,
        "val_loss_l1_global": 0.0,
        "val_loss_l1_brain": 0.0,
        "val_loss_freeze": 0.0,
        "val_loss_adv": 0.0,
    }

    rows = []
    n_batches = 0

    for batch in loader:
        X = batch["X"].to(DEVICE, non_blocking=True)
        y = batch["y"].to(DEVICE, non_blocking=True)
        brain = batch["brain_mask"].to(DEVICE, non_blocking=True)
        is_water_only = bool(batch["is_water_only"][0])
        file_name = batch["file_name"][0]

        y_base = base_model(X).clamp_min(0.0)

        freeze_roi = make_freeze_focus_roi(
            y_gt=y,
            brain_mask=brain,
            radius_vox=freeze_radius_current,
        )

        g_losses = compute_generator_losses(
            D=D,
            X=X,
            y_gt=y,
            y_base=y_base,
            brain_mask=brain,
            freeze_roi=freeze_roi,
            G=G,
            alpha_resid=ALPHA_RESID,
            lambda_adv_current=lambda_adv_current,
            lambda_freeze_current=lambda_freeze_current,
        )

        y_fake = g_losses["y_fake"]

        accum["val_loss_g_total"] += float(g_losses["loss_g_total"].item())
        accum["val_loss_l1_global"] += float(g_losses["loss_l1_global"].item())
        accum["val_loss_l1_brain"] += float(g_losses["loss_l1_brain"].item())
        accum["val_loss_freeze"] += float(g_losses["loss_freeze"].item())
        accum["val_loss_adv"] += float(g_losses["loss_adv"].item())
        n_batches += 1

        pred_np = y_fake[0, 0].detach().cpu().numpy()
        gt_np = y[0, 0].detach().cpu().numpy()
        brain_np = brain[0, 0].detach().cpu().numpy()

        metrics = compute_metrics_one_sample(
            pred=pred_np,
            gt=gt_np,
            brain_mask=brain_np,
            dx_mm=DX_MM,
            is_water_only=is_water_only,
        )
        metrics["file"] = file_name
        rows.append(metrics)

    for k in accum:
        accum[k] /= max(n_batches, 1)

    summary = summarize_metrics(rows)
    sel_score = model_selection_score(summary)

    accum.update(summary)
    accum["selection_score"] = sel_score
    accum["lambda_freeze_current"] = lambda_freeze_current
    accum["freeze_radius_current"] = freeze_radius_current
    return accum, rows


# =========================================================
# CHECKPOINTS / LOGS
# =========================================================
def save_checkpoint(
    path: str,
    epoch: int,
    G: nn.Module,
    D: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    base_ckpt_path: str,
    train_stats: Dict[str, float],
    val_stats: Dict[str, float],
    lambda_freeze_current: float,
    freeze_radius_current: int,
):
    ckpt = {
        "epoch": epoch,
        "generator_lowres": G.state_dict(),
        "discriminator_lowres": D.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "base_ckpt_path": base_ckpt_path,
        "alpha_resid": ALPHA_RESID,
        "low_res_shape": LOW_RES_SHAPE,
        "config": {
            "LAMBDA_GLOBAL": LAMBDA_GLOBAL,
            "LAMBDA_BRAIN": LAMBDA_BRAIN,
            "LAMBDA_FREEZE_BASE": LAMBDA_FREEZE,
            "LAMBDA_ADV": LAMBDA_ADV,
            "ADV_WARMUP_EPOCHS": ADV_WARMUP_EPOCHS,
            "FREEZE_RADIUS_VOX_BASE": FREEZE_RADIUS_VOX,
            "DX_MM": DX_MM,
            "BATCH_SIZE": BATCH_SIZE,
            "LR_G": LR_G,
            "LR_D": LR_D,
        },
        "current_epoch_settings": {
            "lambda_freeze_current": lambda_freeze_current,
            "freeze_radius_current": freeze_radius_current,
        },
        "train_stats": train_stats,
        "val_stats": val_stats,
    }
    torch.save(ckpt, path)


def append_log_csv(csv_path: str, row: Dict[str, float]):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_val_per_sample_csv(csv_path: str, rows: List[Dict[str, float]]):
    if len(rows) == 0:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)

    print("========================================")
    print("Low-Resolution Global Residual cGAN + Freeze | Final Fine-Tuning")
    print("========================================")
    print("DEVICE:", DEVICE)
    print("TRAIN_DIR:", TRAIN_DIR)
    print("VAL_DIR:", VAL_DIR)
    print("BASE_CKPT_PATH:", BASE_CKPT_PATH)
    print("RESUME_CKPT_PATH:", RESUME_CKPT_PATH)
    print("OUT_DIR:", OUT_DIR)
    print("LOW_RES_SHAPE:", LOW_RES_SHAPE)
    print("ALPHA_RESID:", ALPHA_RESID)
    print("========================================")

    train_ds = TusGanDataset(TRAIN_DIR, expected_shape=EXPECTED_SHAPE)
    val_ds = TusGanDataset(VAL_DIR, expected_shape=EXPECTED_SHAPE)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    base_model, base_ckpt = load_base_unet(BASE_CKPT_PATH, DEVICE)
    print(f"UNet base cargado desde epoch={int(base_ckpt.get('epoch', -1))}")

    G = LowResGlobalGenerator3D(x_ch=2).to(DEVICE)
    D = LowResPatchDiscriminator3D(x_ch=2).to(DEVICE)

    opt_g = torch.optim.Adam(G.parameters(), lr=LR_G, betas=BETAS, weight_decay=WEIGHT_DECAY)
    opt_d = torch.optim.Adam(D.parameters(), lr=LR_D, betas=BETAS, weight_decay=WEIGHT_DECAY)

    start_epoch = 0
    best_score = float("inf")

    if RESUME_CKPT_PATH is not None and os.path.exists(RESUME_CKPT_PATH):
        resume_ckpt, start_epoch, best_score = load_resume_lowres_freeze_ckpt(
            ckpt_path=RESUME_CKPT_PATH,
            G=G,
            D=D,
            device=DEVICE,
        )
        print(f"Reanudando desde: {RESUME_CKPT_PATH}")
        print(f"Checkpoint cargado de epoch={start_epoch}")
        print(f"Best score previo={best_score:.6f}" if np.isfinite(best_score) else "Best score previo no disponible")
    else:
        print("No se reanuda ning�n checkpoint.")

    log_csv = os.path.join(OUT_DIR, "training_log.csv")

    for epoch in range(start_epoch + 1, start_epoch + EPOCHS + 1):
        print(f"\n===== Epoch {epoch:03d}/{start_epoch + EPOCHS:03d} =====")

        lambda_freeze_current = get_lambda_freeze(epoch)
        freeze_radius_current = get_freeze_radius(epoch)

        train_stats = train_one_epoch(
            epoch=epoch,
            base_model=base_model,
            G=G,
            D=D,
            loader=train_loader,
            opt_g=opt_g,
            opt_d=opt_d,
            lambda_freeze_current=lambda_freeze_current,
            freeze_radius_current=freeze_radius_current,
        )

        val_stats, val_rows = validate_one_epoch(
            epoch=epoch,
            base_model=base_model,
            G=G,
            D=D,
            loader=val_loader,
            lambda_freeze_current=lambda_freeze_current,
            freeze_radius_current=freeze_radius_current,
        )

        log_row = {
            "epoch": epoch,
            "train_loss_d": train_stats["loss_d"],
            "train_loss_g_total": train_stats["loss_g_total"],
            "train_loss_l1_global": train_stats["loss_l1_global"],
            "train_loss_l1_brain": train_stats["loss_l1_brain"],
            "train_loss_freeze": train_stats["loss_freeze"],
            "train_loss_adv": train_stats["loss_adv"],
            "train_time_sec": train_stats["time_sec"],
            "lambda_adv_current": train_stats["lambda_adv_current"],
            "lambda_freeze_current": train_stats["lambda_freeze_current"],
            "freeze_radius_current": train_stats["freeze_radius_current"],

            "val_loss_g_total": val_stats["val_loss_g_total"],
            "val_loss_l1_global": val_stats["val_loss_l1_global"],
            "val_loss_l1_brain": val_stats["val_loss_l1_brain"],
            "val_loss_freeze": val_stats["val_loss_freeze"],
            "val_loss_adv": val_stats["val_loss_adv"],

            "mean_mae_global": val_stats["mean_mae_global"],
            "mean_ssim_global": val_stats["mean_ssim_global"],
            "mean_mae_brain": val_stats["mean_mae_brain"],
            "mean_ssim_brain": val_stats["mean_ssim_brain"],
            "mean_peak_rel_err_brain": val_stats["mean_peak_rel_err_brain"],
            "mean_peak_loc_err_mm_brain": val_stats["mean_peak_loc_err_mm_brain"],
            "mean_dice_focus_brain_thr50": val_stats["mean_dice_focus_brain_thr50"],
            "selection_score": val_stats["selection_score"],
        }
        append_log_csv(log_csv, log_row)

        print(
            f"Train | D={train_stats['loss_d']:.4f} | G={train_stats['loss_g_total']:.4f} | "
            f"L1g={train_stats['loss_l1_global']:.4f} | L1brain={train_stats['loss_l1_brain']:.4f} | "
            f"Freeze={train_stats['loss_freeze']:.4f} | Adv={train_stats['loss_adv']:.4f} | "
            f"freeze_lambda={lambda_freeze_current:.2f} | freeze_r={freeze_radius_current} | "
            f"t={train_stats['time_sec']:.1f}s"
        )

        print(
            f"Val   | G={val_stats['val_loss_g_total']:.4f} | "
            f"MAEbrain={val_stats['mean_mae_brain']:.5f} | "
            f"SSIMbrain={val_stats['mean_ssim_brain']:.5f} | "
            f"PeakRel={val_stats['mean_peak_rel_err_brain']:.5f} | "
            f"PeakLoc(mm)={val_stats['mean_peak_loc_err_mm_brain']:.5f} | "
            f"Dice50={val_stats['mean_dice_focus_brain_thr50']:.5f} | "
            f"Score={val_stats['selection_score']:.5f}"
        )

        val_per_sample_csv = os.path.join(OUT_DIR, f"val_epoch_{epoch:03d}_per_sample.csv")
        save_val_per_sample_csv(val_per_sample_csv, val_rows)

        save_checkpoint(
            path=os.path.join(OUT_DIR, "last_lowres_freeze_cgan.pth"),
            epoch=epoch,
            G=G,
            D=D,
            opt_g=opt_g,
            opt_d=opt_d,
            base_ckpt_path=BASE_CKPT_PATH,
            train_stats=train_stats,
            val_stats=val_stats,
            lambda_freeze_current=lambda_freeze_current,
            freeze_radius_current=freeze_radius_current,
        )

        if val_stats["selection_score"] < best_score:
            best_score = val_stats["selection_score"]
            save_checkpoint(
                path=os.path.join(OUT_DIR, "best_lowres_freeze_cgan.pth"),
                epoch=epoch,
                G=G,
                D=D,
                opt_g=opt_g,
                opt_d=opt_d,
                base_ckpt_path=BASE_CKPT_PATH,
                train_stats=train_stats,
                val_stats=val_stats,
                lambda_freeze_current=lambda_freeze_current,
                freeze_radius_current=freeze_radius_current,
            )
            print(f"Nuevo mejor modelo | selection_score={best_score:.5f}")

    print("\nFine-tuning terminado.")
    print("Best checkpoint:", os.path.join(OUT_DIR, "best_lowres_freeze_cgan.pth"))
    print("Last checkpoint:", os.path.join(OUT_DIR, "last_lowres_freeze_cgan.pth"))
    print("Log CSV:", log_csv)


if __name__ == "__main__":
    main()