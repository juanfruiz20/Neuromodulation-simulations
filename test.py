import os
import re
import csv
import glob
import json
import math
import random
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# SSIM opcional
try:
    from pytorch_msssim import ssim as torch_ssim
    HAS_TORCH_SSIM = True
except Exception:
    HAS_TORCH_SSIM = False

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE_SSIM = True
except Exception:
    HAS_SKIMAGE_SSIM = False

from Data_loader import list_npz_files, split_files, TusDataset
from Model_Unet3D import ResUNet3D_HQ


# =========================================================
# CONFIG
# =========================================================

EVAL_CKPT_DIR = r"C:\Users\USUARIO\Desktop\UIC Bioingenieria\TFG\modelosTest"

CKPT_NAMES = None
# CKPT_NAMES = [
#     "epoch_090.pth",
#     "epoch_060.pth",
#     "epoch_080.pth",
#     "best_old_train.pth",
# ]

# ----------------------------
# TEST SPLIT
# ----------------------------
DATA_DIR_OVERRIDE = r"C:\Users\USUARIO\Desktop\UIC Bioingenieria\TFG\dataset_TUS_1000"
SEED_OVERRIDE = 123
TRAIN_RATIO_OVERRIDE = 0.85
VAL_RATIO_OVERRIDE = 0.10

REFERENCE_SPLIT_CKPT = None

# ----------------------------
# Evaluación
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
PIN_MEMORY = True
BATCH_SIZE = 1
USE_AMP = True

# ----------------------------
# Métricas físicas
# ----------------------------
DX_MM = 0.5
FOCUS_RADIUS_MM = 3.0          # esfera focal alrededor del pico GT
FOCUS_LOC_RADIUS_MM = 12.0     # ROI grande alrededor del focus_pos
SIGNAL_THR = 0.05

# ----------------------------
# Ranking compuesto
# ----------------------------
# Menor = mejor
WEIGHT_PEAK_LOC_MM = 0.60
WEIGHT_PEAK_REL = 0.25
WEIGHT_MAE_FOCUS = 0.15

# ----------------------------
# Visualización opcional
# Se hará SOLO para el mejor checkpoint final
# ----------------------------
MAKE_VIS = True
OUT_DIR = EVAL_CKPT_DIR
FIG_DIR = os.path.join(OUT_DIR, "best_ckpt_figs")
os.makedirs(FIG_DIR, exist_ok=True)

VIS_RANDOM = True
N_VIS_RANDOM = 6
FORCE_VIS_FILES = set()
PROFILE_SAMPLES = 256
CMAP = "jet"

# ----------------------------
# SSIM
# ----------------------------
SSIM_CLIP_01 = True
SSIM_DATA_RANGE = 1.0


# =========================================================
# Reproducibilidad
# =========================================================
def seed_all(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Utilidades generales
# =========================================================
def parse_epoch_from_name(path: str):
    name = os.path.basename(path)
    m = re.search(r"epoch_(\d+)\.pth$", name)
    if m:
        return int(m.group(1))
    return None


def find_checkpoints(ckpt_dir: str, ckpt_names: Optional[List[str]] = None) -> List[str]:
    if ckpt_names is None:
        files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
    else:
        files = [os.path.join(ckpt_dir, x) for x in ckpt_names]

    files = [p for p in files if os.path.exists(p)]
    if len(files) == 0:
        raise FileNotFoundError(f"No encontré checkpoints en: {ckpt_dir}")

    return files


def load_ckpt_raw(path: str, device="cpu"):
    return torch.load(path, map_location=device)


def get_split_params():
    if DATA_DIR_OVERRIDE is not None:
        return {
            "DATA_DIR": DATA_DIR_OVERRIDE,
            "SEED": SEED_OVERRIDE,
            "TRAIN_RATIO": TRAIN_RATIO_OVERRIDE,
            "VAL_RATIO": VAL_RATIO_OVERRIDE,
        }

    if REFERENCE_SPLIT_CKPT is None:
        raise ValueError(
            "No hay DATA_DIR_OVERRIDE ni REFERENCE_SPLIT_CKPT. "
            "Debes definir uno de los dos."
        )

    raw = load_ckpt_raw(REFERENCE_SPLIT_CKPT, device="cpu")
    cfg = raw.get("config", {})

    data_dir = cfg.get("DATA_DIR", None)
    seed = cfg.get("SEED", 123)
    train_ratio = cfg.get("TRAIN_RATIO", 0.85)
    val_ratio = cfg.get("VAL_RATIO", 0.10)

    if data_dir is None:
        raise ValueError(
            "El checkpoint de referencia no contiene DATA_DIR en config.")

    return {
        "DATA_DIR": data_dir,
        "SEED": seed,
        "TRAIN_RATIO": train_ratio,
        "VAL_RATIO": val_ratio,
    }


def build_model_from_ckpt_config(cfg: Dict[str, Any]):
    return ResUNet3D_HQ(
        in_ch=cfg.get("IN_CH", 4),
        out_ch=cfg.get("OUT_CH", 1),
        base=cfg.get("BASE", 16),
        norm_kind=cfg.get("NORM_KIND", "group"),
        use_se=cfg.get("USE_SE", True),
        out_positive=cfg.get("OUT_POSITIVE", True),
    )


# =========================================================
# Denormalización del target
# =========================================================
def get_y_denorm_mode_and_params(stats: Dict[str, Any]):
    if stats is None:
        return ("identity", None)

    if isinstance(stats, dict) and "y" in stats and isinstance(stats["y"], dict):
        ystats = stats["y"]
        if "mean" in ystats and "std" in ystats:
            return ("zscore", (float(ystats["mean"]), float(ystats["std"])))
        if "mu" in ystats and "sigma" in ystats:
            return ("zscore", (float(ystats["mu"]), float(ystats["sigma"])))
        if "min" in ystats and "max" in ystats:
            return ("minmax", (float(ystats["min"]), float(ystats["max"])))

    zscore_keys = [
        ("y_mean", "y_std"),
        ("target_mean", "target_std"),
        ("mean_y", "std_y"),
        ("output_mean", "output_std"),
    ]
    for k1, k2 in zscore_keys:
        if k1 in stats and k2 in stats:
            return ("zscore", (float(stats[k1]), float(stats[k2])))

    minmax_keys = [
        ("y_min", "y_max"),
        ("target_min", "target_max"),
        ("min_y", "max_y"),
        ("output_min", "output_max"),
    ]
    for k1, k2 in minmax_keys:
        if k1 in stats and k2 in stats:
            return ("minmax", (float(stats[k1]), float(stats[k2])))

    return ("identity", None)


def denormalize_y(y: torch.Tensor, mode: str, params):
    if mode == "identity" or params is None:
        return y
    if mode == "zscore":
        mean, std = params
        return y * std + mean
    if mode == "minmax":
        y_min, y_max = params
        return y * (y_max - y_min) + y_min
    return y


# =========================================================
# Métricas base
# =========================================================
def argmax_3d(vol: np.ndarray) -> Tuple[int, int, int]:
    return np.unravel_index(np.argmax(vol), vol.shape)


def peak_loc_error_mm(pred: np.ndarray, target: np.ndarray, dx_mm: float) -> float:
    p = argmax_3d(pred)
    t = argmax_3d(target)
    dist_vox = math.sqrt((p[0] - t[0])**2 + (p[1] - t[1])
                         ** 2 + (p[2] - t[2])**2)
    return dist_vox * dx_mm


def peak_loc_error_vox(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.array(argmax_3d(pred))
    t = np.array(argmax_3d(target))
    return float(np.linalg.norm(p - t))


def peak_rel_err(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pmax = float(np.max(pred))
    tmax = float(np.max(target))
    return abs(pmax - tmax) / (abs(tmax) + eps)


def peak_abs_err(pred: np.ndarray, target: np.ndarray) -> float:
    return float(abs(float(np.max(pred)) - float(np.max(target))))


def sphere_mask(shape: Tuple[int, int, int], center_zyx: Tuple[int, int, int], radius_vox: float):
    zc, yc, xc = center_zyx
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist2 = (zz - zc)**2 + (yy - yc)**2 + (xx - xc)**2
    return dist2 <= radius_vox**2


def mae_focus_sphere(pred: np.ndarray, target: np.ndarray, radius_mm: float, dx_mm: float) -> float:
    gt_peak = argmax_3d(target)
    radius_vox = float(radius_mm / dx_mm)
    mask = sphere_mask(target.shape, gt_peak, radius_vox)
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def pearsonr(a, b, eps=1e-8):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum()
    den = math.sqrt((a * a).sum() * (b * b).sum()) + eps
    return float(num / den)


def metric_lmae(pred, gt):
    return float(np.mean(np.abs(pred - gt)))


def metric_mse(pred, gt):
    return float(np.mean((pred - gt) ** 2))


def metric_lrrmse(pred, gt, eps=1e-8):
    num = np.sum((pred - gt) ** 2, dtype=np.float64)
    den = np.sum(gt ** 2, dtype=np.float64) + eps
    return float(np.sqrt(num / den))


def grad3d_np(x):
    g0 = x[1:, :, :] - x[:-1, :, :]
    g1 = x[:, 1:, :] - x[:, :-1, :]
    g2 = x[:, :, 1:] - x[:, :, :-1]
    return g0, g1, g2


def metric_gdl(pred, gt):
    p0, p1, p2 = grad3d_np(pred)
    g0, g1, g2 = grad3d_np(gt)
    d0 = np.mean((p0 - g0) ** 2)
    d1 = np.mean((p1 - g1) ** 2)
    d2 = np.mean((p2 - g2) ** 2)
    return float((d0 + d1 + d2) / 3.0)


def metric_ssim(pred, gt, clip_01=True, data_range=1.0):
    pred_use = pred.astype(np.float32).copy()
    gt_use = gt.astype(np.float32).copy()

    if clip_01:
        pred_use = np.clip(pred_use, 0.0, 1.0)
        gt_use = np.clip(gt_use, 0.0, 1.0)

    if HAS_TORCH_SSIM:
        try:
            p = torch.from_numpy(pred_use[None, None, ...])
            g = torch.from_numpy(gt_use[None, None, ...])
            val = torch_ssim(p, g, data_range=data_range, size_average=True)
            return float(val.item())
        except Exception:
            pass

    if HAS_SKIMAGE_SSIM:
        vals = []
        D = pred_use.shape[0]
        for i in range(D):
            vals.append(skimage_ssim(
                gt_use[i], pred_use[i], data_range=data_range))
        return float(np.mean(vals))

    return float("nan")


# =========================================================
# Métricas ROI / focus / brain
# =========================================================
def sphere_roi_mask(shape, center_xyz_m, dx_m, radius_m):
    Nx, Ny, Nz = shape
    center_vox = (np.asarray(center_xyz_m, dtype=np.float32) / np.float32(dx_m)) + \
                 (np.array([Nx, Ny, Nz], dtype=np.float32) / 2.0)
    cx, cy, cz = center_vox

    xi, yi, zi = np.ogrid[:Nx, :Ny, :Nz]
    r_vox = float(radius_m / dx_m)
    dist2 = (xi - cx) ** 2 + (yi - cy) ** 2 + (zi - cz) ** 2
    return dist2 <= (r_vox * r_vox)


def peak_loc_in_mask(vol, mask):
    masked = vol * mask
    idx = np.array(np.unravel_index(np.argmax(masked), masked.shape))
    return idx


# =========================================================
# Visualización
# =========================================================
def sample_line_trilinear(vol, p0, p1, n=256):
    vol = vol.astype(np.float32)
    pts = np.linspace(p0, p1, n, dtype=np.float32)
    Nx, Ny, Nz = vol.shape
    out = np.zeros((n,), dtype=np.float32)

    for i, (x, y, z) in enumerate(pts):
        x0 = int(np.clip(np.floor(x), 0, Nx - 1))
        y0 = int(np.clip(np.floor(y), 0, Ny - 1))
        z0 = int(np.clip(np.floor(z), 0, Nz - 1))
        x1 = min(x0 + 1, Nx - 1)
        y1 = min(y0 + 1, Ny - 1)
        z1 = min(z0 + 1, Nz - 1)

        xd = x - x0
        yd = y - y0
        zd = z - z0

        c000 = vol[x0, y0, z0]
        c100 = vol[x1, y0, z0]
        c010 = vol[x0, y1, z0]
        c110 = vol[x1, y1, z0]
        c001 = vol[x0, y0, z1]
        c101 = vol[x1, y0, z1]
        c011 = vol[x0, y1, z1]
        c111 = vol[x1, y1, z1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        out[i] = c0 * (1 - zd) + c1 * zd

    return out


def plot_triplet(vol, cx, cy, cz, title_prefix, path_png, vmin=None, vmax=None, cmap="jet"):
    sag = vol[cx, :, :].T
    cor = vol[:, cy, :].T
    axi = vol[:, :, cz].T

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, sl, ttl in zip(axes, [sag, cor, axi], ["sag", "cor", "ax"]):
        im = ax.imshow(sl, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title_prefix}-{ttl}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close(fig)


# =========================================================
# Resumen estadístico
# =========================================================
def summarize_metric(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p90": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


# =========================================================
# Score compuesto
# =========================================================
def minmax_normalize(vals: List[float]):
    finite = [v for v in vals if np.isfinite(v)]
    if len(finite) == 0:
        return [float("nan")] * len(vals)

    vmin = min(finite)
    vmax = max(finite)

    out = []
    for v in vals:
        if not np.isfinite(v):
            out.append(float("nan"))
        elif abs(vmax - vmin) < 1e-12:
            out.append(0.0)
        else:
            out.append((v - vmin) / (vmax - vmin))
    return out


def add_test_score(rows: List[Dict[str, Any]]):
    locs = [r["mean_peak_loc_error_mm"] for r in rows]
    peaks = [r["mean_peak_rel_err"] for r in rows]
    maes = [r["mean_mae_focus_sphere"] for r in rows]

    locs_n = minmax_normalize(locs)
    peaks_n = minmax_normalize(peaks)
    maes_n = minmax_normalize(maes)

    for r, a, b, c in zip(rows, locs_n, peaks_n, maes_n):
        if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
            r["test_clinical_score"] = (
                WEIGHT_PEAK_LOC_MM * a
                + WEIGHT_PEAK_REL * b
                + WEIGHT_MAE_FOCUS * c
            )
        else:
            r["test_clinical_score"] = float("nan")
    return rows


def add_rank_average(rows: List[Dict[str, Any]]):
    keys = [
        "mean_peak_loc_error_mm",
        "mean_peak_rel_err",
        "mean_mae_focus_sphere",
    ]
    for r in rows:
        r["rank_avg"] = 0.0

    for k in keys:
        ordered = sorted(rows, key=lambda x: x[k])
        for i, r in enumerate(ordered, start=1):
            r["rank_avg"] += i

    for r in rows:
        r["rank_avg"] /= len(keys)
    return rows


def sort_rows(rows: List[Dict[str, Any]]):
    return sorted(
        rows,
        key=lambda r: (
            r["test_clinical_score"],
            r["rank_avg"],
            r["mean_peak_loc_error_mm"],
            r["mean_peak_rel_err"],
            r["mean_mae_focus_sphere"],
        )
    )


# =========================================================
# CSV / JSON
# =========================================================
def save_csv(path: str, rows: List[Dict[str, Any]]):
    if len(rows) == 0:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# =========================================================
# Evaluación por checkpoint
# =========================================================
@torch.no_grad()
def evaluate_checkpoint_on_test(
    ckpt_path: str,
    test_files: List[str],
    device: str,
    dx_mm_default: float,
    focus_radius_mm: float,
    focus_loc_radius_mm: float,
):
    raw = load_ckpt_raw(ckpt_path, device="cpu")
    cfg = raw.get("config", {})
    stats = raw.get("stats", None)

    model = build_model_from_ckpt_config(cfg).to(device)
    model.load_state_dict(raw["model"])
    model.eval()

    y_denorm_mode, y_denorm_params = get_y_denorm_mode_and_params(stats)

    test_ds = TusDataset(test_files, stats=stats, normalize=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    per_case_rows = []

    metrics_store = {
        "peak_loc_error_mm": [],
        "peak_loc_error_vox": [],
        "peak_abs_err": [],
        "peak_rel_err": [],
        "mae_focus_sphere": [],
        "lmae": [],
        "mse": [],
        "rmse": [],
        "nrmse_max": [],
        "lrrmse": [],
        "gdl": [],
        "ssim": [],
        "pearson_r": [],
        "peak_loc_err_vox_brain": [],
        "peak_loc_err_vox_focusroi": [],
        "mae_signal_gt_thr": [],
    }

    for i, (X, y) in enumerate(test_loader):
        fpath = test_files[i]
        fname = os.path.basename(fpath)
        npz = np.load(fpath)

        dx_m = float(npz["dx"]) if "dx" in npz else dx_mm_default * 1e-3
        dx_mm_case = dx_m * 1e3

        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(USE_AMP and device == "cuda")):
            pred = model(X).float()

        pred = denormalize_y(pred, y_denorm_mode, y_denorm_params)
        y = denormalize_y(y, y_denorm_mode, y_denorm_params)

        pred = pred.clamp_min(0.0)
        y = y.clamp_min(0.0)

        pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)
        gt_np = y.squeeze().detach().cpu().numpy().astype(np.float32)

        # ----------------------------
        # Métricas base
        # ----------------------------
        loc_mm = peak_loc_error_mm(pred_np, gt_np, dx_mm=dx_mm_case)
        loc_vox = peak_loc_error_vox(pred_np, gt_np)
        p_abs = peak_abs_err(pred_np, gt_np)
        p_rel = peak_rel_err(pred_np, gt_np)
        mae_focus = mae_focus_sphere(
            pred_np, gt_np, radius_mm=focus_radius_mm, dx_mm=dx_mm_case)

        lmae = metric_lmae(pred_np, gt_np)
        mse = metric_mse(pred_np, gt_np)
        rmse = float(np.sqrt(mse))
        nrmse = float(rmse / (float(gt_np.max()) + 1e-8))
        lrrmse = metric_lrrmse(pred_np, gt_np)
        gdl = metric_gdl(pred_np, gt_np)
        ssim_val = metric_ssim(
            pred_np, gt_np, clip_01=SSIM_CLIP_01, data_range=SSIM_DATA_RANGE)
        r = pearsonr(pred_np, gt_np)

        # ----------------------------
        # Métricas ROI/focus/brain
        # ----------------------------
        brain_mask = npz["mask_brain"].astype(
            np.float32) if "mask_brain" in npz else None

        peak_loc_err_vox_brain = float("nan")
        if brain_mask is not None:
            idx_gt_b = peak_loc_in_mask(gt_np, brain_mask)
            idx_pr_b = peak_loc_in_mask(pred_np, brain_mask)
            peak_loc_err_vox_brain = float(np.linalg.norm(idx_pr_b - idx_gt_b))

        peak_loc_err_vox_focusroi = float("nan")
        if "focus_pos" in npz:
            focus_pos = npz["focus_pos"].astype(np.float32)
            roi_big = sphere_roi_mask(
                gt_np.shape,
                focus_pos,
                dx_m,
                radius_m=focus_loc_radius_mm * 1e-3
            )
            gt_roi = gt_np.copy()
            pr_roi = pred_np.copy()
            gt_roi[~roi_big] = -1e9
            pr_roi[~roi_big] = -1e9
            idx_gt_fr = np.array(np.unravel_index(
                np.argmax(gt_roi), gt_np.shape))
            idx_pr_fr = np.array(np.unravel_index(
                np.argmax(pr_roi), gt_np.shape))
            peak_loc_err_vox_focusroi = float(
                np.linalg.norm(idx_pr_fr - idx_gt_fr))

        sig_mask = (gt_np > SIGNAL_THR)
        mae_sig = float(np.mean(np.abs(
            pred_np[sig_mask] - gt_np[sig_mask]))) if sig_mask.any() else float("nan")

        # guardar
        row = {
            "checkpoint": os.path.basename(ckpt_path),
            "epoch": parse_epoch_from_name(ckpt_path),
            "case_idx": i,
            "file": fname,

            "peak_loc_error_mm": loc_mm,
            "peak_loc_error_vox": loc_vox,
            "peak_abs_err": p_abs,
            "peak_rel_err": p_rel,
            "mae_focus_sphere": mae_focus,

            "lmae": lmae,
            "mse": mse,
            "rmse": rmse,
            "nrmse_max": nrmse,
            "lrrmse": lrrmse,
            "gdl": gdl,
            "ssim": ssim_val,
            "pearson_r": r,

            "peak_loc_err_vox_brain": peak_loc_err_vox_brain,
            "peak_loc_err_vox_focusroi": peak_loc_err_vox_focusroi,
            "mae_signal_gt_thr": mae_sig,
        }
        per_case_rows.append(row)

        metrics_store["peak_loc_error_mm"].append(loc_mm)
        metrics_store["peak_loc_error_vox"].append(loc_vox)
        metrics_store["peak_abs_err"].append(p_abs)
        metrics_store["peak_rel_err"].append(p_rel)
        metrics_store["mae_focus_sphere"].append(mae_focus)
        metrics_store["lmae"].append(lmae)
        metrics_store["mse"].append(mse)
        metrics_store["rmse"].append(rmse)
        metrics_store["nrmse_max"].append(nrmse)
        metrics_store["lrrmse"].append(lrrmse)
        metrics_store["gdl"].append(gdl)
        metrics_store["ssim"].append(ssim_val)
        metrics_store["pearson_r"].append(r)
        metrics_store["peak_loc_err_vox_brain"].append(peak_loc_err_vox_brain)
        metrics_store["peak_loc_err_vox_focusroi"].append(
            peak_loc_err_vox_focusroi)
        metrics_store["mae_signal_gt_thr"].append(mae_sig)

    summary = {
        "checkpoint": os.path.basename(ckpt_path),
        "epoch": parse_epoch_from_name(ckpt_path),
        "n_samples": int(len(test_files)),
    }

    for k, vals in metrics_store.items():
        stats_k = summarize_metric(vals)
        summary[f"mean_{k}"] = stats_k["mean"]
        summary[f"median_{k}"] = stats_k["median"]
        summary[f"std_{k}"] = stats_k["std"]
        summary[f"p90_{k}"] = stats_k["p90"]

    return summary, per_case_rows


# =========================================================
# Visualización del mejor checkpoint
# =========================================================
@torch.no_grad()
def make_visualizations_for_best_ckpt(best_ckpt_path: str, test_files: List[str]):
    if not MAKE_VIS:
        return

    raw = load_ckpt_raw(best_ckpt_path, device="cpu")
    cfg = raw.get("config", {})
    stats = raw.get("stats", None)

    model = build_model_from_ckpt_config(cfg).to(DEVICE)
    model.load_state_dict(raw["model"])
    model.eval()

    y_denorm_mode, y_denorm_params = get_y_denorm_mode_and_params(stats)

    test_ds = TusDataset(test_files, stats=stats, normalize=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    vis_files = set()
    for f in test_files:
        if os.path.basename(f) in FORCE_VIS_FILES:
            vis_files.add(f)

    if VIS_RANDOM:
        tmp = test_files.copy()
        random.shuffle(tmp)
        for f in tmp[:min(N_VIS_RANDOM, len(tmp))]:
            vis_files.add(f)

    print(
        f"\nGenerando visualizaciones para {len(vis_files)} casos del mejor checkpoint...")

    for i, (X, y) in enumerate(test_loader):
        fpath = test_files[i]
        if fpath not in vis_files:
            continue

        npz = np.load(fpath)
        dx_m = float(npz["dx"]) if "dx" in npz else DX_MM * 1e-3

        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(USE_AMP and DEVICE == "cuda")):
            pred = model(X).float()

        pred = denormalize_y(pred, y_denorm_mode, y_denorm_params)
        y = denormalize_y(y, y_denorm_mode, y_denorm_params)

        pred = pred.clamp_min(0.0)
        y = y.clamp_min(0.0)

        pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)
        gt_np = y.squeeze().detach().cpu().numpy().astype(np.float32)

        idx_gt = np.array(argmax_3d(gt_np))
        cx, cy, cz = idx_gt.tolist()
        base = os.path.splitext(os.path.basename(fpath))[0]

        vmin = 0.0
        vmax = float(gt_np.max())

        plot_triplet(gt_np, cx, cy, cz, f"{base}-GT", os.path.join(
            FIG_DIR, f"{base}_GT.png"), vmin=vmin, vmax=vmax, cmap=CMAP)
        plot_triplet(pred_np, cx, cy, cz, f"{base}-Pred", os.path.join(
            FIG_DIR, f"{base}_Pred.png"), vmin=vmin, vmax=vmax, cmap=CMAP)
        plot_triplet(np.abs(pred_np - gt_np), cx, cy, cz, f"{base}-AbsErr", os.path.join(
            FIG_DIR, f"{base}_AbsErr.png"), vmin=0.0, vmax=None, cmap=CMAP)

        if "source_pos" in npz and "focus_pos" in npz:
            src_pos = npz["source_pos"].astype(np.float32)
            foc_pos = npz["focus_pos"].astype(np.float32)

            def m_to_vox(p_m):
                return (p_m / dx_m) + np.array(gt_np.shape, dtype=np.float32) / 2.0

            p0 = m_to_vox(src_pos)
            p1 = m_to_vox(foc_pos)

            prof_gt = sample_line_trilinear(gt_np, p0, p1, n=PROFILE_SAMPLES)
            prof_pr = sample_line_trilinear(pred_np, p0, p1, n=PROFILE_SAMPLES)

            fig = plt.figure(figsize=(7, 4))
            plt.plot(prof_gt, label="GT")
            plt.plot(prof_pr, label="Pred")
            plt.title(f"{base} | Line profile source→focus")
            plt.xlabel("samples along line")
            plt.ylabel("signal")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f"{base}_profile.png"), dpi=150)
            plt.close(fig)


# =========================================================
# MAIN
# =========================================================
def main():
    seed_all(123)

    ckpt_files = find_checkpoints(EVAL_CKPT_DIR, CKPT_NAMES)

    print("Checkpoints a evaluar:")
    for p in ckpt_files:
        print(" -", os.path.basename(p))

    split_params = get_split_params()
    data_dir = split_params["DATA_DIR"]
    seed = split_params["SEED"]
    train_ratio = split_params["TRAIN_RATIO"]
    val_ratio = split_params["VAL_RATIO"]

    print("\nSplit usado para TEST:")
    print(f"DATA_DIR = {data_dir}")
    print(f"SEED = {seed}")
    print(f"TRAIN_RATIO = {train_ratio}")
    print(f"VAL_RATIO = {val_ratio}")

    files = list_npz_files(data_dir)
    train_files, val_files, test_files = split_files(
        files,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    print(
        f"\nTotal sims: {len(files)} | train {len(train_files)} | val {len(val_files)} | test {len(test_files)}")

    summary_rows = []
    all_case_rows = []

    for i, ckpt_path in enumerate(ckpt_files, start=1):
        print(
            f"\n[{i}/{len(ckpt_files)}] Evaluando en TEST: {os.path.basename(ckpt_path)}")

        summary, case_rows = evaluate_checkpoint_on_test(
            ckpt_path=ckpt_path,
            test_files=test_files,
            device=DEVICE,
            dx_mm_default=DX_MM,
            focus_radius_mm=FOCUS_RADIUS_MM,
            focus_loc_radius_mm=FOCUS_LOC_RADIUS_MM,
        )

        summary_rows.append(summary)
        all_case_rows.extend(case_rows)

        print(
            f"  mean_peak_loc_error_mm = {summary['mean_peak_loc_error_mm']:.4f}\n"
            f"  mean_peak_rel_err      = {summary['mean_peak_rel_err']:.6f}\n"
            f"  mean_mae_focus_sphere  = {summary['mean_mae_focus_sphere']:.6f}\n"
            f"  mean_lrrmse            = {summary['mean_lrrmse']:.6f}\n"
            f"  mean_ssim              = {summary['mean_ssim']:.6f}\n"
            f"  mean_pearson_r         = {summary['mean_pearson_r']:.6f}"
        )

    summary_rows = add_test_score(summary_rows)
    summary_rows = add_rank_average(summary_rows)
    summary_rows = sort_rows(summary_rows)

    summary_csv = os.path.join(OUT_DIR, "final_test_comparison_summary.csv")
    per_case_csv = os.path.join(OUT_DIR, "final_test_comparison_per_case.csv")
    summary_json = os.path.join(OUT_DIR, "final_test_comparison_summary.json")

    save_csv(summary_csv, summary_rows)
    save_csv(per_case_csv, all_case_rows)
    save_json(summary_json, {
        "n_checkpoints": len(ckpt_files),
        "n_test_cases": len(test_files),
        "ranking": summary_rows,
    })

    print("\n" + "=" * 90)
    print("RANKING FINAL EN TEST")
    print("=" * 90)
    for rank, r in enumerate(summary_rows, start=1):
        print(
            f"{rank}) {r['checkpoint']} | "
            f"score={r['test_clinical_score']:.6f} | "
            f"rank_avg={r['rank_avg']:.3f} | "
            f"peak_loc_mm={r['mean_peak_loc_error_mm']:.4f} | "
            f"peak_rel={r['mean_peak_rel_err']:.6f} | "
            f"mae_focus={r['mean_mae_focus_sphere']:.6f} | "
            f"lrrmse={r['mean_lrrmse']:.6f} | "
            f"ssim={r['mean_ssim']:.6f}"
        )

    best_ckpt = os.path.join(EVAL_CKPT_DIR, summary_rows[0]["checkpoint"])

    print("\nCSV resumen guardado en:")
    print(summary_csv)

    print("\nCSV por caso guardado en:")
    print(per_case_csv)

    print("\nJSON resumen guardado en:")
    print(summary_json)

    print("\n" + "=" * 90)
    print("MEJOR POR CADA MÉTRICA")
    print("=" * 90)

    best_loc = min(summary_rows, key=lambda r: r["mean_peak_loc_error_mm"])
    best_peak = min(summary_rows, key=lambda r: r["mean_peak_rel_err"])
    best_mae = min(summary_rows, key=lambda r: r["mean_mae_focus_sphere"])
    best_lrrmse = min(summary_rows, key=lambda r: r["mean_lrrmse"])
    best_ssim = max(summary_rows, key=lambda r: r["mean_ssim"])
    best_r = max(summary_rows, key=lambda r: r["mean_pearson_r"])

    print(
        f"Mejor en peak_loc_error_mm: {best_loc['checkpoint']} -> {best_loc['mean_peak_loc_error_mm']:.4f}")
    print(
        f"Mejor en peak_rel_err:      {best_peak['checkpoint']} -> {best_peak['mean_peak_rel_err']:.6f}")
    print(
        f"Mejor en mae_focus_sphere:  {best_mae['checkpoint']} -> {best_mae['mean_mae_focus_sphere']:.6f}")
    print(
        f"Mejor en lrrmse:            {best_lrrmse['checkpoint']} -> {best_lrrmse['mean_lrrmse']:.6f}")
    print(
        f"Mejor en ssim:              {best_ssim['checkpoint']} -> {best_ssim['mean_ssim']:.6f}")
    print(
        f"Mejor en pearson_r:         {best_r['checkpoint']} -> {best_r['mean_pearson_r']:.6f}")

    if MAKE_VIS:
        make_visualizations_for_best_ckpt(best_ckpt, test_files)

    if not HAS_TORCH_SSIM and not HAS_SKIMAGE_SSIM:
        print("\n⚠️ SSIM no se calculó porque no está instalado pytorch-msssim ni scikit-image.")
        print("   Recomendado: pip install pytorch-msssim")


if __name__ == "__main__":
    main()
