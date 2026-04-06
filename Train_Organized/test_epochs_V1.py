import os
import csv
import math
import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
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
TEST_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"
CKPT_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_V2_expB"
OUT_DIR = os.path.join(CKPT_DIR, "test_metrics_brain_v2")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Si el dataset se gener� con dx=1 mm, d�jalo en 1.0
DX_MM = 1.0

NUM_WORKERS = 0
PIN_MEMORY = True

os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# AUTO-DISCOVER DE CHECKPOINTS
# =========================================================
def discover_checkpoints(ckpt_dir: str) -> List[str]:
    names = []

    for fixed_name in ["best.pth", "last.pth"]:
        p = os.path.join(ckpt_dir, fixed_name)
        if os.path.exists(p):
            names.append(fixed_name)

    epoch_paths = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pth")))
    names.extend([os.path.basename(p) for p in epoch_paths])

    out = []
    seen = set()
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


CKPT_NAMES = discover_checkpoints(CKPT_DIR)


# =========================================================
# BRAIN MASK RECONSTRUCTION
# =========================================================
def reconstruct_brain_mask(mask_skull: np.ndarray, is_water_only: bool) -> np.ndarray:
    skull = mask_skull > 0.5

    if is_water_only or skull.sum() == 0:
        return np.zeros_like(mask_skull, dtype=np.float32)

    filled = binary_fill_holes(skull)
    brain = np.logical_and(filled, np.logical_not(skull))
    return brain.astype(np.float32)


# =========================================================
# DATASET
# =========================================================
class TusTestDataset(Dataset):
    """
    Espera en cada .npz:
      - is_water_only
      - mask_skull
      - p_max_norm
      - source_mask

    Reconstruye brain_mask desde mask_skull.
    """

    def __init__(self, data_dir: str, expected_shape=(128, 128, 128)):
        super().__init__()
        self.data_dir = data_dir
        self.expected_shape = tuple(expected_shape)
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if len(self.files) == 0:
            raise RuntimeError(
                f"No se encontraron archivos .npz en: {data_dir}")

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
# HELPERS
# =========================================================
def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Mantiene tu arquitectura importada desde Model_Unet3D
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
    return model, ckpt


def get_peak_index_masked(vol: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    masked = np.where(mask > 0, vol, -np.inf)
    flat_idx = int(np.argmax(masked))
    return np.unravel_index(flat_idx, vol.shape)


def crop_to_mask_bbox(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return vol
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    return vol[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]


def compute_ssim_safe(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    if not HAS_SKIMAGE:
        return float("nan")

    try:
        return float(skimage_ssim(a, b, data_range=data_range))
    except Exception:
        pass

    vals = []
    for z in range(a.shape[0]):
        az = a[z]
        bz = b[z]

        if min(az.shape) < 7:
            continue

        try:
            vals.append(float(skimage_ssim(az, bz, data_range=data_range)))
        except Exception:
            continue

    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-8) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    inter = np.logical_and(mask_a, mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()
    return float((2.0 * inter) / (denom + eps))


# =========================================================
# METRICS
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

    # -----------------------------
    # Global metrics
    # -----------------------------
    mse_global = float(np.mean((pred - gt) ** 2))
    mae_global = float(np.mean(np.abs(pred - gt)))
    ssim_global = compute_ssim_safe(pred, gt, data_range=1.0)

    out = {
        "mse_global": mse_global,
        "mae_global": mae_global,
        "ssim_global": ssim_global,
        "is_water_only": int(is_water_only),
    }

    # -----------------------------
    # Brain-only metrics
    # -----------------------------
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
            "dice_focus_brain_thr70": float("nan"),
            "gt_peak_val_brain": float("nan"),
            "pred_peak_val_brain": float("nan"),
            "gt_peak_z": -1,
            "gt_peak_y": -1,
            "gt_peak_x": -1,
            "pred_peak_z": -1,
            "pred_peak_y": -1,
            "pred_peak_x": -1,
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

    ssim_brain = compute_ssim_safe(
        pred_brain_crop, gt_brain_crop, data_range=1.0)

    # -----------------------------
    # Peak in brain
    # -----------------------------
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

    # -----------------------------
    # Dice focus
    # -----------------------------
    gt_thr50 = 0.50 * gt_peak_val_brain
    pr_thr50 = 0.50 * pred_peak_val_brain

    gt_thr70 = 0.70 * gt_peak_val_brain
    pr_thr70 = 0.70 * pred_peak_val_brain

    gt_focus_thr50 = np.logical_and(gt >= gt_thr50, brain_mask)
    pr_focus_thr50 = np.logical_and(pred >= pr_thr50, brain_mask)

    gt_focus_thr70 = np.logical_and(gt >= gt_thr70, brain_mask)
    pr_focus_thr70 = np.logical_and(pred >= pr_thr70, brain_mask)

    dice_focus_brain_thr50 = dice_score(gt_focus_thr50, pr_focus_thr50)
    dice_focus_brain_thr70 = dice_score(gt_focus_thr70, pr_focus_thr70)

    out.update({
        "mse_brain": mse_brain,
        "mae_brain": mae_brain,
        "ssim_brain": ssim_brain,
        "peak_abs_err_brain": peak_abs_err_brain,
        "peak_rel_err_brain": peak_rel_err_brain,
        "peak_loc_err_vox_brain": float(peak_loc_err_vox_brain),
        "peak_loc_err_mm_brain": float(peak_loc_err_mm_brain),
        "dice_focus_brain_thr50": dice_focus_brain_thr50,
        "dice_focus_brain_thr70": dice_focus_brain_thr70,
        "gt_peak_val_brain": gt_peak_val_brain,
        "pred_peak_val_brain": pred_peak_val_brain,
        "gt_peak_z": int(gt_peak_idx[0]),
        "gt_peak_y": int(gt_peak_idx[1]),
        "gt_peak_x": int(gt_peak_idx[2]),
        "pred_peak_z": int(pred_peak_idx[0]),
        "pred_peak_y": int(pred_peak_idx[1]),
        "pred_peak_x": int(pred_peak_idx[2]),
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
        "dice_focus_brain_thr70",
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
            out[f"median_{k}"] = float("nan")
            out[f"std_{k}"] = float("nan")
            out[f"p90_{k}"] = float("nan")
        else:
            out[f"mean_{k}"] = float(np.mean(vals))
            out[f"median_{k}"] = float(np.median(vals))
            out[f"std_{k}"] = float(np.std(vals))
            out[f"p90_{k}"] = float(np.percentile(vals, 90))

    return out


def save_per_sample_csv(csv_path: str, rows: List[Dict[str, float]]):
    if len(rows) == 0:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(csv_path: str, summary: Dict[str, float]):
    fieldnames = list(summary.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


def rankdata_desc(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(-arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


def rankdata_asc(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, loader: DataLoader, device: str):
    model, ckpt = load_model(ckpt_path, device)
    all_rows = []

    for batch in loader:
        X = batch["X"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        brain = batch["brain_mask"].to(device, non_blocking=True)
        is_water_only = bool(batch["is_water_only"][0])
        file_name = batch["file_name"][0]

        pred = model(X)
        pred = pred.clamp_min(0.0)

        pred_np = pred[0, 0].detach().cpu().numpy()
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
        all_rows.append(metrics)

    return all_rows, ckpt


def build_ranking(ranking_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if len(ranking_rows) == 0:
        return ranking_rows

    peak_loc = [r["mean_peak_loc_err_mm_brain"] for r in ranking_rows]
    peak_rel = [r["mean_peak_rel_err_brain"] for r in ranking_rows]
    mae_brain = [r["mean_mae_brain"] for r in ranking_rows]
    ssim_brain = [r["mean_ssim_brain"] for r in ranking_rows]
    dice50 = [r["mean_dice_focus_brain_thr50"] for r in ranking_rows]
    dice70 = [r["mean_dice_focus_brain_thr70"] for r in ranking_rows]

    r_peak_loc = rankdata_asc(peak_loc)
    r_peak_rel = rankdata_asc(peak_rel)
    r_mae_brain = rankdata_asc(mae_brain)
    r_ssim_brain = rankdata_desc(ssim_brain)
    r_dice50 = rankdata_desc(dice50)
    r_dice70 = rankdata_desc(dice70)

    composite = (
        0.35 * r_peak_loc +
        0.25 * r_peak_rel +
        0.15 * r_mae_brain +
        0.10 * r_ssim_brain +
        0.10 * r_dice50 +
        0.05 * r_dice70
    )

    for i, row in enumerate(ranking_rows):
        row["rank_peak_loc"] = float(r_peak_loc[i])
        row["rank_peak_rel"] = float(r_peak_rel[i])
        row["rank_mae_brain"] = float(r_mae_brain[i])
        row["rank_ssim_brain"] = float(r_ssim_brain[i])
        row["rank_dice50"] = float(r_dice50[i])
        row["rank_dice70"] = float(r_dice70[i])
        row["composite_score"] = float(composite[i])

    ranking_rows = sorted(ranking_rows, key=lambda x: x["composite_score"])
    return ranking_rows


def main():
    print("Device:", DEVICE)
    print("TEST_DIR:", TEST_DIR)
    print("CKPT_DIR:", CKPT_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("DX_MM:", DX_MM)
    print("Checkpoints found:", CKPT_NAMES)

    if len(CKPT_NAMES) == 0:
        print("No se encontraron checkpoints.")
        return

    test_ds = TusTestDataset(TEST_DIR)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    ranking_rows = []

    for ckpt_name in CKPT_NAMES:
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"Checkpoint no encontrado, se omite: {ckpt_path}")
            continue

        print(f"\nEvaluando {ckpt_name} ...")
        rows, ckpt = evaluate_checkpoint(ckpt_path, test_loader, DEVICE)

        per_sample_csv = os.path.join(
            OUT_DIR, f"{os.path.splitext(ckpt_name)[0]}_per_sample.csv"
        )
        save_per_sample_csv(per_sample_csv, rows)

        summary = summarize_metrics(rows)
        summary["checkpoint"] = ckpt_name
        summary["epoch"] = int(ckpt.get("epoch", -1))

        summary_csv = os.path.join(
            OUT_DIR, f"{os.path.splitext(ckpt_name)[0]}_summary.csv"
        )
        save_summary_csv(summary_csv, summary)

        ranking_rows.append({
            "checkpoint": ckpt_name,
            "epoch": int(ckpt.get("epoch", -1)),
            "n_samples": int(summary["n_samples"]),
            "n_water_only": int(summary["n_water_only"]),
            "n_non_water_only": int(summary["n_non_water_only"]),
            "mean_mse_global": summary["mean_mse_global"],
            "mean_mae_global": summary["mean_mae_global"],
            "mean_ssim_global": summary["mean_ssim_global"],
            "mean_mse_brain": summary["mean_mse_brain"],
            "mean_mae_brain": summary["mean_mae_brain"],
            "mean_ssim_brain": summary["mean_ssim_brain"],
            "mean_peak_abs_err_brain": summary["mean_peak_abs_err_brain"],
            "mean_peak_rel_err_brain": summary["mean_peak_rel_err_brain"],
            "mean_peak_loc_err_mm_brain": summary["mean_peak_loc_err_mm_brain"],
            "mean_dice_focus_brain_thr50": summary["mean_dice_focus_brain_thr50"],
            "mean_dice_focus_brain_thr70": summary["mean_dice_focus_brain_thr70"],
        })

        print(
            f"{ckpt_name} | "
            f"peak_loc_mm_brain={summary['mean_peak_loc_err_mm_brain']:.4f} | "
            f"peak_rel_brain={summary['mean_peak_rel_err_brain']:.4f} | "
            f"mae_brain={summary['mean_mae_brain']:.4f} | "
            f"ssim_brain={summary['mean_ssim_brain']:.4f} | "
            f"dice50={summary['mean_dice_focus_brain_thr50']:.4f} | "
            f"dice70={summary['mean_dice_focus_brain_thr70']:.4f}"
        )

    if len(ranking_rows) == 0:
        print("\nNo se evalu� ning�n checkpoint.")
        return

    ranking_rows = build_ranking(ranking_rows)

    ranking_csv = os.path.join(OUT_DIR, "checkpoint_ranking.csv")
    with open(ranking_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranking_rows)

    print(f"\nRanking guardado en: {ranking_csv}")
    print("\nTop checkpoints:")
    for row in ranking_rows[:5]:
        print(
            f"  {row['checkpoint']} | epoch={row['epoch']} | "
            f"score={row['composite_score']:.3f} | "
            f"peak_loc_mm={row['mean_peak_loc_err_mm_brain']:.4f} | "
            f"peak_rel={row['mean_peak_rel_err_brain']:.4f} | "
            f"mae_brain={row['mean_mae_brain']:.4f}"
        )


if __name__ == "__main__":
    main()
