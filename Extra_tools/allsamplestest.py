import os
import csv
import math
import glob
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_fill_holes

from src.modelos.ResUnet3D import ResUNet3D_HQ


# =========================================================
# CONFIG
# =========================================================

TEST_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

CKPT_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_TFG/epoch_220.pth"

OUT_CSV = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/test_per_sample_3metrics_dice90.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DX_MM = 1.0

DICE_THR = 90

NUM_WORKERS = 0
PIN_MEMORY = True if DEVICE == "cuda" else False


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
      - source_mask
      - mask_skull
      - p_max_norm
      - is_water_only o water_only
    """

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

            if "is_water_only" in d:
                is_water_only = bool(np.array(d["is_water_only"]).item())
            elif "water_only" in d:
                is_water_only = bool(np.array(d["water_only"]).item())
            else:
                raise RuntimeError(
                    f"{os.path.basename(path)} no contiene is_water_only ni water_only"
                )

        for name, arr in [
            ("source_mask", src),
            ("mask_skull", skull),
            ("p_max_norm", y),
        ]:
            if arr.shape != self.expected_shape:
                raise RuntimeError(
                    f"Shape inválida en {os.path.basename(path)} | "
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
# MODEL LOADING
# =========================================================

def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    model = ResUNet3D_HQ(
        in_ch=2,
        out_ch=1,
        base=16,
        norm_kind="group",
        use_se=True,
        out_positive=True,
    ).to(device)

    if isinstance(ckpt, dict) and "G" in ckpt:
        print("Cargando generador G desde checkpoint cGAN")
        model.load_state_dict(ckpt["G"], strict=True)

    elif isinstance(ckpt, dict) and "model" in ckpt:
        print("Cargando modelo desde checkpoint U-Net")
        model.load_state_dict(ckpt["model"], strict=True)

    else:
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)
        raise KeyError(
            f"No se encontró ni 'G' ni 'model' en el checkpoint: {ckpt_path}. "
            f"Keys disponibles: {keys}"
        )

    model.eval()
    return model, ckpt


# =========================================================
# METRIC HELPERS
# =========================================================

def get_peak_index_masked(vol: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    masked = np.where(mask > 0, vol, -np.inf)
    flat_idx = int(np.argmax(masked))
    return np.unravel_index(flat_idx, vol.shape)


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-8) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    inter = np.logical_and(mask_a, mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()

    return float((2.0 * inter) / (denom + eps))


def compute_metrics_one_sample(
    pred: np.ndarray,
    gt: np.ndarray,
    brain_mask: np.ndarray,
    dx_mm: float,
    is_water_only: bool,
) -> Dict[str, float]:

    pred = np.clip(pred.astype(np.float32), 0.0, None)
    gt = gt.astype(np.float32)
    brain_mask_bool = brain_mask > 0.5

    out = {
        "is_water_only": int(is_water_only),
    }

    # =====================================================
    # Caso water-only o sin brain mask
    # =====================================================

    if is_water_only or brain_mask_bool.sum() == 0:
        out.update({
            "peak_abs_err_brain": float("nan"),

            # Escala 0-1
            "peak_rel_err_brain": float("nan"),

            # Porcentaje
            "peak_rel_err_brain_percent": float("nan"),

            "peak_loc_err_vox_brain": float("nan"),
            "peak_loc_err_mm_brain": float("nan"),

            # Dice 90
            "dice_focus_brain_thr90": float("nan"),
            "dice_focus_brain_thr90_percent": float("nan"),

            # Diagnóstico Dice 90
            "gt_focus_voxels_thr90": 0,
            "pred_focus_voxels_thr90": 0,
            "intersection_voxels_thr90": 0,

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

    # =====================================================
    # Peak metrics dentro del brain mask
    # =====================================================

    gt_peak_idx = get_peak_index_masked(gt, brain_mask_bool)
    pred_peak_idx = get_peak_index_masked(pred, brain_mask_bool)

    gt_peak_val_brain = float(gt[gt_peak_idx])
    pred_peak_val_brain = float(pred[pred_peak_idx])

    peak_abs_err_brain = abs(pred_peak_val_brain - gt_peak_val_brain)

    # Escala 0-1
    peak_rel_err_brain = peak_abs_err_brain / (abs(gt_peak_val_brain) + 1e-8)

    # Porcentaje
    peak_rel_err_brain_percent = 100.0 * peak_rel_err_brain

    peak_loc_err_vox_brain = math.sqrt(
        (pred_peak_idx[0] - gt_peak_idx[0]) ** 2 +
        (pred_peak_idx[1] - gt_peak_idx[1]) ** 2 +
        (pred_peak_idx[2] - gt_peak_idx[2]) ** 2
    )

    peak_loc_err_mm_brain = peak_loc_err_vox_brain * dx_mm

    # =====================================================
    # Dice 90
    # =====================================================

    frac = DICE_THR / 100.0

    gt_thr = frac * gt_peak_val_brain
    pr_thr = frac * pred_peak_val_brain

    gt_focus = np.logical_and(gt >= gt_thr, brain_mask_bool)
    pr_focus = np.logical_and(pred >= pr_thr, brain_mask_bool)

    dice90_decimal = dice_score(gt_focus, pr_focus)
    dice90_percent = 100.0 * dice90_decimal

    gt_focus_voxels_thr90 = int(gt_focus.sum())
    pred_focus_voxels_thr90 = int(pr_focus.sum())
    intersection_voxels_thr90 = int(np.logical_and(gt_focus, pr_focus).sum())

    out.update({
        "peak_abs_err_brain": peak_abs_err_brain,

        # Escala 0-1
        "peak_rel_err_brain": peak_rel_err_brain,

        # Porcentaje
        "peak_rel_err_brain_percent": peak_rel_err_brain_percent,

        "peak_loc_err_vox_brain": float(peak_loc_err_vox_brain),
        "peak_loc_err_mm_brain": float(peak_loc_err_mm_brain),

        # Dice 90 escala 0-1
        "dice_focus_brain_thr90": dice90_decimal,

        # Dice 90 porcentaje
        "dice_focus_brain_thr90_percent": dice90_percent,

        # Diagnóstico Dice 90
        "gt_focus_voxels_thr90": gt_focus_voxels_thr90,
        "pred_focus_voxels_thr90": pred_focus_voxels_thr90,
        "intersection_voxels_thr90": intersection_voxels_thr90,

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


# =========================================================
# CSV SAVE
# =========================================================

def save_csv(csv_path: str, rows):
    if len(rows) == 0:
        raise RuntimeError("No hay filas para guardar.")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# EVALUATION
# =========================================================

@torch.no_grad()
def evaluate_test_set():
    print("DEVICE:", DEVICE)
    print("TEST_DIR:", TEST_DIR)
    print("CKPT_PATH:", CKPT_PATH)
    print("OUT_CSV:", OUT_CSV)

    dataset = TusTestDataset(TEST_DIR)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model, ckpt = load_model(CKPT_PATH, DEVICE)

    rows = []

    for i, batch in enumerate(loader, start=1):
        X = batch["X"].to(DEVICE, non_blocking=True)
        y = batch["y"].to(DEVICE, non_blocking=True)
        brain = batch["brain_mask"].to(DEVICE, non_blocking=True)

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

        row = {
            "sample_index": i,
            "file": file_name,
            "target": "Overall",
            "dx_mm": DX_MM,
        }

        row.update(metrics)
        rows.append(row)

        print(
            f"[{i}/{len(dataset)}] {file_name} | "
            f"Peak rel err: {metrics['peak_rel_err_brain']:.4f} "
            f"({metrics['peak_rel_err_brain_percent']:.2f}%) | "
            f"Loc err: {metrics['peak_loc_err_mm_brain']:.2f} mm | "
            f"Dice90: {metrics['dice_focus_brain_thr90']:.4f} "
            f"({metrics['dice_focus_brain_thr90_percent']:.2f}%) | "
            f"A90 GT vox: {metrics['gt_focus_voxels_thr90']} | "
            f"A90 Pred vox: {metrics['pred_focus_voxels_thr90']} | "
            f"A90 Inter: {metrics['intersection_voxels_thr90']}"
        )

    save_csv(OUT_CSV, rows)

    print("\nCSV guardado en:")
    print(OUT_CSV)

    # =====================================================
    # Summary en porcentaje para la gráfica
    # =====================================================

    peak_percent = np.array(
        [r["peak_rel_err_brain_percent"] for r in rows],
        dtype=np.float64
    )

    dice90_percent = np.array(
        [r["dice_focus_brain_thr90_percent"] for r in rows],
        dtype=np.float64
    )

    loc_mm = np.array(
        [r["peak_loc_err_mm_brain"] for r in rows],
        dtype=np.float64
    )

    print("\n================ OVERALL SUMMARY ================")
    print(f"Samples: {len(rows)}")

    print(f"Mean peak relative error [%]: {np.nanmean(peak_percent):.4f}")
    print(f"Median peak relative error [%]: {np.nanmedian(peak_percent):.4f}")

    print(f"Mean Dice A90 [%]: {np.nanmean(dice90_percent):.4f}")
    print(f"Median Dice A90 [%]: {np.nanmedian(dice90_percent):.4f}")

    print(f"Mean peak location error [mm]: {np.nanmean(loc_mm):.4f}")
    print(f"Median peak location error [mm]: {np.nanmedian(loc_mm):.4f}")
    print("=================================================")

    print("\nColumnas principales para la gráfica:")
    print("  peak_rel_err_brain_percent")
    print("  dice_focus_brain_thr90_percent")
    print("  peak_loc_err_mm_brain")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    evaluate_test_set()