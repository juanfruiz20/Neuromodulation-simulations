import os
import random
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from Data_loaderV2 import TusDataset
from Unet3D_V2 import ResUNet3D_HQ


# =========================================================
# CONFIG
# =========================================================

# Carpeta donde está el checkpoint
CKPT_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_V2"

# SOLO 1 checkpoint
CKPT_NAME = "best.pth"

# Carpeta YA lista con npz de test
NPZ_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

# Si quieres testear UN npz concreto, pon la ruta aquí.
# Si lo dejas en None, usará NPZ_DIR.
DIRECT_NPZ_PATH = None
# DIRECT_NPZ_PATH = r"C:\Users\USUARIO\Desktop\UIC Bioingenieria\TFG\dataset_TUS_SplitV1\test\sample_0233.npz"

SEED = 123

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True
NUM_WORKERS = 0
PIN_MEMORY = True

# Salida
OUT_DIR = os.path.join(CKPT_DIR, "visual_single_model_nocont")
os.makedirs(OUT_DIR, exist_ok=True)

# Visualización
CMAP = "jet"
SAVE_ABS_ERR = False
OVERLAY_CONTOURS = False
DRAW_CROSSHAIR = False

# ---------------------------------------------------------
# Selección de casos cuando NO usas DIRECT_NPZ_PATH
# ---------------------------------------------------------

# 1) Casos concretos por nombre
FORCE_CASE_FILES = [
    # "sample_0233.npz",
    # "sample_0101.npz",
]

# 2) Aleatorios
USE_RANDOM_CASES = True
N_RANDOM_CASES = 10

# 3) Primeros N
USE_FIRST_N_CASES = False
FIRST_N_CASES = 10


# =========================================================
# REPRODUCIBILIDAD
# =========================================================
def seed_all(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# HELPERS
# =========================================================
def load_ckpt_raw(path: str, device="cpu"):
    return torch.load(path, map_location=device)


def list_npz_in_dir(npz_dir: str) -> List[str]:
    files = sorted([
        os.path.join(npz_dir, f)
        for f in os.listdir(npz_dir)
        if f.endswith(".npz")
    ])
    if len(files) == 0:
        raise ValueError(f"No encontré .npz en: {npz_dir}")
    return files


def build_model_from_ckpt_config(cfg: Dict[str, Any]):
    return ResUNet3D_HQ(
        in_ch=cfg.get("IN_CH", 2),
        out_ch=cfg.get("OUT_CH", 1),
        base=cfg.get("BASE", 16),
        norm_kind=cfg.get("NORM_KIND", "group"),
        use_se=cfg.get("USE_SE", True),
        out_positive=cfg.get("OUT_POSITIVE", True),
    )


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


def argmax_3d(vol: np.ndarray) -> Tuple[int, int, int]:
    return np.unravel_index(np.argmax(vol), vol.shape)


def pretty_model_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    return base.replace("_", " ")


def get_slice_triplet(vol: np.ndarray, cx: int, cy: int, cz: int):
    """
    vol shape assumed: [D, H, W]
    """
    sag = vol[cx, :, :].T
    cor = vol[:, cy, :].T
    axi = vol[:, :, cz].T
    return sag, cor, axi


def get_case_selection(all_cases: List[str]) -> List[str]:
    selected = []

    force_set = set(FORCE_CASE_FILES)
    for f in all_cases:
        if os.path.basename(f) in force_set:
            selected.append(f)

    remaining = [f for f in all_cases if f not in selected]

    if len(selected) > 0:
        pass
    elif USE_RANDOM_CASES:
        random.shuffle(remaining)
        selected.extend(remaining[:min(N_RANDOM_CASES, len(remaining))])
    elif USE_FIRST_N_CASES:
        selected.extend(all_cases[:min(FIRST_N_CASES, len(all_cases))])
    else:
        selected = all_cases.copy()

    # quitar duplicados preservando orden
    out = []
    seen = set()
    for f in selected:
        if f not in seen:
            out.append(f)
            seen.add(f)

    return out


# =========================================================
# CARGA DEL MODELO
# =========================================================
def load_single_model(ckpt_dir: str, ckpt_name: str, device: str):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No existe checkpoint: {ckpt_path}")

    raw = load_ckpt_raw(ckpt_path, device="cpu")
    cfg = raw.get("config", {})
    stats = raw.get("stats", None)

    model = build_model_from_ckpt_config(cfg).to(device)
    model.load_state_dict(raw["model"])
    model.eval()

    y_denorm_mode, y_denorm_params = get_y_denorm_mode_and_params(stats)

    model_info = {
        "name": ckpt_name,
        "pretty_name": pretty_model_name(ckpt_name),
        "path": ckpt_path,
        "model": model,
        "stats": stats,
        "config": cfg,
        "denorm_mode": y_denorm_mode,
        "denorm_params": y_denorm_params,
    }
    return model_info


# =========================================================
# PREDICCIÓN DE UN CASO
# =========================================================
@torch.no_grad()
def predict_single_case(model_info: Dict[str, Any], case_file: str, device: str):
    with np.load(case_file) as d:
        src = d["source_mask"].astype(np.float32)
        skull = d["mask_skull"].astype(np.float32)
        y = d["p_max_norm"].astype(np.float32)

    expected_shape = (128, 128, 128)
    if src.shape != expected_shape or skull.shape != expected_shape or y.shape != expected_shape:
        raise RuntimeError(
            f"Shape inválida en {os.path.basename(case_file)} | "
            f"src:{src.shape} skull:{skull.shape} y:{y.shape} | "
            f"esperado:{expected_shape}"
        )

    # Entrada [1, 2, 128, 128, 128]
    X = np.stack([src, skull], axis=0)[np.newaxis, ...]
    # Target [1, 1, 128, 128, 128]
    y = y[np.newaxis, np.newaxis, ...]

    X = torch.from_numpy(X).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(USE_AMP and device == "cuda")
    ):
        pred = model_info["model"](X).float()

    pred = denormalize_y(
        pred, model_info["denorm_mode"], model_info["denorm_params"]
    )
    y = denormalize_y(
        y, model_info["denorm_mode"], model_info["denorm_params"]
    )

    pred = pred.clamp_min(0.0)
    y = y.clamp_min(0.0)

    pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)
    gt_np = y.squeeze().detach().cpu().numpy().astype(np.float32)

    return pred_np, gt_np


# =========================================================
# FIGURAS
# =========================================================
def save_comparison_figure(
    case_name: str,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    model_name: str,
    out_path: str,
    cmap: str = "jet",
    overlay_contours: bool = True,
    crosshair: bool = True,
):
    idx_gt = np.array(argmax_3d(gt_np))
    cx, cy, cz = idx_gt.tolist()

    vols = [gt_np, pred_np]
    row_titles = ["GT", pretty_model_name(model_name)]

    vmax = max(float(gt_np.max()), float(pred_np.max()))
    vmin = 0.0

    gt_level = 0.5 * float(gt_np.max())
    pred_level = 0.5 * float(pred_np.max())

    nrows = 2
    ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(17.5, 6.5))

    im = None
    col_titles = ["Sagittal", "Coronal", "Axial"]

    cross_coords = {
        "Sagittal": (cy, cz),
        "Coronal": (cx, cz),
        "Axial": (cx, cy),
    }

    gt_sag, gt_cor, gt_axi = get_slice_triplet(gt_np, cx, cy, cz)
    pred_sag, pred_cor, pred_axi = get_slice_triplet(pred_np, cx, cy, cz)

    gt_slices = [gt_sag, gt_cor, gt_axi]
    pred_slices = [pred_sag, pred_cor, pred_axi]

    for r, (row_title, slices) in enumerate(zip(row_titles, [gt_slices, pred_slices])):
        for c in range(ncols):
            ax = axes[r, c]
            im = ax.imshow(
                slices[c],
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )

            if r == 0:
                ax.set_title(col_titles[c], fontsize=12, pad=8)

            if c == 0:
                ax.set_ylabel(row_title, fontsize=11)

            if crosshair:
                x0, y0 = cross_coords[col_titles[c]]
                ax.plot(
                    x0, y0,
                    marker="+",
                    markersize=10,
                    markeredgewidth=1.6,
                    color="white"
                )

            if overlay_contours and r == 1:
                try:
                    ax.contour(gt_slices[c], levels=[gt_level], colors="white", linewidths=1.0)
                except Exception:
                    pass
                try:
                    ax.contour(pred_slices[c], levels=[pred_level], colors="red", linewidths=1.0)
                except Exception:
                    pass

            ax.axis("off")

    mae = float(np.mean(np.abs(pred_np - gt_np)))
    mse = float(np.mean((pred_np - gt_np) ** 2))

    fig.suptitle(
        f"{case_name} | GT vs {pretty_model_name(model_name)} | MAE={mae:.6f} | MSE={mse:.6f}",
        fontsize=15,
        y=0.995
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.90, 0.975])

    cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Amplitude", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_abs_error_figure(
    case_name: str,
    gt_np: np.ndarray,
    pred_np: np.ndarray,
    model_name: str,
    out_path: str,
    cmap: str = "jet",
    crosshair: bool = True,
):
    idx_gt = np.array(argmax_3d(gt_np))
    cx, cy, cz = idx_gt.tolist()

    err = np.abs(pred_np - gt_np)

    vmax = float(err.max())
    vmin = 0.0

    ncols = 3

    fig, axes = plt.subplots(1, ncols, figsize=(17.5, 3.8))

    col_titles = ["Sagittal", "Coronal", "Axial"]
    cross_coords = {
        "Sagittal": (cy, cz),
        "Coronal": (cx, cz),
        "Axial": (cx, cy),
    }

    sag, cor, axi = get_slice_triplet(err, cx, cy, cz)
    slices = [sag, cor, axi]

    im = None
    for c in range(ncols):
        ax = axes[c]
        im = ax.imshow(
            slices[c],
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        ax.set_title(col_titles[c], fontsize=12, pad=8)

        if crosshair:
            x0, y0 = cross_coords[col_titles[c]]
            ax.plot(
                x0, y0,
                marker="+",
                markersize=10,
                markeredgewidth=1.6,
                color="white"
            )

        ax.axis("off")

    mae = float(np.mean(np.abs(pred_np - gt_np)))
    mse = float(np.mean((pred_np - gt_np) ** 2))

    fig.suptitle(
        f"{case_name} | Absolute error | {pretty_model_name(model_name)} | MAE={mae:.6f} | MSE={mse:.6f}",
        fontsize=15,
        y=0.995
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.90, 0.93])

    cax = fig.add_axes([0.92, 0.18, 0.02, 0.62])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("|Pred - GT|", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================
def main():
    seed_all(SEED)

    print("Usando dispositivo:", DEVICE)

    # -----------------------------------------------------
    # Casos a procesar
    # -----------------------------------------------------
    if DIRECT_NPZ_PATH is not None:
        if not os.path.exists(DIRECT_NPZ_PATH):
            raise FileNotFoundError(f"No existe DIRECT_NPZ_PATH: {DIRECT_NPZ_PATH}")
        selected_cases = [DIRECT_NPZ_PATH]

        print("\nModo: NPZ directo")
        print("Caso seleccionado:")
        print(" -", os.path.basename(DIRECT_NPZ_PATH))

    else:
        all_cases = list_npz_in_dir(NPZ_DIR)
        selected_cases = get_case_selection(all_cases)

        if len(selected_cases) == 0:
            raise ValueError("No se seleccionó ningún caso para visualizar.")

        print("\nModo: carpeta de npz lista")
        print(f"Carpeta: {NPZ_DIR}")
        print("Casos seleccionados:")
        for f in selected_cases:
            print(" -", os.path.basename(f))

    # -----------------------------------------------------
    # Modelo
    # -----------------------------------------------------
    model_info = load_single_model(CKPT_DIR, CKPT_NAME, DEVICE)

    print("\nModelo cargado:")
    print(" -", model_info["name"])

    # -----------------------------------------------------
    # Procesamiento
    # -----------------------------------------------------
    for case_file in selected_cases:
        case_name = os.path.splitext(os.path.basename(case_file))[0]
        print(f"\nProcesando {case_name}...")

        pred_np, gt_np = predict_single_case(model_info, case_file, DEVICE)

        out_cmp = os.path.join(
            OUT_DIR,
            f"{case_name}_GT_vs_{os.path.splitext(CKPT_NAME)[0]}.png"
        )

        save_comparison_figure(
            case_name=case_name,
            gt_np=gt_np,
            pred_np=pred_np,
            model_name=model_info["name"],
            out_path=out_cmp,
            cmap=CMAP,
            overlay_contours=OVERLAY_CONTOURS,
            crosshair=DRAW_CROSSHAIR,
        )

        if SAVE_ABS_ERR:
            out_err = os.path.join(
                OUT_DIR,
                f"{case_name}_AbsErr_{os.path.splitext(CKPT_NAME)[0]}.png"
            )
            save_abs_error_figure(
                case_name=case_name,
                gt_np=gt_np,
                pred_np=pred_np,
                model_name=model_info["name"],
                out_path=out_err,
                cmap=CMAP,
                crosshair=DRAW_CROSSHAIR,
            )

    print("\nListo.")
    print("Figuras guardadas en:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()