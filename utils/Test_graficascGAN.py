import os
import glob
import random
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # �til en servidores sin interfaz gr�fica

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_fill_holes

from src.modelos.ResUnet3D import ResUNet3D_HQ


# =========================================================
# CONFIG
# =========================================================
TEST_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

# Cambia entre best.pth, last.pth o epoch_xxx.pth
CKPT_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_exp04_300epoch/epoch_290.pth"

OUT_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/tablas/visualscGAN300"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DX_MM = 1.0

NUM_WORKERS = 0
PIN_MEMORY = True if DEVICE == "cuda" else False

# Casos a visualizar
N_RANDOM_CASES = 5
FIXED_RANDOM_SEED = 42

# Si quieres forzar casos concretos, ponlos aqu�.
# Si est� vac�o, el script escoge siempre los mismos 5 no-water usando seed.
FORCE_FILES = set()
# Ejemplo:
# FORCE_FILES = {"sample_0012.npz", "sample_0456.npz"}

# Scatter
SCATTER_THR = 0.01
SCATTER_MAX_POINTS = 50000

# Perfil aproximado source-centroid -> GT peak
PROFILE_SAMPLES = 256

os.makedirs(OUT_DIR, exist_ok=True)


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
      - is_water_only o water_only
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
                    f"Shape inv�lida en {os.path.basename(path)} | "
                    f"{name}: {arr.shape} | esperado: {self.expected_shape}"
                )

        brain = reconstruct_brain_mask(skull, is_water_only)

        # Entrada de tu cGAN: [source_mask, skull_mask]
        X = np.stack([src, skull], axis=0)   # [2, D, H, W]
        y = y[np.newaxis, ...]               # [1, D, H, W]
        brain = brain[np.newaxis, ...]       # [1, D, H, W]
        src = src[np.newaxis, ...]           # [1, D, H, W]

        return {
            "X": torch.from_numpy(X),
            "y": torch.from_numpy(y),
            "brain_mask": torch.from_numpy(brain),
            "source_mask": torch.from_numpy(src),
            "is_water_only": is_water_only,
            "file_name": os.path.basename(path),
        }


# =========================================================
# FIXED NON-WATER CASE SELECTION
# =========================================================
def read_is_water_only(path: str) -> bool:
    with np.load(path) as d:
        if "is_water_only" in d:
            return bool(np.array(d["is_water_only"]).item())
        elif "water_only" in d:
            return bool(np.array(d["water_only"]).item())
        else:
            raise RuntimeError(
                f"{os.path.basename(path)} no contiene is_water_only ni water_only"
            )


def select_fixed_random_non_water_cases(dataset, n_cases=5, seed=42):
    """
    Selecciona siempre los mismos n_cases no-water de forma pseudoaleatoria.
    Mientras no cambie la carpeta test, los 5 casos ser�n siempre los mismos.
    """
    valid_files = []

    for path in dataset.files:
        is_water_only = read_is_water_only(path)

        if not is_water_only:
            valid_files.append(os.path.basename(path))

    if len(valid_files) < n_cases:
        raise RuntimeError(
            f"Solo se encontraron {len(valid_files)} casos no-water, "
            f"pero pediste {n_cases}."
        )

    rng = random.Random(seed)
    selected = rng.sample(valid_files, n_cases)

    return set(selected)


def validate_force_files(dataset, force_files):
    """
    Comprueba que los FORCE_FILES existan y que no sean water-only.
    """
    if not force_files:
        return

    available = {os.path.basename(p): p for p in dataset.files}

    for fname in force_files:
        if fname not in available:
            raise RuntimeError(
                f"FORCE_FILE no encontrado en TEST_DIR: {fname}"
            )

        is_water_only = read_is_water_only(available[fname])
        if is_water_only:
            raise RuntimeError(
                f"FORCE_FILE es water-only y no deber�a usarse: {fname}"
            )


# =========================================================
# MODEL / CGAN GENERATOR
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

    # Compatible con checkpoints de cGAN y U-Net.
    # Para cGAN se usa SOLO el Generator G.
    if isinstance(ckpt, dict) and "G" in ckpt:
        print("Cargando GENERATOR G desde checkpoint cGAN")
        model.load_state_dict(ckpt["G"], strict=True)

    elif isinstance(ckpt, dict) and "model" in ckpt:
        print("Cargando modelo desde checkpoint U-Net")
        model.load_state_dict(ckpt["model"], strict=True)

    else:
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())
        else:
            keys = type(ckpt)

        raise KeyError(
            f"No se encontr� ni 'G' ni 'model' en el checkpoint: {ckpt_path}. "
            f"Keys disponibles: {keys}"
        )

    model.eval()
    return model


# =========================================================
# HELPERS
# =========================================================
def get_peak_index_masked(vol: np.ndarray, mask: np.ndarray):
    masked = np.where(mask > 0, vol, -np.inf)
    flat_idx = int(np.argmax(masked))
    return np.unravel_index(flat_idx, vol.shape)


def get_source_centroid(source_mask: np.ndarray):
    coords = np.argwhere(source_mask > 0.5)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0).astype(np.float32)  # z, y, x


def sample_line_trilinear(vol, p0, p1, n=256):
    """
    vol shape [Z, Y, X]
    p0, p1 en coords voxel flotantes [z, y, x]
    """
    vol = vol.astype(np.float32)
    pts = np.linspace(p0, p1, n, dtype=np.float32)
    Nz, Ny, Nx = vol.shape
    out = np.zeros((n,), dtype=np.float32)

    for i, (z, y, x) in enumerate(pts):
        z0 = int(np.clip(np.floor(z), 0, Nz - 1))
        y0 = int(np.clip(np.floor(y), 0, Ny - 1))
        x0 = int(np.clip(np.floor(x), 0, Nx - 1))

        z1 = min(z0 + 1, Nz - 1)
        y1 = min(y0 + 1, Ny - 1)
        x1 = min(x0 + 1, Nx - 1)

        zd = z - z0
        yd = y - y0
        xd = x - x0

        c000 = vol[z0, y0, x0]
        c001 = vol[z0, y0, x1]
        c010 = vol[z0, y1, x0]
        c011 = vol[z0, y1, x1]
        c100 = vol[z1, y0, x0]
        c101 = vol[z1, y0, x1]
        c110 = vol[z1, y1, x0]
        c111 = vol[z1, y1, x1]

        c00 = c000 * (1 - xd) + c001 * xd
        c01 = c010 * (1 - xd) + c011 * xd
        c10 = c100 * (1 - xd) + c101 * xd
        c11 = c110 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd

        out[i] = c0 * (1 - zd) + c1 * zd

    return out


# =========================================================
# PLOTS
# =========================================================
def plot_scatter_gt_vs_pred(gt, pred, brain_mask, save_path, thr=0.01, max_points=50000):
    mask = (brain_mask > 0.5) & (gt > thr)

    gt_vals = gt[mask]
    pred_vals = pred[mask]

    if len(gt_vals) == 0:
        print(f"[WARN] Scatter vac�o, no se guarda: {save_path}")
        return

    if len(gt_vals) > max_points:
        idx = np.random.choice(len(gt_vals), max_points, replace=False)
        gt_vals = gt_vals[idx]
        pred_vals = pred_vals[idx]

    plt.figure(figsize=(5.4, 5.0))
    plt.scatter(gt_vals, pred_vals, s=3, alpha=0.28)

    min_val = min(float(gt_vals.min()), float(pred_vals.min()))
    max_val = max(float(gt_vals.max()), float(pred_vals.max()))

    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)

    plt.xlabel("Ground truth")
    plt.ylabel("Prediction")
    plt.title(f"GT vs Pred (brain, GT > {thr})")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_axis_profiles(gt, pred, brain_mask, save_path, dx_mm=1.0):
    z, y, x = get_peak_index_masked(gt, brain_mask)

    prof_x_gt = gt[z, y, :]
    prof_x_pr = pred[z, y, :]

    prof_y_gt = gt[z, :, x]
    prof_y_pr = pred[z, :, x]

    prof_z_gt = gt[:, y, x]
    prof_z_pr = pred[:, y, x]

    x_axis = np.arange(gt.shape[2]) * dx_mm
    y_axis = np.arange(gt.shape[1]) * dx_mm
    z_axis = np.arange(gt.shape[0]) * dx_mm

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(x_axis, prof_x_gt, label="GT")
    axes[0].plot(x_axis, prof_x_pr, label="Pred")
    axes[0].set_title(f"X profile | z={z}, y={y}")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("p_max_norm")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(y_axis, prof_y_gt, label="GT")
    axes[1].plot(y_axis, prof_y_pr, label="Pred")
    axes[1].set_title(f"Y profile | z={z}, x={x}")
    axes[1].set_xlabel("y (mm)")
    axes[1].set_ylabel("p_max_norm")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(z_axis, prof_z_gt, label="GT")
    axes[2].plot(z_axis, prof_z_pr, label="Pred")
    axes[2].set_title(f"Z profile | y={y}, x={x}")
    axes[2].set_xlabel("z (mm)")
    axes[2].set_ylabel("p_max_norm")
    axes[2].grid(True, alpha=0.25)

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_source_centroid_to_peak_profile(
    gt,
    pred,
    source_mask,
    brain_mask,
    save_path,
    n_samples=256
):
    src_centroid = get_source_centroid(source_mask)
    if src_centroid is None:
        print(f"[WARN] Source centroid no encontrado, no se guarda: {save_path}")
        return

    gt_peak = np.array(get_peak_index_masked(gt, brain_mask), dtype=np.float32)

    prof_gt = sample_line_trilinear(gt, src_centroid, gt_peak, n=n_samples)
    prof_pr = sample_line_trilinear(pred, src_centroid, gt_peak, n=n_samples)

    plt.figure(figsize=(7, 4))
    plt.plot(prof_gt, label="GT")
    plt.plot(prof_pr, label="Pred")
    plt.title("Approx. profile: source-mask centroid � GT brain peak")
    plt.xlabel("Samples along line")
    plt.ylabel("p_max_norm")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_triplet(gt, pred, brain_mask, save_path):
    z, y, x = get_peak_index_masked(gt, brain_mask)

    gt_sag = gt[:, :, x]
    gt_cor = gt[:, y, :]
    gt_ax = gt[z, :, :]

    pr_sag = pred[:, :, x]
    pr_cor = pred[:, y, :]
    pr_ax = pred[z, :, :]

    err = np.abs(pred - gt)
    er_sag = err[:, :, x]
    er_cor = err[:, y, :]
    er_ax = err[z, :, :]

    vmin = 0.0
    vmax = float(gt.max())

    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    rows = [
        ("GT", [gt_sag, gt_cor, gt_ax]),
        ("Pred", [pr_sag, pr_cor, pr_ax]),
        ("|Error|", [er_sag, er_cor, er_ax]),
    ]

    cols = ["Sagittal", "Coronal", "Axial"]

    for i, (row_name, imgs) in enumerate(rows):
        for j, img in enumerate(imgs):
            ax = axes[i, j]

            if row_name == "|Error|":
                im = ax.imshow(img.T, origin="lower", cmap="jet")
            else:
                im = ax.imshow(img.T, origin="lower", cmap="jet", vmin=vmin, vmax=vmax)

            ax.set_title(f"{row_name} - {cols[j]}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# =========================================================
# MAIN
# =========================================================
@torch.no_grad()
def main():
    print("Device:", DEVICE)
    print("TEST_DIR:", TEST_DIR)
    print("CKPT_PATH:", CKPT_PATH)
    print("OUT_DIR:", OUT_DIR)
    print("DX_MM:", DX_MM)

    # 1) Crear dataset
    test_ds = TusTestDataset(TEST_DIR)

    # 2) Seleccionar siempre los mismos 5 casos no-water
    validate_force_files(test_ds, FORCE_FILES)

    if FORCE_FILES:
        vis_files = set(FORCE_FILES)
    else:
        vis_files = select_fixed_random_non_water_cases(
            test_ds,
            n_cases=N_RANDOM_CASES,
            seed=FIXED_RANDOM_SEED,
        )

    print("\nCasos seleccionados para visualizaci�n:")
    for f in sorted(vis_files):
        print(" -", f)

    print(f"\nSe visualizar�n {len(vis_files)} casos.")

    # 3) Crear DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # 4) Cargar modelo/generator
    model = load_model(CKPT_PATH, DEVICE)

    # 5) Inferencia y visualizaci�n
    for batch in test_loader:
        X = batch["X"].to(DEVICE, non_blocking=True)
        y = batch["y"].to(DEVICE, non_blocking=True)
        brain = batch["brain_mask"].to(DEVICE, non_blocking=True)
        src = batch["source_mask"].to(DEVICE, non_blocking=True)

        is_water_only = bool(batch["is_water_only"][0])
        file_name = batch["file_name"][0]

        if file_name not in vis_files:
            continue

        if is_water_only:
            print(f"[SKIP] {file_name} es water-only.")
            continue

        pred = model(X).clamp_min(0.0)

        pred_np = pred[0, 0].detach().cpu().numpy().astype(np.float32)
        gt_np = y[0, 0].detach().cpu().numpy().astype(np.float32)
        brain_np = brain[0, 0].detach().cpu().numpy().astype(np.float32)
        src_np = src[0, 0].detach().cpu().numpy().astype(np.float32)

        if brain_np.sum() == 0:
            print(f"[SKIP] {file_name} no tiene brain mask �til.")
            continue

        base = os.path.splitext(file_name)[0]
        case_dir = os.path.join(OUT_DIR, base)
        os.makedirs(case_dir, exist_ok=True)

        # 0) Slices GT / Pred / Error
        plot_triplet(
            gt=gt_np,
            pred=pred_np,
            brain_mask=brain_np,
            save_path=os.path.join(case_dir, f"{base}_triplet.png"),
        )

        # 1) Scatter GT vs Pred
        plot_scatter_gt_vs_pred(
            gt=gt_np,
            pred=pred_np,
            brain_mask=brain_np,
            save_path=os.path.join(case_dir, f"{base}_scatter_brain.png"),
            thr=SCATTER_THR,
            max_points=SCATTER_MAX_POINTS,
        )

        # 2) Perfiles X/Y/Z en el pico GT cerebral
        plot_axis_profiles(
            gt=gt_np,
            pred=pred_np,
            brain_mask=brain_np,
            save_path=os.path.join(case_dir, f"{base}_axis_profiles.png"),
            dx_mm=DX_MM,
        )

        # 3) Perfil aproximado centroide source_mask -> pico GT cerebral
        plot_source_centroid_to_peak_profile(
            gt=gt_np,
            pred=pred_np,
            source_mask=src_np,
            brain_mask=brain_np,
            save_path=os.path.join(case_dir, f"{base}_sourceCentroid_to_peak_profile.png"),
            n_samples=PROFILE_SAMPLES,
        )

        print(f"[OK] Figuras guardadas para: {file_name}")

    print("\n[DONE] Visualizaciones terminadas.")


if __name__ == "__main__":
    main()