import os
import glob
import csv
import random
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_fill_holes

from src.modelos.ResUnet3D import ResUNet3D_HQ


# =========================================================
# CONFIG
# =========================================================
TEST_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

CKPT_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_exp04_300epoch/epoch_290.pth"

OUT_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/tablas/visualscGAN300/extra"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DX_MM = 1.0

NUM_WORKERS = 0
PIN_MEMORY = True if DEVICE == "cuda" else False

# Mismos 5 casos no-water siempre
N_RANDOM_CASES = 5
FIXED_RANDOM_SEED = 42

# Si quieres fijar manualmente los 5 casos, ponlos aqu�.
# Si est� vac�o, selecciona siempre los mismos 5 no-water con seed=42.
FORCE_FILES = set()
# FORCE_FILES = {
#     "sample_0012.npz",
#     "sample_0034.npz",
#     "sample_0088.npz",
#     "sample_0142.npz",
#     "sample_0201.npz",
# }

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
# CASE SELECTION
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
    valid_files = []

    for path in dataset.files:
        if not read_is_water_only(path):
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
    if not force_files:
        return

    available = {os.path.basename(p): p for p in dataset.files}

    for fname in force_files:
        if fname not in available:
            raise RuntimeError(f"FORCE_FILE no encontrado en TEST_DIR: {fname}")

        if read_is_water_only(available[fname]):
            raise RuntimeError(f"FORCE_FILE es water-only y no deber�a usarse: {fname}")


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

    if isinstance(ckpt, dict) and "G" in ckpt:
        print("Cargando GENERATOR G desde checkpoint cGAN")
        model.load_state_dict(ckpt["G"], strict=True)

    elif isinstance(ckpt, dict) and "model" in ckpt:
        print("Cargando modelo desde checkpoint U-Net")
        model.load_state_dict(ckpt["model"], strict=True)

    else:
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)
        raise KeyError(
            f"No se encontr� ni 'G' ni 'model' en el checkpoint: {ckpt_path}. "
            f"Keys disponibles: {keys}"
        )

    model.eval()
    return model


# =========================================================
# PROFILE HELPERS
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
    vol shape: [Z, Y, X]
    p0, p1: coordenadas flotantes [z, y, x]
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


def extend_ray_to_volume_boundary(p0, p1, vol_shape):
    """
    Extiende la l�nea desde p0 en direcci�n a p1 hasta el borde del volumen.

    p0: source centroid [z, y, x]
    p1: GT peak [z, y, x]
    vol_shape: (Z, Y, X)
    """
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)

    direction = p1 - p0
    norm = np.linalg.norm(direction)

    if norm < 1e-8:
        return p1.astype(np.float32)

    direction = direction / norm

    bounds_min = np.array([0, 0, 0], dtype=np.float32)
    bounds_max = np.array(
        [vol_shape[0] - 1, vol_shape[1] - 1, vol_shape[2] - 1],
        dtype=np.float32,
    )

    t_candidates = []

    for i in range(3):
        if direction[i] > 1e-8:
            t = (bounds_max[i] - p0[i]) / direction[i]
            t_candidates.append(t)
        elif direction[i] < -1e-8:
            t = (bounds_min[i] - p0[i]) / direction[i]
            t_candidates.append(t)

    t_candidates = [t for t in t_candidates if t > 0]

    if len(t_candidates) == 0:
        return p1.astype(np.float32)

    t_exit = min(t_candidates)
    p_end = p0 + t_exit * direction

    return p_end.astype(np.float32)


def pearson_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return float("nan")

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


# =========================================================
# APPROX PROFILE PLOT
# =========================================================
def plot_approx_profile_centroid_to_volume_end(
    gt,
    pred,
    source_mask,
    brain_mask,
    save_path,
    n_samples=256,
    dx_mm=1.0,
    save_csv=True,
):
    """
    Perfil aproximado del haz:

        source centroid -> GT brain peak -> volume boundary

    Calcula Pearson correlation entre:
        prof_gt y prof_pred
    """
    src_centroid = get_source_centroid(source_mask)

    if src_centroid is None:
        print(f"[WARN] Source centroid no encontrado: {save_path}")
        return float("nan")

    gt_peak = np.array(get_peak_index_masked(gt, brain_mask), dtype=np.float32)

    line_end = extend_ray_to_volume_boundary(
        p0=src_centroid,
        p1=gt_peak,
        vol_shape=gt.shape,
    )

    prof_gt = sample_line_trilinear(
        vol=gt,
        p0=src_centroid,
        p1=line_end,
        n=n_samples,
    )

    prof_pred = sample_line_trilinear(
        vol=pred,
        p0=src_centroid,
        p1=line_end,
        n=n_samples,
    )

    pearson_r = pearson_corr_1d(prof_gt, prof_pred)

    total_dist_vox = float(np.linalg.norm(line_end - src_centroid))
    peak_dist_vox = float(np.linalg.norm(gt_peak - src_centroid))

    if total_dist_vox > 1e-8:
        peak_sample_idx = int(round((peak_dist_vox / total_dist_vox) * (n_samples - 1)))
    else:
        peak_sample_idx = 0

    peak_sample_idx = int(np.clip(peak_sample_idx, 0, n_samples - 1))

    distance_axis_mm = np.linspace(
        0,
        total_dist_vox * dx_mm,
        n_samples,
        dtype=np.float32,
    )

    plt.figure(figsize=(8, 4.5))
    plt.plot(distance_axis_mm, prof_gt, label="GT")
    plt.plot(distance_axis_mm, prof_pred, label="Pred")
    plt.axvline(
        distance_axis_mm[peak_sample_idx],
        linestyle="--",
        linewidth=1,
        label="GT peak position",
    )

    plt.title(f"Approx. beam profile | Pearson r = {pearson_r:.4f}")
    plt.xlabel("Distance from source centroid (mm)")
    plt.ylabel("p_max_norm")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

    if save_csv:
        csv_path = save_path.replace(".png", ".csv")

        profile_data = np.column_stack([
            distance_axis_mm,
            prof_gt,
            prof_pred,
        ])

        header = (
            f"Pearson_r={pearson_r:.8f}\n"
            f"source_centroid_z={src_centroid[0]:.4f},"
            f"source_centroid_y={src_centroid[1]:.4f},"
            f"source_centroid_x={src_centroid[2]:.4f}\n"
            f"gt_peak_z={gt_peak[0]:.4f},"
            f"gt_peak_y={gt_peak[1]:.4f},"
            f"gt_peak_x={gt_peak[2]:.4f}\n"
            f"line_end_z={line_end[0]:.4f},"
            f"line_end_y={line_end[1]:.4f},"
            f"line_end_x={line_end[2]:.4f}\n"
            "distance_mm,GT,Pred"
        )

        np.savetxt(
            csv_path,
            profile_data,
            delimiter=",",
            header=header,
            comments="",
        )

    print(f"[OK] {os.path.basename(save_path)} | Pearson r = {pearson_r:.4f}")

    return pearson_r


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
    print("PROFILE_SAMPLES:", PROFILE_SAMPLES)

    test_ds = TusTestDataset(TEST_DIR)

    validate_force_files(test_ds, FORCE_FILES)

    if FORCE_FILES:
        vis_files = set(FORCE_FILES)
    else:
        vis_files = select_fixed_random_non_water_cases(
            test_ds,
            n_cases=N_RANDOM_CASES,
            seed=FIXED_RANDOM_SEED,
        )

    print("\nCasos seleccionados:")
    for f in sorted(vis_files):
        print(" -", f)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = load_model(CKPT_PATH, DEVICE)

    summary_rows = []

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

        save_path = os.path.join(
            case_dir,
            f"{base}_approx_profile_centroid_to_volume_end.png"
        )

        pearson_r = plot_approx_profile_centroid_to_volume_end(
            gt=gt_np,
            pred=pred_np,
            source_mask=src_np,
            brain_mask=brain_np,
            save_path=save_path,
            n_samples=PROFILE_SAMPLES,
            dx_mm=DX_MM,
            save_csv=True,
        )

        summary_rows.append({
            "file": file_name,
            "pearson_profile": pearson_r,
        })

    summary_csv = os.path.join(OUT_DIR, "approx_profile_pearson_summary.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "pearson_profile"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nResumen guardado en: {summary_csv}")
    print("[DONE] Approx profiles terminados.")


if __name__ == "__main__":
    main()