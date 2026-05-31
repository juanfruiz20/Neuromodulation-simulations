from src.models.resunet_3d import ResUNet3D_HQ
import torch
from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation,
    center_of_mass,
)
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")


# =========================================================
# CONFIG
# =========================================================
DATA_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

# =========================================================
# MODEL A / MODEL B
# =========================================================
CKPT_PATH_A = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_expDexpB/epoch_030.pth"

CKPT_PATH_B = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_TFG/epoch_220.pth"

MODEL_A_LABEL = "U-Net Full Loss"
MODEL_B_LABEL = "cGAN Full Loss 220e"

OUT_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/tablas/ModelA_vs_ModelB_GT_examples"

EXPECTED_SHAPE = (128, 128, 128)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Para la figura 4x4
N_CASES = 4
CASE_LABELS = ["(a)", "(b)", "(c)", "(d)"]

TOP_LABEL_FONTSIZE = 15
TOP_PLANE_FONTSIZE = 10
LEFT_LABEL_FONTSIZE = 13

FIGSIZE_4X4 = (11.2, 10.6)

WSPACE = 0.02
HSPACE = 0.02

LEFT_MARGIN = 0.085
RIGHT_MARGIN = 0.995
TOP_MARGIN = 0.93
BOTTOM_MARGIN = 0.035
# Si quieres forzar 4 casos concretos, ponlos aqu�.
# Si lo dejas vac�o, el script selecciona 4 casos no-water autom�ticamente.
FORCE_FILES = set()

# Ejemplo:
FORCE_FILES = {
    "sample_0174.npz",
    "sample_0151.npz",
    "sample_0639.npz",
    "sample_0040.npz",
}

# =========================================================
# PLANE SELECTION
# =========================================================
AUTO_SELECT_BEST_PLANE = True

# Solo se usa si AUTO_SELECT_BEST_PLANE = False
PLANES = ["sagittal", "coronal", "axial"]

CMAP_ANATOMY = "gray"
CMAP_INTENSITY = "jet"

DPI = 320
SAVE_NAME = "modelA_modelB_GT_geometry_4cases_4x4_best_plane.png"

SHOW_PEAK_MARKER = False

# =========================================================
# Model config
# Ajusta esto solo si tu clase ResUNet3D_HQ requiere otros argumentos.
# =========================================================
MODEL_KWARGS = dict(
    in_ch=2,
    out_ch=1,
    out_positive=True,
)

# =========================================================
# Transducer contour visualization
# =========================================================
TRANSDUCER_DILATION_ITERS = 2
TRANSDUCER_CONTOUR_COLOR = "yellow"
TRANSDUCER_CONTOUR_WIDTH = 1.6
TRANSDUCER_CONTOUR_ALPHA = 0.95
TRANSDUCER_THRESHOLD = 0.5

# "full_projection" -> proyecta todo el transductor sobre el plano.
# "slice"           -> muestra solo el corte exacto del transductor.
TRANSDUCER_VIS_MODE = "full_projection"

# =========================================================
# Diverse case selection
# =========================================================
DIVERSE_SELECTION_SEED_INDEX = 11

os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# DATA HELPERS
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


def reconstruct_brain_mask(mask_skull: np.ndarray, is_water_only: bool) -> np.ndarray:
    skull = mask_skull > 0.5

    if is_water_only or skull.sum() == 0:
        return np.zeros_like(mask_skull, dtype=np.float32)

    filled = binary_fill_holes(skull)
    brain = np.logical_and(filled, np.logical_not(skull))

    return brain.astype(np.float32)


def choose_anatomy_volume(d, skull):
    """
    Si el .npz tiene alg�n volumen anat�mico, lo usa.
    Si no, usa mask_skull.
    """
    possible_keys = [
        "ct",
        "CT",
        "ct_norm",
        "skull_ct",
        "rho",
        "density",
        "medium_density",
        "c0",
        "sound_speed",
    ]

    for key in possible_keys:
        if key in d:
            arr = d[key].astype(np.float32)
            if arr.shape == skull.shape:
                return arr, key

    return skull.astype(np.float32), "mask_skull"


def load_case(path: str, expected_shape=(128, 128, 128)):
    with np.load(path) as d:
        src = d["source_mask"].astype(np.float32)
        skull = d["mask_skull"].astype(np.float32)

        if "p_max_norm" not in d:
            raise RuntimeError(
                f"{os.path.basename(path)} no contiene 'p_max_norm'."
            )

        gt = d["p_max_norm"].astype(np.float32)
        anatomy, anatomy_key = choose_anatomy_volume(d, skull)

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
        ("p_max_norm", gt),
        ("anatomy", anatomy),
    ]:
        if arr.shape != expected_shape:
            raise RuntimeError(
                f"Shape inv�lida en {os.path.basename(path)} | "
                f"{name}: {arr.shape} | esperado: {expected_shape}"
            )

    brain = reconstruct_brain_mask(skull, is_water_only)

    return {
        "file_name": os.path.basename(path),
        "path": path,
        "source_mask": src,
        "mask_skull": skull,
        "anatomy": anatomy,
        "anatomy_key": anatomy_key,
        "gt": gt,
        "brain_mask": brain,
        "is_water_only": is_water_only,
    }


def get_peak_index(gt: np.ndarray, brain_mask: np.ndarray):
    """
    Peak del ground truth dentro del brain_mask.
    Retorna coordenadas [z, y, x].
    """
    if brain_mask is not None and brain_mask.sum() > 0:
        masked = np.where(brain_mask > 0.5, gt, -np.inf)
        flat_idx = int(np.argmax(masked))
    else:
        flat_idx = int(np.argmax(gt))

    return np.unravel_index(flat_idx, gt.shape)


def get_skull_center(mask_skull: np.ndarray):
    """
    Centro del cr�neo para que la geometr�a se vea bien.
    """
    pts = np.argwhere(mask_skull > 0.5)

    if len(pts) == 0:
        zc, yc, xc = np.array(mask_skull.shape) // 2
    else:
        zc, yc, xc = np.round(pts.mean(axis=0)).astype(int)

    return int(zc), int(yc), int(xc)


def get_source_center(source_mask: np.ndarray):
    """
    Centro aproximado del transductor/fuente.
    Retorna coordenadas [z, y, x].
    """
    pts = np.argwhere(source_mask > 0.5)

    if len(pts) == 0:
        return np.array(source_mask.shape, dtype=np.float32) / 2.0

    return pts.mean(axis=0).astype(np.float32)


def choose_most_significant_plane(case):
    """
    Escoge autom�ticamente el plano m�s significativo.

    Criterio:
    se elige el plano donde la proyecci�n 2D del vector entre
    el centro del transductor y el pico del ground truth es mayor.

    axial    -> muestra Y-X, ignora Z
    coronal  -> muestra Z-X, ignora Y
    sagittal -> muestra Z-Y, ignora X
    """
    gt = case["gt"]
    brain = case["brain_mask"]
    src = case["source_mask"]

    peak = np.array(get_peak_index(gt, brain), dtype=np.float32)  # [z, y, x]
    src_center = get_source_center(src)                           # [z, y, x]

    delta = np.abs(peak - src_center)

    scores = {
        "axial": float(np.linalg.norm(delta[[1, 2]])),     # Y-X
        "coronal": float(np.linalg.norm(delta[[0, 2]])),   # Z-X
        "sagittal": float(np.linalg.norm(delta[[0, 1]])),  # Z-Y
    }

    best_plane = max(scores, key=scores.get)

    print(f"[INFO] Best plane for {case['file_name']}: {best_plane}")
    print(f"[INFO] Plane scores: {scores}")

    return best_plane


def get_planes_for_case(case):
    if AUTO_SELECT_BEST_PLANE:
        return [choose_most_significant_plane(case)]

    return PLANES


def extract_plane(vol, z, y, x, plane: str):
    """
    vol shape: [Z, Y, X]
    """
    if plane == "axial":
        return vol[z, :, :]        # Y-X
    elif plane == "coronal":
        return vol[:, y, :]        # Z-X
    elif plane == "sagittal":
        return vol[:, :, x]        # Z-Y
    else:
        raise ValueError(f"Plano no reconocido: {plane}")


def project_volume_to_plane(vol, plane: str):
    """
    Proyecci�n m�xima del volumen completo sobre un plano.
    Se usa solo para visualizar el contorno del transductor.
    """
    if plane == "axial":
        return np.max(vol, axis=0)      # Y-X
    elif plane == "coronal":
        return np.max(vol, axis=1)      # Z-X
    elif plane == "sagittal":
        return np.max(vol, axis=2)      # Z-Y
    else:
        raise ValueError(f"Plano no reconocido: {plane}")


def peak_marker_coords(z, y, x, plane: str):
    if plane == "axial":
        return x, y
    elif plane == "coronal":
        return x, z
    elif plane == "sagittal":
        return y, z
    else:
        raise ValueError(f"Plano no reconocido: {plane}")


def normalize_for_display(arr):
    arr = arr.astype(np.float32)
    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))

    if amax <= amin:
        return np.zeros_like(arr, dtype=np.float32)

    return (arr - amin) / (amax - amin)


def make_transducer_vis_mask(src_3d: np.ndarray, dilation_iters=2):
    """
    Engrosa ligeramente el transductor solo para visualizaci�n.
    """
    src_bin = src_3d > 0.5
    src_vis = binary_dilation(src_bin, iterations=dilation_iters)
    return src_vis.astype(np.float32)


def draw_transducer_contour(ax, src_slice):
    """
    Dibuja el transductor como contorno amarillo.
    """
    if src_slice is None:
        return

    if np.nanmax(src_slice) <= TRANSDUCER_THRESHOLD:
        return

    ax.contour(
        src_slice,
        levels=[TRANSDUCER_THRESHOLD],
        colors=TRANSDUCER_CONTOUR_COLOR,
        linewidths=TRANSDUCER_CONTOUR_WIDTH,
        alpha=TRANSDUCER_CONTOUR_ALPHA,
    )


def draw_peak_marker(ax, z, y, x, plane):
    """
    Dibuja marcador del peak si SHOW_PEAK_MARKER=True.
    """
    if not SHOW_PEAK_MARKER:
        return

    px, py = peak_marker_coords(z, y, x, plane)
    ax.plot(
        px,
        py,
        marker="x",
        markersize=5,
        markeredgewidth=1.2,
        color="white",
    )


# =========================================================
# MODEL HELPERS
# =========================================================
def strip_prefix_if_present(state_dict, prefixes):
    new_sd = {}

    for k, v in state_dict.items():
        new_key = k

        for p in prefixes:
            if new_key.startswith(p):
                new_key = new_key[len(p):]

        new_sd[new_key] = v

    return new_sd


def extract_model_state_dict(ckpt):
    """
    Intenta extraer el state_dict del modelo desde distintos formatos
    comunes de checkpoints.
    """
    if not isinstance(ckpt, dict):
        return ckpt

    possible_keys = [
        "generator_state_dict",
        "gen_state_dict",
        "G_state_dict",
        "netG_state_dict",
        "model_state_dict",
        "state_dict",
        "generator",
        "gen",
        "G",
        "model",
    ]

    for key in possible_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    # Si parece que el checkpoint ya es directamente un state_dict
    if all(isinstance(k, str) for k in ckpt.keys()):
        tensor_like = [torch.is_tensor(v) for v in ckpt.values()]
        if len(tensor_like) > 0 and any(tensor_like):
            return ckpt

    raise RuntimeError(
        "No se pudo encontrar el state_dict del modelo dentro del checkpoint."
    )


def load_model(ckpt_path: str, device: str, label: str):
    print("\n======================================")
    print(f"Loading {label}")
    print("======================================")
    print("CKPT_PATH:", ckpt_path)
    print("DEVICE:", device)

    model = ResUNet3D_HQ(**MODEL_KWARGS).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = extract_model_state_dict(ckpt)

    state_dict = strip_prefix_if_present(
        state_dict,
        prefixes=[
            "module.",
            "generator.",
            "gen.",
            "G.",
            "netG.",
            "model.",
        ],
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[INFO] {label} missing keys: {len(missing)}")
    print(f"[INFO] {label} unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[WARN] First missing keys:", missing[:5])

    if len(unexpected) > 0:
        print("[WARN] First unexpected keys:", unexpected[:5])

    model.eval()
    return model


@torch.no_grad()
def predict_case(model, case, device):
    """
    Input del modelo: [source_mask, mask_skull]
    Output esperado: p_max_norm predicho.
    """
    src = case["source_mask"].astype(np.float32)
    skull = case["mask_skull"].astype(np.float32)

    x_np = np.stack([src, skull], axis=0)  # [2, Z, Y, X]
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # [1, 2, Z, Y, X]

    pred = model(x)

    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    pred_np = pred[0, 0].detach().cpu().numpy().astype(np.float32)

    pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=0.0, neginf=0.0)
    pred_np = np.clip(pred_np, 0.0, None)

    return pred_np


# =========================================================
# DIVERSE CASE SELECTION
# =========================================================
def compute_case_feature_vector(case):
    gt = case["gt"]
    skull = case["mask_skull"]
    src = case["source_mask"]
    brain = case["brain_mask"]

    zpk, ypk, xpk = get_peak_index(gt, brain)

    src_bin = src > 0.5
    if src_bin.sum() > 0:
        cz, cy, cx = center_of_mass(src_bin)
    else:
        cz, cy, cx = (
            gt.shape[0] / 2,
            gt.shape[1] / 2,
            gt.shape[2] / 2,
        )

    skull_frac = float((skull > 0.5).mean())
    peak_val = float(gt.max())

    dz = (zpk - cz) / gt.shape[0]
    dy = (ypk - cy) / gt.shape[1]
    dx = (xpk - cx) / gt.shape[2]

    dist_src_peak = np.sqrt(dz * dz + dy * dy + dx * dx)

    feat = np.array(
        [
            skull_frac,
            cz / gt.shape[0],
            cy / gt.shape[1],
            cx / gt.shape[2],
            zpk / gt.shape[0],
            ypk / gt.shape[1],
            xpk / gt.shape[2],
            dist_src_peak,
            peak_val,
        ],
        dtype=np.float32,
    )

    return feat


def farthest_point_sampling(features, n_select=4, seed_index=0):
    n = len(features)

    if n <= n_select:
        return list(range(n))

    feats = np.stack(features, axis=0)

    mu = feats.mean(axis=0, keepdims=True)
    sigma = feats.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0

    feats = (feats - mu) / sigma

    selected = [seed_index % n]
    remaining = set(range(n)) - set(selected)

    while len(selected) < n_select:
        best_idx = None
        best_score = -np.inf

        for idx in remaining:
            dists = [np.linalg.norm(feats[idx] - feats[s]) for s in selected]
            score = min(dists)

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def select_diverse_non_water_cases(data_dir: str, n_cases=4):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    if len(files) == 0:
        raise RuntimeError(f"No se encontraron archivos .npz en: {data_dir}")

    candidate_cases = []

    for path in files:
        if read_is_water_only(path):
            continue

        case = load_case(path, expected_shape=EXPECTED_SHAPE)
        candidate_cases.append(case)

    if len(candidate_cases) < n_cases:
        raise RuntimeError(
            f"Solo se encontraron {len(candidate_cases)} casos no-water, "
            f"pero pediste {n_cases}."
        )

    features = [compute_case_feature_vector(c) for c in candidate_cases]

    selected_idx = farthest_point_sampling(
        features,
        n_select=n_cases,
        seed_index=DIVERSE_SELECTION_SEED_INDEX,
    )

    return [candidate_cases[i] for i in selected_idx]


# =========================================================
# PLOT: 4 ROWS x 4 COLUMNS
# Rows: Model A / Model B / GT / Geometry
# Columns: Cases
# =========================================================
def plot_modelA_modelB_gt_geometry_4x4(cases, save_path):
    """
    Figura final:
        columnas = casos
        filas    = Model A, Model B, Ground truth, Geometry

    Cada caso usa su plano automáticamente seleccionado.
    """

    if len(cases) != 4:
        raise RuntimeError(
            f"Esta versión está pensada para exactamente 4 casos. "
            f"Recibidos: {len(cases)}"
        )

    plane_titles = {
        "axial": "Axial",
        "coronal": "Coronal",
        "sagittal": "Sagittal",
    }

    n_rows = 4
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=FIGSIZE_4X4,
    )

    axes = np.asarray(axes)

    row_labels = [
        MODEL_A_LABEL,
        MODEL_B_LABEL,
        "Ground truth",
        "Geometry",
    ]

    for col, case in enumerate(cases):
        gt = case["gt"]
        pred_a = case["pred_a"]
        pred_b = case["pred_b"]

        src = case["source_mask"]
        skull = case["mask_skull"]
        anatomy = case["anatomy"]
        brain = case["brain_mask"]

        # Plano automático para este caso
        plane = get_planes_for_case(case)[0]

        # Predicciones/GT centrados en el pico del GT
        z_peak, y_peak, x_peak = get_peak_index(gt, brain)

        # Geometría centrada en el cráneo
        z_skull, y_skull, x_skull = get_skull_center(skull)

        case_vmin = 0.0
        case_vmax = float(max(gt.max(), pred_a.max(), pred_b.max()))
        if case_vmax <= 0:
            case_vmax = 1.0

        src_vis_3d = make_transducer_vis_mask(
            src,
            dilation_iters=TRANSDUCER_DILATION_ITERS,
        )

        # -------------------------------------------------
        # Row 0: Prediction Model A
        # -------------------------------------------------
        ax = axes[0, col]
        pred_a_slice = extract_plane(pred_a, z_peak, y_peak, x_peak, plane)

        ax.imshow(
            pred_a_slice,
            cmap=CMAP_INTENSITY,
            vmin=case_vmin,
            vmax=case_vmax,
            origin="lower",
            interpolation="nearest",
        )
        draw_peak_marker(ax, z_peak, y_peak, x_peak, plane)

        # -------------------------------------------------
        # Row 1: Prediction Model B
        # -------------------------------------------------
        ax = axes[1, col]
        pred_b_slice = extract_plane(pred_b, z_peak, y_peak, x_peak, plane)

        ax.imshow(
            pred_b_slice,
            cmap=CMAP_INTENSITY,
            vmin=case_vmin,
            vmax=case_vmax,
            origin="lower",
            interpolation="nearest",
        )
        draw_peak_marker(ax, z_peak, y_peak, x_peak, plane)

        # -------------------------------------------------
        # Row 2: Ground truth
        # -------------------------------------------------
        ax = axes[2, col]
        gt_slice = extract_plane(gt, z_peak, y_peak, x_peak, plane)

        ax.imshow(
            gt_slice,
            cmap=CMAP_INTENSITY,
            vmin=case_vmin,
            vmax=case_vmax,
            origin="lower",
            interpolation="nearest",
        )
        draw_peak_marker(ax, z_peak, y_peak, x_peak, plane)

        # -------------------------------------------------
        # Row 3: Geometry
        # -------------------------------------------------
        ax = axes[3, col]
        anatomy_slice = extract_plane(
            anatomy, z_skull, y_skull, x_skull, plane)
        anatomy_slice = normalize_for_display(anatomy_slice)

        ax.imshow(
            anatomy_slice,
            cmap=CMAP_ANATOMY,
            origin="lower",
            interpolation="nearest",
        )

        if TRANSDUCER_VIS_MODE == "full_projection":
            src_vis_slice = project_volume_to_plane(src_vis_3d, plane)
        else:
            src_vis_slice = extract_plane(
                src_vis_3d,
                z_skull,
                y_skull,
                x_skull,
                plane,
            )

        draw_transducer_contour(ax, src_vis_slice)

        # -------------------------------------------------
        # Título superior de columna
        # Sin sample_xxxx
        # -------------------------------------------------
        col_label = CASE_LABELS[col]
        plane_name = plane_titles.get(plane, plane)

        axes[0, col].set_title(
            f"{col_label}\n{plane_name}",
            fontsize=TOP_LABEL_FONTSIZE,
            fontweight="bold",
            pad=4,
        )

    # -------------------------------------------------
    # Formato general de ejes
    # -------------------------------------------------
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].set_aspect("equal")

    # -------------------------------------------------
    # Labels de la izquierda: más grandes y más cerca
    # -------------------------------------------------
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(
            label,
            fontsize=LEFT_LABEL_FONTSIZE,
            fontweight="bold",
            rotation=90,
            labelpad=10,
        )

    # -------------------------------------------------
    # Espaciado fino
    # -------------------------------------------------
    plt.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        top=TOP_MARGIN,
        bottom=BOTTOM_MARGIN,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    fig.savefig(
        save_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"\n[OK] Figura 4x4 guardada en:\n{save_path}")


# =========================================================
# RUN
# =========================================================
def main():
    print("===============================================")
    print("Model A vs Model B vs Ground truth visualization")
    print("4 cases | 4 rows x 4 columns | automatic best plane")
    print("===============================================")
    print("DATA_DIR:", DATA_DIR)
    print("CKPT_PATH_A:", CKPT_PATH_A)
    print("CKPT_PATH_B:", CKPT_PATH_B)
    print("OUT_DIR:", OUT_DIR)
    print("DEVICE:", DEVICE)
    print("N_CASES:", N_CASES)
    print("AUTO_SELECT_BEST_PLANE:", AUTO_SELECT_BEST_PLANE)

    model_a = load_model(CKPT_PATH_A, DEVICE, MODEL_A_LABEL)
    model_b = load_model(CKPT_PATH_B, DEVICE, MODEL_B_LABEL)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))

    if len(all_files) == 0:
        raise RuntimeError(f"No se encontraron archivos .npz en: {DATA_DIR}")

    available = {os.path.basename(p): p for p in all_files}

    if FORCE_FILES:
        if len(FORCE_FILES) != 4:
            raise RuntimeError(
                f"Para esta versi�n 4x4, FORCE_FILES debe tener exactamente 4 archivos. "
                f"Actualmente tiene: {len(FORCE_FILES)}"
            )

        selected_cases = []

        for fname in sorted(FORCE_FILES):
            if fname not in available:
                raise RuntimeError(f"FORCE_FILE no encontrado: {fname}")

            if read_is_water_only(available[fname]):
                raise RuntimeError(
                    f"FORCE_FILE es water-only y no deber�a usarse: {fname}"
                )

            selected_cases.append(
                load_case(available[fname], expected_shape=EXPECTED_SHAPE)
            )

    else:
        selected_cases = select_diverse_non_water_cases(
            DATA_DIR,
            n_cases=N_CASES,
        )

    if len(selected_cases) != 4:
        raise RuntimeError(
            f"Esta versi�n necesita exactamente 4 casos. "
            f"Casos seleccionados: {len(selected_cases)}"
        )

    print("\nCasos no-water seleccionados:")
    for c in selected_cases:
        print(" -", c["file_name"])

    print("\nRunning model predictions...")

    for case in selected_cases:
        pred_a = predict_case(model_a, case, DEVICE)
        pred_b = predict_case(model_b, case, DEVICE)

        case["pred_a"] = pred_a
        case["pred_b"] = pred_b

        print(
            f"[INFO] {case['file_name']} | "
            f"GT max={case['gt'].max():.4f} | "
            f"A max={pred_a.max():.4f} | "
            f"B max={pred_b.max():.4f}"
        )

    save_path = os.path.join(OUT_DIR, SAVE_NAME)

    plot_modelA_modelB_gt_geometry_4x4(
        cases=selected_cases,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
