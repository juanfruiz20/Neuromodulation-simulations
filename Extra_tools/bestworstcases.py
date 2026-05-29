import os
import glob
import csv
import inspect
import numpy as np

import torch

from scipy.ndimage import binary_fill_holes

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from src.modelos.ResUnet3D import ResUNet3D_HQ


# =========================================================
# CONFIG
# =========================================================
TEST_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

CKPT_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_expDexpB/epoch_030.pth"

OUT_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/ranking_test_cases2"

EXPECTED_SHAPE = (128, 128, 128)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Si False, ignora casos water-only.
EVALUATE_WATER_ONLY = False

# Si True, exige que el checkpoint coincida exactamente con la arquitectura.
# Recomendado para ranking serio.
STRICT_LOAD = True

# Ranking weights.
# Puedes cambiarlos seg�n lo que quieras enfatizar.
WEIGHTS = {
    "ssim_global": 0.30,
    "ssim_brain": 0.35,
    "peak_loc_err_mm_brain": 0.35,
}

# Modelo. Si tu checkpoint no es ResUNet3D_HQ, cambia esta clase.
MODEL_CLASS = ResUNet3D_HQ

# Candidatos de argumentos. El script usa solo los que existan en la firma real.
MODEL_CANDIDATE_KWARGS = dict(
    in_ch=2,
    out_ch=1,
    out_positive=True,
    in_channels=2,
    out_channels=1,
    n_channels=2,
    n_classes=1,
    positive_out=True,
)

PER_CASE_CSV = "per_case_metrics.csv"
RANKING_CSV = "ranked_test_cases.csv"

os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# MODEL HELPERS
# =========================================================
def build_model(device: str):
    sig = inspect.signature(MODEL_CLASS)
    valid_params = sig.parameters

    final_kwargs = {
        k: v for k, v in MODEL_CANDIDATE_KWARGS.items()
        if k in valid_params
    }

    print("[INFO] Model class:", MODEL_CLASS.__name__)
    print("[INFO] Model signature:")
    print(sig)
    print("[INFO] Model kwargs used:")
    print(final_kwargs)

    model = MODEL_CLASS(**final_kwargs).to(device)
    return model


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

    if all(isinstance(k, str) for k in ckpt.keys()):
        tensor_like = [torch.is_tensor(v) for v in ckpt.values()]
        if len(tensor_like) > 0 and any(tensor_like):
            return ckpt

    raise RuntimeError(
        "No se pudo encontrar el state_dict del modelo dentro del checkpoint."
    )


def load_model(ckpt_path: str, device: str):
    print("\n======================================")
    print("Loading model")
    print("======================================")
    print("CKPT_PATH:", ckpt_path)
    print("DEVICE:", device)

    model = build_model(device)

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

    load_info = model.load_state_dict(state_dict, strict=STRICT_LOAD)

    if STRICT_LOAD:
        print("[INFO] Checkpoint loaded with strict=True.")
    else:
        missing, unexpected = load_info
        print(f"[INFO] Missing keys: {len(missing)}")
        print(f"[INFO] Unexpected keys: {len(unexpected)}")

        if len(missing) > 0:
            print("[WARN] First missing keys:", missing[:10])

        if len(unexpected) > 0:
            print("[WARN] First unexpected keys:", unexpected[:10])

    model.eval()
    return model


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
            return False


def reconstruct_brain_mask(mask_skull: np.ndarray, is_water_only: bool) -> np.ndarray:
    skull = mask_skull > 0.5

    if is_water_only or skull.sum() == 0:
        return np.zeros_like(mask_skull, dtype=np.float32)

    filled = binary_fill_holes(skull)
    brain = np.logical_and(filled, np.logical_not(skull))

    return brain.astype(np.float32)


def infer_spacing_zyx(d):
    """
    Devuelve spacing en orden [z, y, x].
    Si el .npz solo tiene dx escalar, usa dx para los tres ejes.
    Si no hay dx, asume 1 mm.
    """
    if "spacing" in d:
        spacing = np.array(d["spacing"]).astype(np.float32).ravel()
        if spacing.size == 3:
            return spacing

    if "voxel_size" in d:
        spacing = np.array(d["voxel_size"]).astype(np.float32).ravel()
        if spacing.size == 3:
            return spacing

    if "dx" in d:
        dx = float(np.array(d["dx"]).item())
        return np.array([dx, dx, dx], dtype=np.float32)

    return np.array([1.0, 1.0, 1.0], dtype=np.float32)


def load_case(path: str):
    with np.load(path) as d:
        src = d["source_mask"].astype(np.float32)
        skull = d["mask_skull"].astype(np.float32)
        gt = d["p_max_norm"].astype(np.float32)

        if "is_water_only" in d:
            is_water_only = bool(np.array(d["is_water_only"]).item())
        elif "water_only" in d:
            is_water_only = bool(np.array(d["water_only"]).item())
        else:
            is_water_only = False

        spacing_zyx = infer_spacing_zyx(d)

    for name, arr in [
        ("source_mask", src),
        ("mask_skull", skull),
        ("p_max_norm", gt),
    ]:
        if arr.shape != EXPECTED_SHAPE:
            raise RuntimeError(
                f"Shape inv�lida en {os.path.basename(path)} | "
                f"{name}: {arr.shape} | esperado: {EXPECTED_SHAPE}"
            )

    brain = reconstruct_brain_mask(skull, is_water_only)

    return {
        "file": os.path.basename(path),
        "path": path,
        "source_mask": src,
        "mask_skull": skull,
        "gt": gt,
        "brain_mask": brain,
        "is_water_only": is_water_only,
        "spacing_zyx": spacing_zyx,
    }


# =========================================================
# PREDICTION
# =========================================================
@torch.no_grad()
def predict_case(model, case, device):
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
# METRICS
# =========================================================
def masked_values(arr, mask):
    m = mask > 0.5
    if m.sum() == 0:
        return np.array([], dtype=np.float32)
    return arr[m].astype(np.float32)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def get_peak_idx_in_mask(vol, mask=None):
    if mask is not None and mask.sum() > 0:
        masked = np.where(mask > 0.5, vol, -np.inf)
        flat_idx = int(np.argmax(masked))
    else:
        flat_idx = int(np.argmax(vol))

    return np.array(np.unravel_index(flat_idx, vol.shape), dtype=np.float32)


def peak_location_error_mm(gt, pred, brain_mask, spacing_zyx):
    gt_idx = get_peak_idx_in_mask(gt, brain_mask)
    pred_idx = get_peak_idx_in_mask(pred, brain_mask)

    delta_vox = pred_idx - gt_idx
    delta_mm = delta_vox * spacing_zyx

    return float(np.linalg.norm(delta_mm)), gt_idx, pred_idx


def peak_value_errors(gt, pred, brain_mask):
    gt_idx = get_peak_idx_in_mask(gt, brain_mask).astype(int)
    pred_idx = get_peak_idx_in_mask(pred, brain_mask).astype(int)

    gt_peak = float(gt[tuple(gt_idx)])
    pred_peak = float(pred[tuple(pred_idx)])

    abs_err = abs(pred_peak - gt_peak)
    rel_err = abs_err / (abs(gt_peak) + 1e-8)

    return gt_peak, pred_peak, float(abs_err), float(rel_err)


def ssim_slicewise(gt, pred, mask=None, crop_to_mask=False):
    """
    SSIM 3D aproximado como promedio de SSIM por slice axial.
    Para brain SSIM, aplica m�scara y opcionalmente crop alrededor del brain.
    """
    if not HAS_SKIMAGE:
        return np.nan

    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    data_min = float(min(gt.min(), pred.min()))
    data_max = float(max(gt.max(), pred.max()))
    data_range = data_max - data_min

    if data_range < 1e-8:
        data_range = 1.0

    scores = []

    Z, Y, X = gt.shape

    for z in range(Z):
        g = gt[z]
        p = pred[z]

        if mask is not None:
            m = mask[z] > 0.5

            if m.sum() < 20:
                continue

            g = np.where(m, g, 0.0)
            p = np.where(m, p, 0.0)

            if crop_to_mask:
                ys, xs = np.where(m)

                if len(ys) == 0 or len(xs) == 0:
                    continue

                pad = 5
                y0 = max(0, int(ys.min()) - pad)
                y1 = min(Y, int(ys.max()) + pad + 1)
                x0 = max(0, int(xs.min()) - pad)
                x1 = min(X, int(xs.max()) + pad + 1)

                g = g[y0:y1, x0:x1]
                p = p[y0:y1, x0:x1]

                if g.shape[0] < 7 or g.shape[1] < 7:
                    continue

        try:
            score = skimage_ssim(
                g,
                p,
                data_range=data_range,
            )
            scores.append(float(score))
        except Exception:
            continue

    if len(scores) == 0:
        return np.nan

    return float(np.mean(scores))


def compute_metrics_for_case(case, pred):
    gt = case["gt"]
    brain = case["brain_mask"]
    spacing_zyx = case["spacing_zyx"]

    # Global metrics
    mse_global = mse(gt, pred)
    mae_global = mae(gt, pred)
    ssim_global = ssim_slicewise(gt, pred, mask=None, crop_to_mask=False)

    # Brain metrics
    gt_brain = masked_values(gt, brain)
    pred_brain = masked_values(pred, brain)

    if gt_brain.size > 0:
        mse_brain = mse(gt_brain, pred_brain)
        mae_brain = mae(gt_brain, pred_brain)
        ssim_brain = ssim_slicewise(gt, pred, mask=brain, crop_to_mask=True)
    else:
        mse_brain = np.nan
        mae_brain = np.nan
        ssim_brain = np.nan

    peak_err_mm, gt_peak_idx, pred_peak_idx = peak_location_error_mm(
        gt,
        pred,
        brain,
        spacing_zyx,
    )

    gt_peak_val, pred_peak_val, peak_abs_err, peak_rel_err = peak_value_errors(
        gt,
        pred,
        brain,
    )

    row = {
        "file": case["file"],
        "is_water_only": int(case["is_water_only"]),

        "mse_global": mse_global,
        "mae_global": mae_global,
        "ssim_global": ssim_global,

        "mse_brain": mse_brain,
        "mae_brain": mae_brain,
        "ssim_brain": ssim_brain,

        "peak_loc_err_mm_brain": peak_err_mm,
        "peak_abs_err_brain": peak_abs_err,
        "peak_rel_err_brain": peak_rel_err,

        "gt_peak_val_brain": gt_peak_val,
        "pred_peak_val_brain": pred_peak_val,

        "gt_peak_z": int(gt_peak_idx[0]),
        "gt_peak_y": int(gt_peak_idx[1]),
        "gt_peak_x": int(gt_peak_idx[2]),

        "pred_peak_z": int(pred_peak_idx[0]),
        "pred_peak_y": int(pred_peak_idx[1]),
        "pred_peak_x": int(pred_peak_idx[2]),

        "spacing_z": float(spacing_zyx[0]),
        "spacing_y": float(spacing_zyx[1]),
        "spacing_x": float(spacing_zyx[2]),
    }

    return row


# =========================================================
# RANKING
# =========================================================
def minmax_higher_better(values):
    arr = np.array(values, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)

    valid = np.isfinite(arr)

    if valid.sum() == 0:
        return out

    v = arr[valid]
    vmin = float(v.min())
    vmax = float(v.max())

    if abs(vmax - vmin) < 1e-8:
        out[valid] = 1.0
    else:
        out[valid] = (v - vmin) / (vmax - vmin)

    return out


def minmax_lower_better(values):
    arr = np.array(values, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)

    valid = np.isfinite(arr)

    if valid.sum() == 0:
        return out

    v = arr[valid]
    vmin = float(v.min())
    vmax = float(v.max())

    if abs(vmax - vmin) < 1e-8:
        out[valid] = 1.0
    else:
        out[valid] = 1.0 - ((v - vmin) / (vmax - vmin))

    return out


def add_ranking(rows):
    ssim_global_vals = [r["ssim_global"] for r in rows]
    ssim_brain_vals = [r["ssim_brain"] for r in rows]
    peak_err_vals = [r["peak_loc_err_mm_brain"] for r in rows]

    ssim_global_score = minmax_higher_better(ssim_global_vals)
    ssim_brain_score = minmax_higher_better(ssim_brain_vals)
    peak_score = minmax_lower_better(peak_err_vals)

    for i, r in enumerate(rows):
        r["score_ssim_global"] = float(ssim_global_score[i])
        r["score_ssim_brain"] = float(ssim_brain_score[i])
        r["score_peak_loc"] = float(peak_score[i])

        r["rank_score"] = float(
            WEIGHTS["ssim_global"] * r["score_ssim_global"]
            + WEIGHTS["ssim_brain"] * r["score_ssim_brain"]
            + WEIGHTS["peak_loc_err_mm_brain"] * r["score_peak_loc"]
        )

    rows_sorted = sorted(rows, key=lambda x: x["rank_score"], reverse=True)

    for rank, r in enumerate(rows_sorted, start=1):
        r["rank"] = rank

    return rows_sorted


# =========================================================
# CSV HELPERS
# =========================================================
def save_csv(rows, path):
    if len(rows) == 0:
        raise RuntimeError("No hay filas para guardar.")

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            writer.writerow(r)

    print(f"[OK] CSV guardado en: {path}")


def print_ranking_summary(rows_sorted, top_k=10):
    print("\n======================================")
    print("TOP CASES")
    print("======================================")

    for r in rows_sorted[:top_k]:
        print(
            f"#{r['rank']:03d} | {r['file']} | "
            f"score={r['rank_score']:.4f} | "
            f"SSIM global={r['ssim_global']:.4f} | "
            f"SSIM brain={r['ssim_brain']:.4f} | "
            f"peak err={r['peak_loc_err_mm_brain']:.2f} mm"
        )

    print("\n======================================")
    print("WORST CASES")
    print("======================================")

    for r in rows_sorted[-top_k:][::-1]:
        print(
            f"#{r['rank']:03d} | {r['file']} | "
            f"score={r['rank_score']:.4f} | "
            f"SSIM global={r['ssim_global']:.4f} | "
            f"SSIM brain={r['ssim_brain']:.4f} | "
            f"peak err={r['peak_loc_err_mm_brain']:.2f} mm"
        )


# =========================================================
# MAIN
# =========================================================
def main():
    print("======================================")
    print("Per-case PTH evaluation and ranking")
    print("======================================")
    print("TEST_DIR:", TEST_DIR)
    print("CKPT_PATH:", CKPT_PATH)
    print("OUT_DIR:", OUT_DIR)
    print("DEVICE:", DEVICE)
    print("HAS_SKIMAGE:", HAS_SKIMAGE)

    if not HAS_SKIMAGE:
        print(
            "[WARN] skimage no est� instalado. "
            "SSIM saldr� como NaN. Instala scikit-image si necesitas SSIM."
        )

    model = load_model(CKPT_PATH, DEVICE)

    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.npz")))

    if len(files) == 0:
        raise RuntimeError(f"No se encontraron archivos .npz en: {TEST_DIR}")

    rows = []

    for idx, path in enumerate(files, start=1):
        is_water_only = read_is_water_only(path)

        if is_water_only and not EVALUATE_WATER_ONLY:
            continue

        print(f"[{idx}/{len(files)}] Evaluating {os.path.basename(path)}")

        case = load_case(path)
        pred = predict_case(model, case, DEVICE)

        row = compute_metrics_for_case(case, pred)
        rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("No se evalu� ning�n caso.")

    per_case_path = os.path.join(OUT_DIR, PER_CASE_CSV)
    save_csv(rows, per_case_path)

    rows_ranked = add_ranking(rows)

    ranking_path = os.path.join(OUT_DIR, RANKING_CSV)
    save_csv(rows_ranked, ranking_path)

    print_ranking_summary(rows_ranked, top_k=10)

    print("\n======================================")
    print("DONE")
    print("======================================")
    print("Per-case metrics:", per_case_path)
    print("Ranking:", ranking_path)


if __name__ == "__main__":
    main()