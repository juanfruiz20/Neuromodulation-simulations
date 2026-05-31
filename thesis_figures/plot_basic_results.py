import torch.nn.functional as F
import torch.nn as nn
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

CKPT_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_resunet3d_hq_3L_l1_fulldata_100epochs/epoch_050.pth"

OUT_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/tablas/GT_vs_PRED_examples_3L_hor"

EXPECTED_SHAPE = (128, 128, 128)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CASES = 2

# Si quieres forzar casos concretos, pon aqu� los nombres exactos.

FORCE_FILES = [
    "sample_0981.npz",
    "sample_0327.npz",
]
# FORCE_FILES = []

PLANES = ["sagittal", "coronal", "axial"]

CMAP_ANATOMY = "gray"
CMAP_INTENSITY = "jet"

DPI = 320
SAVE_NAME = "gt_vs_prediction_2_cases_3_planes_resunet3d_hq_3L.png"

SHOW_PEAK_MARKER = False

# =========================================================
# Transducer contour visualization
# =========================================================
TRANSDUCER_DILATION_ITERS = 2
TRANSDUCER_CONTOUR_COLOR = "yellow"
TRANSDUCER_CONTOUR_WIDTH = 1.6
TRANSDUCER_CONTOUR_ALPHA = 0.95
TRANSDUCER_THRESHOLD = 0.5

# Full projection helps the transducer contour appear in all three views.
TRANSDUCER_VIS_MODE = "full_projection"

# =========================================================
# Diverse case selection
# =========================================================
DIVERSE_SELECTION_SEED_INDEX = 11

os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# REDUCED ResUNet3D_HQ: 3 LEVELS
# =========================================================
def norm3d(ch: int, kind: str = "group", groups: int = 8):
    if kind == "instance":
        return nn.InstanceNorm3d(ch, affine=True)

    g = min(groups, ch)
    while g > 1 and ch % g != 0:
        g -= 1

    return nn.GroupNorm(num_groups=g, num_channels=ch)


class SEBlock3D(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()

        r = max(1, ch // reduction)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(ch, r, kernel_size=1)
        self.fc2 = nn.Conv3d(r, ch, kernel_size=1)

    def forward(self, x):
        s = self.pool(x)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))

        return x * s


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.n1 = norm3d(out_ch, kind=norm_kind)

        self.conv2 = nn.Conv3d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.n2 = norm3d(out_ch, kind=norm_kind)

        self.act = nn.SiLU(inplace=True)

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=1,
                bias=False,
            )

        self.se = SEBlock3D(out_ch, reduction=8) if use_se else nn.Identity()

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)

        x = self.act(self.n1(self.conv1(x)))
        x = self.n2(self.conv2(x))
        x = self.se(x)

        x = self.act(x + identity)

        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()

        self.down = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.n = norm3d(out_ch, kind=norm_kind)
        self.act = nn.SiLU(inplace=True)

        self.block = ResBlock3D(
            out_ch,
            out_ch,
            norm_kind=norm_kind,
            use_se=use_se,
        )

    def forward(self, x):
        x = self.act(self.n(self.down(x)))
        x = self.block(x)

        return x


class UpConcat(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, norm_kind="group", use_se=True):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=False,
        )

        self.conv_up = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=1,
            bias=False,
        )

        self.block = ResBlock3D(
            out_ch + skip_ch,
            out_ch,
            norm_kind=norm_kind,
            use_se=use_se,
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv_up(x)

        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(
                x,
                size=skip.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )

        x = torch.cat([skip, x], dim=1)
        x = self.block(x)

        return x


class ResUNet3D_HQ_3L(nn.Module):
    """
    Reduced-depth ResUNet3D_HQ with 3 downsampling levels.

    Input:
      [B, 2, 128, 128, 128]

    Output:
      [B, 1, 128, 128, 128]
    """

    def __init__(
        self,
        in_ch=2,
        out_ch=1,
        base=16,
        norm_kind="group",
        use_se=True,
        out_positive=True,
    ):
        super().__init__()

        self.out_positive = out_positive
        self.out_act = nn.Softplus() if out_positive else nn.Identity()

        # Encoder
        self.stem = ResBlock3D(
            in_ch,
            base,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # 128^3

        self.d1 = Down(
            base,
            base * 2,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # 64^3

        self.d2 = Down(
            base * 2,
            base * 4,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # 32^3

        self.d3 = Down(
            base * 4,
            base * 8,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # 16^3

        # Bottleneck
        self.mid = ResBlock3D(
            base * 8,
            base * 16,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # 16^3

        # Decoder
        self.u3 = UpConcat(
            base * 16,
            base * 4,
            base * 4,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # -> 32^3

        self.u2 = UpConcat(
            base * 4,
            base * 2,
            base * 2,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # -> 64^3

        self.u1 = UpConcat(
            base * 2,
            base,
            base,
            norm_kind=norm_kind,
            use_se=use_se,
        )  # -> 128^3

        self.head = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)   # base, 128^3
        s1 = self.d1(s0)    # 2b, 64^3
        s2 = self.d2(s1)    # 4b, 32^3
        s3 = self.d3(s2)    # 8b, 16^3

        m = self.mid(s3)    # 16b, 16^3

        x = self.u3(m, s2)  # 4b, 32^3
        x = self.u2(x, s1)  # 2b, 64^3
        x = self.u1(x, s0)  # b, 128^3

        x = self.head(x)
        x = self.out_act(x)

        return x


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


def extract_generator_state_dict(ckpt):
    """
    Extrae el state_dict desde distintos formatos de checkpoint.
    Para esta U-Net reducida normalmente ser� ckpt["model"].
    """
    if not isinstance(ckpt, dict):
        return ckpt

    possible_keys = [
        "model",
        "model_state_dict",
        "state_dict",
        "generator_state_dict",
        "gen_state_dict",
        "G_state_dict",
        "netG_state_dict",
        "generator",
        "gen",
        "G",
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


def load_generator_model(ckpt_path: str, device: str):
    print("\n======================================")
    print("Loading ResUNet3D_HQ_3L model")
    print("======================================")
    print("CKPT_PATH:", ckpt_path)
    print("DEVICE:", device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint inv�lido: {ckpt_path}")

    cfg = ckpt.get("config", {})

    model_type = cfg.get("model_type", "ResUNet3D_HQ_3L")
    base = int(cfg.get("base", 16))
    norm_kind = cfg.get("norm_kind", "group")
    use_se = bool(cfg.get("use_se", True))
    out_positive = bool(cfg.get("out_positive", True))

    print("Checkpoint config:")
    print(f"  model_type={model_type}")
    print(f"  base={base}")
    print(f"  norm_kind={norm_kind}")
    print(f"  use_se={use_se}")
    print(f"  out_positive={out_positive}")

    if model_type != "ResUNet3D_HQ_3L":
        print(
            f"[WARN] El checkpoint dice model_type={model_type}, "
            "pero este script est� preparado para ResUNet3D_HQ_3L."
        )

    model = ResUNet3D_HQ_3L(
        in_ch=2,
        out_ch=1,
        base=base,
        norm_kind=norm_kind,
        use_se=use_se,
        out_positive=out_positive,
    ).to(device)

    state_dict = extract_generator_state_dict(ckpt)

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

    print(f"[INFO] Missing keys: {len(missing)}")
    print(f"[INFO] Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[WARN] First missing keys:", missing[:10])

    if len(unexpected) > 0:
        print("[WARN] First unexpected keys:", unexpected[:10])

    if len(missing) > 0 or len(unexpected) > 0:
        print(
            "[WARN] El modelo carg� con strict=False. "
            "Si hay muchas missing/unexpected keys, revisa que el checkpoint "
            "corresponda exactamente a ResUNet3D_HQ_3L."
        )

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

    # Para visualizaci�n estable
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


def farthest_point_sampling(features, n_select=2, seed_index=0):
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


def select_diverse_non_water_cases(data_dir: str, n_cases=2):
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
# PLOT GT VS PRED
# =========================================================
def plot_gt_vs_prediction(cases, save_path):
    plane_titles = {
        "axial": "Axial",
        "coronal": "Coronal",
        "sagittal": "Sagittal",
    }

    n_cases = len(cases)
    n_planes = len(PLANES)

    # Figura más alta y menos ancha para que las imágenes cuadradas llenen el espacio
    fig = plt.figure(figsize=(9.2, 5.25))

    # Grid externo: un bloque por caso
    outer = fig.add_gridspec(
        nrows=1,
        ncols=n_cases,
        left=0.043,
        right=0.997,
        bottom=0.055,
        top=0.855,
        wspace=0.018,   # espacio entre Case 1 y Case 2
    )

    all_axes = []

    row_geom = 0
    row_gt = 1
    row_pred = 2

    for case_idx, case in enumerate(cases):
        # Grid interno: 3 filas x 3 planos por cada caso
        inner = outer[case_idx].subgridspec(
            nrows=3,
            ncols=n_planes,
            wspace=0.002,   # MUY pegado horizontalmente dentro del caso
            hspace=0.002,   # MUY pegado verticalmente
        )

        case_axes = np.empty((3, n_planes), dtype=object)

        gt = case["gt"]
        pred = case["pred"]
        src = case["source_mask"]
        skull = case["mask_skull"]
        anatomy = case["anatomy"]
        brain = case["brain_mask"]

        z_peak, y_peak, x_peak = get_peak_index(gt, brain)
        z_skull, y_skull, x_skull = get_skull_center(skull)

        case_vmin = 0.0
        case_vmax = float(max(gt.max(), pred.max()))
        if case_vmax <= 0:
            case_vmax = 1.0

        src_vis_3d = make_transducer_vis_mask(
            src,
            dilation_iters=TRANSDUCER_DILATION_ITERS,
        )

        for plane_idx, plane in enumerate(PLANES):
            ax_geom = fig.add_subplot(inner[row_geom, plane_idx])
            ax_gt = fig.add_subplot(inner[row_gt, plane_idx])
            ax_pred = fig.add_subplot(inner[row_pred, plane_idx])

            case_axes[row_geom, plane_idx] = ax_geom
            case_axes[row_gt, plane_idx] = ax_gt
            case_axes[row_pred, plane_idx] = ax_pred

            # =================================================
            # GEOMETRY
            # =================================================
            anatomy_slice = extract_plane(
                anatomy,
                z_skull,
                y_skull,
                x_skull,
                plane,
            )
            anatomy_slice = normalize_for_display(anatomy_slice)

            ax_geom.imshow(
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

            draw_transducer_contour(ax_geom, src_vis_slice)

            # =================================================
            # GROUND TRUTH
            # =================================================
            gt_slice = extract_plane(
                gt,
                z_peak,
                y_peak,
                x_peak,
                plane,
            )

            ax_gt.imshow(
                gt_slice,
                cmap=CMAP_INTENSITY,
                vmin=case_vmin,
                vmax=case_vmax,
                origin="lower",
                interpolation="nearest",
            )

            # =================================================
            # PREDICTION
            # =================================================
            pred_slice = extract_plane(
                pred,
                z_peak,
                y_peak,
                x_peak,
                plane,
            )

            ax_pred.imshow(
                pred_slice,
                cmap=CMAP_INTENSITY,
                vmin=case_vmin,
                vmax=case_vmax,
                origin="lower",
                interpolation="nearest",
            )

            if SHOW_PEAK_MARKER:
                px, py = peak_marker_coords(z_peak, y_peak, x_peak, plane)
                for ax in [ax_gt, ax_pred]:
                    ax.plot(
                        px,
                        py,
                        marker="+",
                        color="white",
                        markersize=5,
                        markeredgewidth=0.9,
                    )

            # Títulos de plano pequeños y pegados
            ax_geom.set_title(
                plane_titles[plane],
                fontsize=8.1,
                fontweight="semibold",
                pad=1.0,
            )

            for ax in [ax_geom, ax_gt, ax_pred]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal", adjustable="box")

                for spine in ax.spines.values():
                    spine.set_color("white")
                    spine.set_linewidth(0.35)

        all_axes.append(case_axes)

    fig.canvas.draw()

    # =====================================================
    # LABELS DE FILA A LA IZQUIERDA, MÁS PEGADOS
    # =====================================================
    first_case_axes = all_axes[0]

    row_labels = {
        row_geom: "Geometry",
        row_gt: "Ground truth",
        row_pred: "Prediction",
    }

    for row_idx, label in row_labels.items():
        pos = first_case_axes[row_idx, 0].get_position()
        y_center = 0.5 * (pos.y0 + pos.y1)

        fig.text(
            pos.x0 - 0.008,
            y_center,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=7.5,
            fontweight="semibold",
        )

    # =====================================================
    # CASE LABELS ARRIBA, SIN SUPERPONERSE
    # =====================================================
    for case_idx in range(n_cases):
        case_axes = all_axes[case_idx]

        pos_left = case_axes[0, 0].get_position()
        pos_right = case_axes[0, n_planes - 1].get_position()

        x_center = 0.5 * (pos_left.x0 + pos_right.x1)
        y_top = pos_left.y1 + 0.060

        fig.text(
            x_center,
            y_top,
            f"Case {case_idx + 1}",
            va="center",
            ha="center",
            fontsize=9.3,
            fontweight="bold",
        )

    # =====================================================
    # SEPARADOR ENTRE CASOS, MUY SUTIL
    # =====================================================
    if n_cases > 1:
        for case_idx in range(1, n_cases):
            pos = all_axes[case_idx][0, 0].get_position()
            x_sep = pos.x0 - 0.006

            fig.lines.append(
                plt.Line2D(
                    [x_sep, x_sep],
                    [0.055, 0.855],
                    transform=fig.transFigure,
                    color="0.86",
                    linewidth=0.45,
                    alpha=0.85,
                )
            )

    fig.savefig(save_path, dpi=DPI, bbox_inches="tight",
                facecolor="white", pad_inches=0.015)
    plt.close(fig)

    print(f"\n[OK] Figura guardada en:\n{save_path}")

# =========================================================
# RUN
# =========================================================


def main():
    print("===============================================")
    print("GT vs Prediction visualization")
    print("2 cases | geometry + GT + prediction | 3 planes")
    print("Model: ResUNet3D_HQ_3L")
    print("===============================================")
    print("DATA_DIR:", DATA_DIR)
    print("CKPT_PATH:", CKPT_PATH)
    print("OUT_DIR:", OUT_DIR)
    print("DEVICE:", DEVICE)

    model = load_generator_model(CKPT_PATH, DEVICE)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))

    if len(all_files) == 0:
        raise RuntimeError(f"No se encontraron archivos .npz en: {DATA_DIR}")

    available = {os.path.basename(p): p for p in all_files}

    if FORCE_FILES:
        selected_cases = []

        for fname in FORCE_FILES:
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

    print("\nCasos no-water seleccionados:")
    for c in selected_cases:
        print(" -", c["file_name"])

    print("\nRunning model predictions...")
    for case in selected_cases:
        pred = predict_case(model, case, DEVICE)
        case["pred"] = pred

        print(
            f"[INFO] {case['file_name']} | "
            f"GT max={case['gt'].max():.4f} | "
            f"Pred max={pred.max():.4f}"
        )

    save_path = os.path.join(OUT_DIR, SAVE_NAME)

    plot_gt_vs_prediction(
        cases=selected_cases,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
