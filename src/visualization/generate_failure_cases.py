from src.models.resunet_3d import ResUNet3D_HQ
import torch
from scipy.ndimage import binary_fill_holes, binary_dilation
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")


# =========================================================
# CONFIG
# =========================================================
TEST_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test"

CKPT_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_TFG/epoch_220.pth"

OUT_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/tablas/failure_cases_manual_peaks"

MODEL_LABEL = "cGAN Full Loss 220e"

EXPECTED_SHAPE = (128, 128, 128)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# PUT YOUR TWO FAILURE CASES HERE
# =========================================================
FORCE_FILES = [
    "sample_0397.npz",
    "sample_0861.npz",
]


PLANES = ["sagittal", "coronal", "axial"]

SAVE_NAME = "failure_cases_manual_2cases_vertical_peak_dots_compact.png"

DPI = 320


# =========================================================
# VISUALIZATION
# =========================================================
CMAP_FIELD = "jet"
CMAP_GEOMETRY = "gray"

# Compact vertical layout
FIGSIZE = (7.0, 13.2)

WSPACE = 0.010
HSPACE = 0.018

LEFT_MARGIN = 0.145
RIGHT_MARGIN = 0.995
TOP_MARGIN = 0.965
BOTTOM_MARGIN = 0.015

TOP_LABEL_FONTSIZE = 14
PLANE_LABEL_FONTSIZE = 10
LEFT_LABEL_FONTSIZE = 11

CASE_LABELS = ["(a)", "(b)"]

SHOW_PEAK_MARKERS = True

# Peak marker colors
GT_COLOR = "white"
PRED_COLOR = "magenta"

# Small dots
GT_DOT_SIZE = 14
PRED_DOT_SIZE = 8
DOT_HALO_SIZE = 28
DOT_ALPHA = 1.0

# Geometry slice:
# "peak"  -> geometry uses slices through the GT peak
# "skull" -> geometry uses slices through the skull center
GEOMETRY_SLICE_MODE = "peak"

# Transducer visualization
TRANSDUCER_DILATION_ITERS = 2
TRANSDUCER_THRESHOLD = 0.5
TRANSDUCER_CONTOUR_COLOR = "yellow"
TRANSDUCER_CONTOUR_WIDTH = 1.4
TRANSDUCER_CONTOUR_ALPHA = 0.95

# "full_projection" -> projects the full transducer onto each plane
# "slice"           -> shows only the exact slice of the transducer
TRANSDUCER_VIS_MODE = "full_projection"

MODEL_KWARGS = dict(
    in_ch=2,
    out_ch=1,
    out_positive=True,
)

os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# DATA HELPERS
# =========================================================
def normalize_for_display(arr):
    arr = arr.astype(np.float32)
    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))

    if amax <= amin:
        return np.zeros_like(arr, dtype=np.float32)

    return (arr - amin) / (amax - amin)


def read_is_water_only(path):
    with np.load(path) as d:
        if "is_water_only" in d:
            return bool(np.array(d["is_water_only"]).item())
        if "water_only" in d:
            return bool(np.array(d["water_only"]).item())

    return False


def get_dx_mm_from_npz(d):
    """
    Tries to recover dx in mm from the .npz.
    If unavailable, assumes 1 mm.
    """
    for key in ["dx", "dx_mm", "spacing", "voxel_size", "voxel_size_mm"]:
        if key in d:
            val = np.array(d[key]).astype(float).ravel()
            if len(val) > 0 and np.isfinite(val[0]) and val[0] > 0:
                return float(val[0])
    return 1.0


def reconstruct_brain_mask(mask_skull, is_water_only):
    skull = mask_skull > 0.5

    if is_water_only or skull.sum() == 0:
        return np.zeros_like(mask_skull, dtype=np.float32)

    filled = binary_fill_holes(skull)
    brain = np.logical_and(filled, np.logical_not(skull))

    return brain.astype(np.float32)


def choose_anatomy_volume(d, skull):
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


def load_case(path, expected_shape=(128, 128, 128)):
    with np.load(path) as d:
        src = d["source_mask"].astype(np.float32)
        skull = d["mask_skull"].astype(np.float32)

        if "p_max_norm" not in d:
            raise RuntimeError(
                f"{os.path.basename(path)} does not contain p_max_norm")

        gt = d["p_max_norm"].astype(np.float32)
        anatomy, anatomy_key = choose_anatomy_volume(d, skull)

        if "is_water_only" in d:
            is_water_only = bool(np.array(d["is_water_only"]).item())
        elif "water_only" in d:
            is_water_only = bool(np.array(d["water_only"]).item())
        else:
            is_water_only = False

        dx_mm = get_dx_mm_from_npz(d)

    for name, arr in [
        ("source_mask", src),
        ("mask_skull", skull),
        ("p_max_norm", gt),
        ("anatomy", anatomy),
    ]:
        if arr.shape != expected_shape:
            raise RuntimeError(
                f"Invalid shape in {os.path.basename(path)} | "
                f"{name}: {arr.shape} | expected: {expected_shape}"
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
        "dx_mm": dx_mm,
    }


def get_peak_index(vol, brain_mask=None):
    if brain_mask is not None and brain_mask.sum() > 0:
        masked = np.where(brain_mask > 0.5, vol, -np.inf)
        flat_idx = int(np.argmax(masked))
    else:
        flat_idx = int(np.argmax(vol))

    return np.unravel_index(flat_idx, vol.shape)


def get_skull_center(mask_skull):
    pts = np.argwhere(mask_skull > 0.5)

    if len(pts) == 0:
        zc, yc, xc = np.array(mask_skull.shape) // 2
    else:
        zc, yc, xc = np.round(pts.mean(axis=0)).astype(int)

    return int(zc), int(yc), int(xc)


def extract_plane(vol, z, y, x, plane):
    if plane == "axial":
        return vol[z, :, :]        # Y-X
    elif plane == "coronal":
        return vol[:, y, :]        # Z-X
    elif plane == "sagittal":
        return vol[:, :, x]        # Z-Y
    else:
        raise ValueError(f"Unknown plane: {plane}")


def project_volume_to_plane(vol, plane):
    if plane == "axial":
        return np.max(vol, axis=0)      # Y-X
    elif plane == "coronal":
        return np.max(vol, axis=1)      # Z-X
    elif plane == "sagittal":
        return np.max(vol, axis=2)      # Z-Y
    else:
        raise ValueError(f"Unknown plane: {plane}")


def peak_marker_coords(z, y, x, plane):
    if plane == "axial":
        return x, y
    elif plane == "coronal":
        return x, z
    elif plane == "sagittal":
        return y, z
    else:
        raise ValueError(f"Unknown plane: {plane}")


def draw_peak_dot(ax, peak, plane, color, size, zorder=25):
    """
    Draws a small colored dot with a black halo.
    """
    if not SHOW_PEAK_MARKERS:
        return

    z, y, x = peak
    px, py = peak_marker_coords(z, y, x, plane)

    # Black halo
    ax.scatter(
        [px],
        [py],
        s=DOT_HALO_SIZE,
        c="black",
        marker="o",
        linewidths=0,
        alpha=DOT_ALPHA,
        zorder=zorder,
    )

    # Main dot
    ax.scatter(
        [px],
        [py],
        s=size,
        c=color,
        marker="o",
        linewidths=0,
        alpha=DOT_ALPHA,
        zorder=zorder + 1,
    )


def draw_gt_and_pred_peaks(ax, gt_peak, pred_peak, plane):
    """
    Draws GT peak first and predicted peak second.
    If they overlap, the magenta dot appears inside/over the white dot.
    """
    draw_peak_dot(
        ax=ax,
        peak=gt_peak,
        plane=plane,
        color=GT_COLOR,
        size=GT_DOT_SIZE,
        zorder=25,
    )

    draw_peak_dot(
        ax=ax,
        peak=pred_peak,
        plane=plane,
        color=PRED_COLOR,
        size=PRED_DOT_SIZE,
        zorder=27,
    )


def make_transducer_vis_mask(src_3d, dilation_iters=2):
    src_bin = src_3d > 0.5
    src_vis = binary_dilation(src_bin, iterations=dilation_iters)
    return src_vis.astype(np.float32)


def draw_transducer_contour(ax, src_slice):
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

    raise RuntimeError("Could not find model state_dict inside checkpoint.")


def load_model(ckpt_path, device):
    print("\n======================================")
    print("Loading model")
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

    print("[INFO] Missing keys:", len(missing))
    print("[INFO] Unexpected keys:", len(unexpected))

    if len(missing) > 0:
        print("[WARN] First missing keys:", missing[:5])

    if len(unexpected) > 0:
        print("[WARN] First unexpected keys:", unexpected[:5])

    model.eval()
    return model


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
# MANUAL CASE SELECTION
# =========================================================
def resolve_manual_files(test_dir, force_files):
    all_npz = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    available = {os.path.basename(p): p for p in all_npz}

    if len(force_files) != 2:
        raise RuntimeError(
            f"This script expects exactly 2 samples in FORCE_FILES. "
            f"Currently: {len(force_files)}"
        )

    selected_paths = []

    for fname in force_files:
        base = os.path.basename(fname)

        if base in available:
            selected_paths.append(available[base])
            continue

        if not base.endswith(".npz"):
            base_npz = base + ".npz"
            if base_npz in available:
                selected_paths.append(available[base_npz])
                continue

        raise RuntimeError(
            f"Sample not found in TEST_DIR: {fname}\n"
            f"Check the name or path."
        )

    return selected_paths


# =========================================================
# PLOT
# =========================================================
def plot_failure_cases_2cases_3planes(cases, save_path):
    """
    Vertical compact layout:
        Columns: Sagittal, Coronal, Axial

        Case (a):
            Rows: Prediction, Ground truth, Geometry

        Case (b):
            Rows: Prediction, Ground truth, Geometry
    """
    if len(cases) != 2:
        raise RuntimeError(
            f"This function expects exactly 2 cases. Received: {len(cases)}")

    n_rows = 6
    n_cols = len(PLANES)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=FIGSIZE,
    )

    axes = np.asarray(axes)

    row_labels = [
        "Prediction",
        "Ground truth",
        "Geometry",
    ]

    plane_titles = {
        "sagittal": "Sagittal",
        "coronal": "Coronal",
        "axial": "Axial",
    }

    for case_idx, case in enumerate(cases):
        gt = case["gt"]
        pred = case["pred"]

        src = case["source_mask"]
        skull = case["mask_skull"]
        anatomy = case["anatomy"]
        brain = case["brain_mask"]

        gt_peak = get_peak_index(gt, brain)
        pred_peak = get_peak_index(pred, brain)

        z_peak, y_peak, x_peak = gt_peak

        if GEOMETRY_SLICE_MODE == "peak":
            z_geom, y_geom, x_geom = z_peak, y_peak, x_peak
        elif GEOMETRY_SLICE_MODE == "skull":
            z_geom, y_geom, x_geom = get_skull_center(skull)
        else:
            raise ValueError(
                f"Unknown GEOMETRY_SLICE_MODE: {GEOMETRY_SLICE_MODE}")

        field_vmin = 0.0
        field_vmax = float(max(gt.max(), pred.max()))
        if field_vmax <= 0:
            field_vmax = 1.0

        src_vis_3d = make_transducer_vis_mask(
            src,
            dilation_iters=TRANSDUCER_DILATION_ITERS,
        )

        row_offset = case_idx * 3

        for plane_idx, plane in enumerate(PLANES):
            col = plane_idx

            # ----------------------------
            # Row 0 of case block: Prediction
            # ----------------------------
            ax = axes[row_offset + 0, col]

            pred_slice = extract_plane(pred, z_peak, y_peak, x_peak, plane)

            ax.imshow(
                pred_slice,
                cmap=CMAP_FIELD,
                vmin=field_vmin,
                vmax=field_vmax,
                origin="lower",
                interpolation="nearest",
            )

            draw_gt_and_pred_peaks(ax, gt_peak, pred_peak, plane)

            # ----------------------------
            # Row 1 of case block: Ground truth
            # ----------------------------
            ax = axes[row_offset + 1, col]

            gt_slice = extract_plane(gt, z_peak, y_peak, x_peak, plane)

            ax.imshow(
                gt_slice,
                cmap=CMAP_FIELD,
                vmin=field_vmin,
                vmax=field_vmax,
                origin="lower",
                interpolation="nearest",
            )

            draw_peak_dot(
                ax=ax,
                peak=gt_peak,
                plane=plane,
                color=GT_COLOR,
                size=GT_DOT_SIZE,
                zorder=25,
            )

            # ----------------------------
            # Row 2 of case block: Geometry
            # ----------------------------
            ax = axes[row_offset + 2, col]

            anatomy_slice = extract_plane(
                anatomy, z_geom, y_geom, x_geom, plane)
            anatomy_slice = normalize_for_display(anatomy_slice)

            ax.imshow(
                anatomy_slice,
                cmap=CMAP_GEOMETRY,
                origin="lower",
                interpolation="nearest",
            )

            if TRANSDUCER_VIS_MODE == "full_projection":
                src_vis_slice = project_volume_to_plane(src_vis_3d, plane)
            else:
                src_vis_slice = extract_plane(
                    src_vis_3d, z_geom, y_geom, x_geom, plane)

            draw_transducer_contour(ax, src_vis_slice)

            # Plane titles only on the first case block
            if case_idx == 0:
                axes[row_offset + 0, col].set_title(
                    plane_titles.get(plane, plane),
                    fontsize=PLANE_LABEL_FONTSIZE,
                    fontweight="semibold",
                    pad=3,
                )

        # Row labels for each case block
        for r, label in enumerate(row_labels):
            axes[row_offset + r, 0].set_ylabel(
                label,
                fontsize=LEFT_LABEL_FONTSIZE,
                fontweight="bold",
                rotation=90,
                labelpad=8,
            )

    # Axis formatting
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].set_aspect("equal")

    plt.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        top=TOP_MARGIN,
        bottom=BOTTOM_MARGIN,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    # Case labels placed on the left side of each block
    for case_idx in range(2):
        row_offset = case_idx * 3

        pos_top = axes[row_offset, 0].get_position()
        pos_bottom = axes[row_offset + 2, 0].get_position()

        y_center = 0.5 * (pos_top.y1 + pos_bottom.y0)
        x_left = pos_top.x0 - 0.090

        fig.text(
            x_left,
            y_center,
            CASE_LABELS[case_idx],
            ha="center",
            va="center",
            fontsize=TOP_LABEL_FONTSIZE,
            fontweight="bold",
            rotation=90,
        )

    fig.savefig(
        save_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )

    plt.close(fig)

    print("\n[OK] Compact vertical failure case figure saved to:")
    print(save_path)


# =========================================================
# RUN
# =========================================================
def main():
    print("======================================================")
    print("Manual failure case visualization")
    print("2 selected samples | compact vertical layout | 3 planes | peak dots")
    print("======================================================")
    print("TEST_DIR:", TEST_DIR)
    print("CKPT_PATH:", CKPT_PATH)
    print("OUT_DIR:", OUT_DIR)
    print("DEVICE:", DEVICE)
    print("FORCE_FILES:", FORCE_FILES)

    selected_paths = resolve_manual_files(TEST_DIR, FORCE_FILES)

    print("\nSelected samples:")
    for p in selected_paths:
        print(" -", os.path.basename(p))

    model = load_model(CKPT_PATH, DEVICE)

    cases = []

    print("\nRunning predictions...")

    for path in selected_paths:
        case = load_case(path, expected_shape=EXPECTED_SHAPE)

        if case["is_water_only"]:
            print(f"[WARN] {case['file_name']} is water-only.")

        pred = predict_case(model, case, DEVICE)

        case["pred"] = pred

        gt_peak = get_peak_index(case["gt"], case["brain_mask"])
        pred_peak = get_peak_index(pred, case["brain_mask"])

        peak_dist_vox = float(np.linalg.norm(
            np.array(gt_peak) - np.array(pred_peak)))
        peak_dist_mm = peak_dist_vox * float(case["dx_mm"])

        print(
            f"[INFO] {case['file_name']} | "
            f"GT max={case['gt'].max():.4f} | "
            f"Pred max={pred.max():.4f} | "
            f"GT peak={gt_peak} | "
            f"Pred peak={pred_peak} | "
            f"Peak dist={peak_dist_vox:.2f} vox / {peak_dist_mm:.2f} mm"
        )

        cases.append(case)

    save_path = os.path.join(OUT_DIR, SAVE_NAME)

    plot_failure_cases_2cases_3planes(
        cases=cases,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
