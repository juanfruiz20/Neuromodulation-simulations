import os
import gc
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_fill_holes

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions

# =========================================================
# IMPORTA TU MODELO
# =========================================================
# Ajusta este import si tu archivo o clase tiene otro nombre.
from src.modelos.ResUnet3D import ResUNet3D_HQ


# =========================================================
# CONFIG
# =========================================================

TEST_DIR = Path("/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test")

MODEL1_CKPT = Path("/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_expDexpB/epoch_030.pth")
MODEL2_CKPT = Path("/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_TFG/epoch_220.pth")

MODEL1_NAME = "U-Net Full Loss"
MODEL2_NAME = "cGAN Full Loss 220e"

OUT_CSV = Path("timing_kwave_model1_model2.csv")
OUT_SUMMARY_CSV = Path("timing_summary.csv")

N_CASES = 10
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Para modelos r�pidos, repetir varias veces da tiempos m�s estables.
N_MODEL_REPEATS = 5
N_WARMUP = 5

USE_AMP = False

# Par�metros iguales a tu generaci�n del dataset
N_CYCLES = 10
CFL = 0.1
PML_SIZE = 10

PROPS = {
    "water": (1482.0, 994.0, 0.0126),
    "skull": (2800.0, 1850.0, 15.0),
    "brain": (1546.0, 1046.0, 0.5),
}


# =========================================================
# UTILIDADES GENERALES
# =========================================================

def synchronize():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def select_test_cases(test_dir, n_cases=10, seed=42):
    files = sorted(test_dir.glob("*.npz"))

    if len(files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos .npz en: {test_dir}")

    random.seed(seed)

    if len(files) <= n_cases:
        selected = files
    else:
        selected = random.sample(files, n_cases)

    return sorted(selected)


def get_dt_Nt_or_fallback(kgrid, dx, c_max, cfl, t_end):
    dt = float(cfl * dx / c_max)
    Nt = int(np.ceil(t_end / dt)) + 1
    return dt, Nt, np.arange(Nt) * dt


def make_tone_burst(dt, Nt, f0, n_cycles, amp):
    t = np.arange(Nt) * dt
    n_on = int(np.round(n_cycles / f0 / dt))
    sig = np.zeros(Nt, dtype=np.float32)

    if n_on > 1:
        sig[:n_on] = amp * np.sin(2 * np.pi * f0 * t[:n_on]) * np.hanning(n_on)

    return sig


def make_source_compatible_for_time_signal(source, p_mask_u8):
    source.mask = p_mask_u8
    source.p_mask = p_mask_u8

    if hasattr(source, "p0"):
        try:
            delattr(source, "p0")
        except Exception:
            pass

    return source


def safe_reshape_pmax(p_max, Nx, Ny, Nz):
    try:
        return p_max.reshape((Nx, Ny, Nz), order="F")
    except Exception:
        return p_max.reshape((Nx, Ny, Nz), order="C")


# =========================================================
# RECONSTRUIR BRAIN MASK
# =========================================================

def reconstruct_brain_mask_from_skull(mask_skull):
    """
    En tu dataset no se guard� m_brain.
    Esta funci�n lo aproxima rellenando el interior del cr�neo.

    Para casos water-only, devuelve todo False.
    """
    mask_skull = mask_skull.astype(bool)

    if mask_skull.sum() == 0:
        return np.zeros_like(mask_skull, dtype=bool)

    filled_head = binary_fill_holes(mask_skull)
    mask_brain = filled_head & (~mask_skull)

    return mask_brain.astype(bool)


# =========================================================
# K-WAVE DESDE UN .NPZ EXISTENTE
# =========================================================

def run_kwave_for_case(npz_path):
    """
    Reconstruye y corre k-Wave usando la informaci�n guardada en tu .npz:
    - mask_skull
    - source_mask
    - A_source
    - f0_hz
    - dx

    No hace plots ni calcula m�tricas.
    Solo ejecuta la simulaci�n y devuelve p_max_norm.
    """

    data_npz = np.load(npz_path)

    mask_skull = data_npz["mask_skull"].astype(bool)
    source_mask = data_npz["source_mask"].astype(np.uint8)

    A_source = float(data_npz["A_source"])
    f0_hz = float(data_npz["f0_hz"])
    dx = float(data_npz["dx"])

    Nx, Ny, Nz = mask_skull.shape

    mask_brain = reconstruct_brain_mask_from_skull(mask_skull)

    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])

    domain_diag = np.sqrt((Nx * dx) ** 2 + (Ny * dx) ** 2 + (Nz * dx) ** 2)
    t_end = 2.0 * domain_diag / PROPS["water"][0]

    dt, Nt, _ = get_dt_Nt_or_fallback(
        kgrid=kgrid,
        dx=dx,
        c_max=PROPS["skull"][0],
        cfl=CFL,
        t_end=t_end,
    )

    kgrid.dt = dt
    kgrid.Nt = Nt

    n_pts = int(source_mask.sum())

    if n_pts == 0:
        raise ValueError(f"source_mask vac�o en {npz_path.name}")

    source = kSource()
    tone = make_tone_burst(dt, Nt, f0_hz, N_CYCLES, A_source)
    source.p = np.tile(tone[None, :], (n_pts, 1)).astype(np.float32)
    source = make_source_compatible_for_time_signal(source, source_mask)

    sound_speed = np.where(
        mask_skull,
        PROPS["skull"][0],
        np.where(mask_brain, PROPS["brain"][0], PROPS["water"][0])
    ).astype(np.float32)

    density = np.where(
        mask_skull,
        PROPS["skull"][1],
        np.where(mask_brain, PROPS["brain"][1], PROPS["water"][1])
    ).astype(np.float32)

    alpha_coeff = np.where(
        mask_skull,
        PROPS["skull"][2],
        np.where(mask_brain, PROPS["brain"][2], PROPS["water"][2])
    ).astype(np.float32)

    medium = kWaveMedium(
        sound_speed=sound_speed,
        density=density,
        alpha_coeff=alpha_coeff,
        alpha_power=1.1,
    )

    sensor = kSensor()
    sensor.mask = np.ones((Nx, Ny, Nz), dtype=np.uint8)
    sensor.record = ["p_max"]

    sim_opts = SimulationOptions(
        save_to_disk=True,
        pml_inside=True,
        pml_size=PML_SIZE,
        data_cast="single",
    )

    exec_opts = SimulationExecutionOptions(
        is_gpu_simulation=True
    )

    result = kspaceFirstOrder3D(
        kgrid=kgrid,
        medium=medium,
        source=source,
        sensor=sensor,
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )

    p_max = safe_reshape_pmax(result["p_max"], Nx, Ny, Nz).astype(np.float32)
    p_max_norm = (p_max / A_source).astype(np.float32)

    return p_max_norm


def time_kwave_case(npz_path):
    gc.collect()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    t0 = time.perf_counter()

    _ = run_kwave_for_case(npz_path)

    t1 = time.perf_counter()

    return t1 - t0


# =========================================================
# MODELOS
# =========================================================

def build_model():
    """
    Ajusta esta funci�n si tu ResUNet3D_HQ usa otros argumentos.
    """
    try:
        model = ResUNet3D_HQ(
            in_ch=2,
            out_ch=1,
            out_positive=True,
        )
    except TypeError:
        try:
            model = ResUNet3D_HQ(
                in_channels=2,
                out_channels=1,
            )
        except TypeError as e:
            raise RuntimeError(
                "No pude construir ResUNet3D_HQ. "
                "Edita build_model() con los argumentos exactos de tu arquitectura."
            ) from e

    return model


def extract_state_dict(ckpt):
    """
    Soporta checkpoints de U-Net y cGAN.
    """
    if not isinstance(ckpt, dict):
        return ckpt

    possible_keys = [
        "model_state_dict",
        "state_dict",
        "generator_state_dict",
        "G_state_dict",
        "netG_state_dict",
        "generator",
        "model",
    ]

    for key in possible_keys:
        if key in ckpt:
            return ckpt[key]

    return ckpt


def clean_state_dict_keys(state_dict):
    cleaned = {}

    prefixes = [
        "module.",
        "model.",
        "generator.",
        "netG.",
        "G.",
    ]

    for k, v in state_dict.items():
        new_k = k

        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]

        cleaned[new_k] = v

    return cleaned


def load_model(ckpt_path):
    model = build_model()

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = extract_state_dict(ckpt)
    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("First missing keys:", missing[:5])

    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:5])

    model.to(DEVICE)
    model.eval()

    return model


def load_input_tensor(npz_path):
    """
    Input igual al usado para entrenar:
    channel 0 = source_mask
    channel 1 = mask_skull

    Output:
    [1, 2, D, H, W]
    """

    data = np.load(npz_path)

    source = data["source_mask"].astype(np.float32)
    skull = data["mask_skull"].astype(np.float32)

    x = np.stack([source, skull], axis=0)
    x = torch.from_numpy(x).unsqueeze(0)

    return x.to(DEVICE, non_blocking=True)


def warmup_model(model, files):
    if len(files) == 0:
        return

    print("Warming up model...")

    with torch.no_grad():
        for i in range(N_WARMUP):
            f = files[i % len(files)]
            x = load_input_tensor(f)

            synchronize()

            if USE_AMP and DEVICE == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(x)
            else:
                _ = model(x)

            synchronize()

            del x

            gc.collect()

            if DEVICE == "cuda":
                torch.cuda.empty_cache()


def time_model_case(model, npz_path, n_repeats=5):
    """
    Mide:
    - tiempo de cargar y preparar tensor
    - tiempo de forward pass
    - tiempo total
    """

    load_times = []
    forward_times = []
    total_times = []

    with torch.no_grad():
        for _ in range(n_repeats):
            gc.collect()

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            t0 = time.perf_counter()

            x = load_input_tensor(npz_path)

            synchronize()
            t1 = time.perf_counter()

            if USE_AMP and DEVICE == "cuda":
                with torch.amp.autocast("cuda"):
                    pred = model(x)
            else:
                pred = model(x)

            synchronize()
            t2 = time.perf_counter()

            load_times.append(t1 - t0)
            forward_times.append(t2 - t1)
            total_times.append(t2 - t0)

            del x, pred

    return {
        "load_tensor_mean_s": float(np.mean(load_times)),
        "load_tensor_std_s": float(np.std(load_times)),
        "forward_mean_s": float(np.mean(forward_times)),
        "forward_std_s": float(np.std(forward_times)),
        "total_mean_s": float(np.mean(total_times)),
        "total_std_s": float(np.std(total_times)),
    }


def save_progress(rows, out_csv):
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved progress to: {out_csv}")


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 80)
    print("TIMING COMPARISON: k-Wave vs Model 1 vs Model 2")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Test dir: {TEST_DIR}")
    print(f"N cases: {N_CASES}")
    print(f"Model repeats: {N_MODEL_REPEATS}")

    files = select_test_cases(TEST_DIR, N_CASES, RANDOM_SEED)

    print("\nSelected cases:")
    for f in files:
        print(" -", f.name)

    rows = []

    for idx, f in enumerate(files):
        rows.append({
            "case_idx": idx,
            "file": f.name,
            "path": str(f),

            "kwave_time_s": np.nan,
            "kwave_error": "",

            "model1_name": MODEL1_NAME,
            "model1_load_tensor_mean_s": np.nan,
            "model1_load_tensor_std_s": np.nan,
            "model1_forward_mean_s": np.nan,
            "model1_forward_std_s": np.nan,
            "model1_total_mean_s": np.nan,
            "model1_total_std_s": np.nan,
            "model1_error": "",

            "model2_name": MODEL2_NAME,
            "model2_load_tensor_mean_s": np.nan,
            "model2_load_tensor_std_s": np.nan,
            "model2_forward_mean_s": np.nan,
            "model2_forward_std_s": np.nan,
            "model2_total_mean_s": np.nan,
            "model2_total_std_s": np.nan,
            "model2_error": "",
        })

    # =====================================================
    # 1) K-WAVE
    # =====================================================
    print("\n" + "=" * 80)
    print("1) Running k-Wave timing")
    print("=" * 80)

    for i, f in enumerate(files):
        print(f"\n[k-Wave] Case {i + 1}/{len(files)}: {f.name}")

        try:
            elapsed = time_kwave_case(f)
            rows[i]["kwave_time_s"] = elapsed
            print(f"  k-Wave time: {elapsed:.4f} s")

        except Exception as e:
            rows[i]["kwave_error"] = repr(e)
            print(f"  ERROR in k-Wave: {repr(e)}")

        save_progress(rows, OUT_CSV)

    # =====================================================
    # 2) MODEL 1
    # =====================================================
    print("\n" + "=" * 80)
    print(f"2) Running {MODEL1_NAME} timing")
    print("=" * 80)

    model1 = load_model(MODEL1_CKPT)
    warmup_model(model1, files)

    for i, f in enumerate(files):
        print(f"\n[{MODEL1_NAME}] Case {i + 1}/{len(files)}: {f.name}")

        try:
            times = time_model_case(model1, f, N_MODEL_REPEATS)

            rows[i]["model1_load_tensor_mean_s"] = times["load_tensor_mean_s"]
            rows[i]["model1_load_tensor_std_s"] = times["load_tensor_std_s"]
            rows[i]["model1_forward_mean_s"] = times["forward_mean_s"]
            rows[i]["model1_forward_std_s"] = times["forward_std_s"]
            rows[i]["model1_total_mean_s"] = times["total_mean_s"]
            rows[i]["model1_total_std_s"] = times["total_std_s"]

            print(
                f"  Forward: {times['forward_mean_s']:.6f} s | "
                f"Total: {times['total_mean_s']:.6f} s"
            )

        except Exception as e:
            rows[i]["model1_error"] = repr(e)
            print(f"  ERROR in {MODEL1_NAME}: {repr(e)}")

        save_progress(rows, OUT_CSV)

    del model1
    gc.collect()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # =====================================================
    # 3) MODEL 2
    # =====================================================
    print("\n" + "=" * 80)
    print(f"3) Running {MODEL2_NAME} timing")
    print("=" * 80)

    model2 = load_model(MODEL2_CKPT)
    warmup_model(model2, files)

    for i, f in enumerate(files):
        print(f"\n[{MODEL2_NAME}] Case {i + 1}/{len(files)}: {f.name}")

        try:
            times = time_model_case(model2, f, N_MODEL_REPEATS)

            rows[i]["model2_load_tensor_mean_s"] = times["load_tensor_mean_s"]
            rows[i]["model2_load_tensor_std_s"] = times["load_tensor_std_s"]
            rows[i]["model2_forward_mean_s"] = times["forward_mean_s"]
            rows[i]["model2_forward_std_s"] = times["forward_std_s"]
            rows[i]["model2_total_mean_s"] = times["total_mean_s"]
            rows[i]["model2_total_std_s"] = times["total_std_s"]

            print(
                f"  Forward: {times['forward_mean_s']:.6f} s | "
                f"Total: {times['total_mean_s']:.6f} s"
            )

        except Exception as e:
            rows[i]["model2_error"] = repr(e)
            print(f"  ERROR in {MODEL2_NAME}: {repr(e)}")

        save_progress(rows, OUT_CSV)

    del model2
    gc.collect()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # =====================================================
    # SUMMARY
    # =====================================================
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    valid = df.dropna(
        subset=[
            "kwave_time_s",
            "model1_total_mean_s",
            "model2_total_mean_s",
        ]
    )

    if len(valid) == 0:
        print("\nNo hay filas completas para hacer resumen.")
        print(f"CSV parcial guardado en: {OUT_CSV}")
        return

    summary_rows = []

    def add_summary(method_name, col):
        summary_rows.append({
            "method": method_name,
            "n_cases": int(valid[col].count()),
            "mean_time_s": float(valid[col].mean()),
            "median_time_s": float(valid[col].median()),
            "std_time_s": float(valid[col].std()),
            "min_time_s": float(valid[col].min()),
            "max_time_s": float(valid[col].max()),
            "p90_time_s": float(valid[col].quantile(0.90)),
        })

    add_summary("k-Wave", "kwave_time_s")
    add_summary(MODEL1_NAME, "model1_total_mean_s")
    add_summary(MODEL2_NAME, "model2_total_mean_s")

    summary_df = pd.DataFrame(summary_rows)

    kwave_mean = summary_df.loc[
        summary_df["method"] == "k-Wave",
        "mean_time_s"
    ].values[0]

    summary_df["speedup_vs_kwave"] = kwave_mean / summary_df["mean_time_s"]
    summary_df["time_reduction_percent"] = (
        1.0 - summary_df["mean_time_s"] / kwave_mean
    ) * 100.0

    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary_df)

    print(f"\nSaved full timing CSV to: {OUT_CSV}")
    print(f"Saved summary CSV to: {OUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()