import os
import numpy as np
import matplotlib.pyplot as plt

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions


# ==============================================================================
# 1) TRANSDUCTOR: máscara bowl (calota esférica) orientada
# ==============================================================================

def create_oriented_bowl(kgrid, focus_pos, transducer_center, radius_curvature, aperture_diameter, thickness):
    F = np.array(focus_pos, dtype=float)
    T = np.array(transducer_center, dtype=float)

    axis_vec = T - F
    axis_len = np.linalg.norm(axis_vec)
    axis_dir = axis_vec / (axis_len + 1e-12)

    X, Y, Z = np.meshgrid(kgrid.x_vec, kgrid.y_vec, kgrid.z_vec, indexing="ij")

    dist_to_focus = np.sqrt((X - F[0])**2 + (Y - F[1])**2 + (Z - F[2])**2)
    shell_mask = np.abs(dist_to_focus - radius_curvature) <= (thickness / 2)

    vx, vy, vz = X - F[0], Y - F[1], Z - F[2]
    dot = vx * axis_dir[0] + vy * axis_dir[1] + vz * axis_dir[2]
    cos_theta = dot / (dist_to_focus + 1e-12)

    half_ap = aperture_diameter / 2.0
    sin_val = np.clip(half_ap / (radius_curvature + 1e-12), -1.0, 1.0)
    min_cos = np.cos(np.arcsin(sin_val))

    return shell_mask & (cos_theta >= min_cos)


# ==============================================================================
# 2) TIEMPO ROBUSTO
# ==============================================================================

def try_make_time(kgrid, c_max, cfl):
    candidates = [
        ("makeTime(c_max, cfl=...)", lambda: kgrid.makeTime(c_max, cfl=cfl)),
        ("makeTime(c_max, CFL=...)", lambda: kgrid.makeTime(c_max, CFL=cfl)),
        ("makeTime(c_max, cfl) posicional", lambda: kgrid.makeTime(c_max, cfl)),
    ]
    for name, fn in candidates:
        try:
            fn()
            return True, name
        except TypeError:
            continue
        except Exception:
            continue
    return False, None


def get_dt_Nt_or_fallback(kgrid, dx, c_max, cfl, t_end):
    dt = getattr(kgrid, "dt", None)
    Nt = getattr(kgrid, "Nt", None)

    t_arr = getattr(kgrid, "t_array", None)
    if (dt is None or Nt is None or int(Nt) < 2) and t_arr is not None:
        t_arr = np.asarray(t_arr).ravel()
        if t_arr.size >= 2:
            dt = float(t_arr[1] - t_arr[0])
            Nt = int(t_arr.size)

    if dt is None or Nt is None or int(Nt) < 2:
        dt = float(cfl * dx / c_max)
        Nt = int(np.ceil(t_end / dt)) + 1

    dt = float(dt)
    Nt = int(Nt)
    if Nt < 2:
        raise RuntimeError(f"Tiempo inválido: dt={dt}, Nt={Nt}")

    t = (np.arange(Nt, dtype=np.float32) * np.float32(dt))
    return dt, Nt, t


# ==============================================================================
# 3) SEÑAL TUS-LIKE
# ==============================================================================

def make_tone_burst(dt, Nt, f0_hz, n_cycles, amplitude_pa):
    dt = float(dt)
    Nt = int(Nt)
    t = np.arange(Nt, dtype=np.float32) * np.float32(dt)

    T = n_cycles / float(f0_hz)
    n_on = int(np.round(T / dt))
    n_on = max(min(n_on, Nt), 2)

    tone = np.sin(2.0 * np.pi * f0_hz * t[:n_on]).astype(np.float32)
    win = np.hanning(n_on).astype(np.float32)

    sig = np.zeros((Nt,), dtype=np.float32)
    sig[:n_on] = (amplitude_pa * tone * win).astype(np.float32)
    return sig


# ==============================================================================
# 4) PARCHE WRAPPER kSource
# ==============================================================================

def make_source_compatible_for_time_signal(source, p_mask_u8):
    pmask = p_mask_u8.astype(np.uint8)
    source.mask = pmask
    source.p_mask = pmask
    source.time_reversal_boundary_data = np.array([], dtype=np.float32)

    if hasattr(source, "p0"):
        try:
            delattr(source, "p0")
        except Exception:
            pass

    for attr in ("u_mask", "ux", "uy", "uz"):
        if hasattr(source, attr):
            try:
                delattr(source, attr)
            except Exception:
                pass
    return source


# ==============================================================================
# 5) RUIDO SUAVE 3D (FFT low-pass)
# ==============================================================================

def _smooth_noise_3d_fft(shape, rng, corr_len_vox=12.0, amplitude=1.0):
    n = rng.standard_normal(shape).astype(np.float32)

    Nx, Ny, Nz = shape
    kx = np.fft.fftfreq(Nx).astype(np.float32)[:, None, None]
    ky = np.fft.fftfreq(Ny).astype(np.float32)[None, :, None]
    kz = np.fft.fftfreq(Nz).astype(np.float32)[None, None, :]

    k2 = (kx*kx + ky*ky + kz*kz)
    sigma = float(corr_len_vox) / 2.0
    filt = np.exp(-(2.0 * (np.pi**2)) * (sigma**2) * k2).astype(np.float32)

    s = np.fft.ifftn(np.fft.fftn(n) * filt).real.astype(np.float32)
    s -= s.mean()
    s /= (np.max(np.abs(s)) + 1e-8)
    return amplitude * s


# ==============================================================================
# 6) Sampler ovoid tipo TAC + gap garantizado (ratios suaves)
# ==============================================================================

def sample_skull_axes_for_fixed_rT_gap_ovoid(
    rng,
    r_T=25.0e-3,
    gap_min=0.5e-3,
    gap_max=2.0e-3,
    ap_ratio=(1.05, 1.1),   # b/a  (AP > LR, pero suave)
    si_ratio=(0.95, 1.05),   # c/a  (SI similar)
):
    upper = r_T - gap_min
    lower = r_T - gap_max
    if lower <= 0 or lower >= upper:
        raise RuntimeError(
            "Rango inválido para skull_outer_max. Revisa r_T y gaps.")

    skull_outer_max = float(rng.uniform(lower, upper))

    r_ap = float(rng.uniform(*ap_ratio))
    r_si = float(rng.uniform(*si_ratio))

    # Forzamos b (AP) como eje mayor
    b_out = skull_outer_max
    a_out = b_out / r_ap
    c_out = a_out * r_si

    # seguridad
    m = max(a_out, b_out, c_out)
    if m > skull_outer_max:
        s = skull_outer_max / (m + 1e-12)
        a_out *= s
        b_out *= s
        c_out *= s

    return float(a_out), float(b_out), float(c_out)


# ==============================================================================
# 7) CRÁNEO PROCEDURAL TAC (ovoide base + aplanado SOLO en XY)
#    - r_xy: superelipse suave en XY (p_xy ~ 2.25)
#    - r_z: elíptico normal en Z
#    - combina con sqrt(r_xy^2 + r_z^2) => 3D redondeado, axial más TAC
#    - sin base_cut/flatten (0) para evitar “caja”
# ==============================================================================

def create_skull_masks_procedural_TAC(
    kgrid,
    a_out, b_out, c_out,
    thickness_min=3.0e-3,
    thickness_max=6.0e-3,
    thickness_var=0.6e-3,
    anisotropy_strength=0.45,
    angular_strength=0.9,
    rough_amp=0.18e-3,
    rough_corr_len_vox=16.0,
    shape_amp=0.4e-3,
    shape_corr_len_vox=34.0,
    flatten_strength=0.0e-3,
    base_cut_strength=0.0e-3,
    p_xy=2.2,                  # 👈 clave: aplanado lateral suave SOLO XY
    center_shift_mm=0.8,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    Nx, Ny, Nz = len(kgrid.x_vec), len(kgrid.y_vec), len(kgrid.z_vec)
    X, Y, Z = np.meshgrid(kgrid.x_vec, kgrid.y_vec, kgrid.z_vec, indexing="ij")

    # shift suave del centro
    shift = rng.uniform(-center_shift_mm*1e-3,
                        center_shift_mm*1e-3, size=3).astype(np.float32)
    Xc = (X - shift[0]).astype(np.float32)
    Yc = (Y - shift[1]).astype(np.float32)
    Zc = (Z - shift[2]).astype(np.float32)

    # direcciones unitarias
    R = np.sqrt(Xc*Xc + Yc*Yc + Zc*Zc) + 1e-12
    ux = (Xc / R).astype(np.float32)
    uy = (Yc / R).astype(np.float32)
    uz = (Zc / R).astype(np.float32)

    # rotación aleatoria para patrón angular de grosor
    q = rng.normal(size=4).astype(np.float32)
    q /= (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    Rm = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)

    uxr = (Rm[0, 0]*ux + Rm[0, 1]*uy + Rm[0, 2]*uz).astype(np.float32)
    uyr = (Rm[1, 0]*ux + Rm[1, 1]*uy + Rm[1, 2]*uz).astype(np.float32)
    uzr = (Rm[2, 0]*ux + Rm[2, 1]*uy + Rm[2, 2]*uz).astype(np.float32)

    # bases suaves para grosor local
    B1 = uzr
    B2 = uxr
    B3 = uyr
    B4 = (uxr*uxr - uyr*uyr)
    B5 = (3.0*uzr*uzr - 1.0)

    wts = rng.normal(size=5).astype(np.float32)
    wts /= (np.linalg.norm(wts) + 1e-12)
    w1, w2, w3, w4, w5 = wts

    ang_field = (w1*B1 + w2*B2 + w3*B3 + 0.7*w4 *
                 B4 + 0.7*w5*B5).astype(np.float32)
    ang_field -= ang_field.mean()
    ang_field /= (np.max(np.abs(ang_field)) + 1e-8)
    ang_field *= float(angular_strength)

    # ruido suave para variación adicional del grosor
    t_noise = _smooth_noise_3d_fft(
        (Nx, Ny, Nz), rng, corr_len_vox=20.0, amplitude=1.0)

    thickness_mean = float(rng.uniform(thickness_min, thickness_max))
    t_local = (
        thickness_mean
        + anisotropy_strength * thickness_var * ang_field
        + 0.45 * thickness_var * t_noise
    ).astype(np.float32)
    t_local = np.clip(t_local, thickness_min, thickness_max).astype(np.float32)

    # rugosidad superficial fina (leve)
    surf_noise = _smooth_noise_3d_fft(
        (Nx, Ny, Nz), rng, corr_len_vox=rough_corr_len_vox, amplitude=1.0)
    surf_disp = (rough_amp * surf_noise).astype(np.float32)

    # deformación global mínima (detalle)
    shape_noise = _smooth_noise_3d_fft(
        (Nx, Ny, Nz), rng, corr_len_vox=shape_corr_len_vox, amplitude=1.0)
    shape_disp = (shape_amp * shape_noise).astype(np.float32)

    # flatten + base_cut (los dejamos en 0 por defecto)
    z_norm = (Zc / (np.max(np.abs(kgrid.z_vec)) + 1e-12)).astype(np.float32)
    flatten = (-flatten_strength * (z_norm**2)).astype(np.float32)
    base_cut = (-base_cut_strength * (z_norm < -0.35).astype(np.float32)
                * ((-z_norm - 0.35) / 0.65)).astype(np.float32)

    # ---- Forma base TAC: aplanado SOLO XY (superelipse suave) + Z elíptico normal
    p_xy = float(p_xy)
    r_xy = (
        (np.abs(Xc) / (a_out + 1e-12))**p_xy +
        (np.abs(Yc) / (b_out + 1e-12))**p_xy
    ) ** (1.0 / p_xy)

    r_z = np.abs(Zc) / (c_out + 1e-12)
    # 3D redondeado, axial TAC-like
    r = np.sqrt(r_xy*r_xy + r_z*r_z).astype(np.float32)

    Rref = float((a_out + b_out + c_out) / 3.0)

    # distancia hacia adentro desde la superficie externa
    d_in = (1.0 - r) * Rref + surf_disp + shape_disp + flatten + base_cut

    mask_head = (d_in >= 0.0)
    mask_skull = mask_head & (d_in <= t_local)
    mask_brain = mask_head & (d_in > t_local)

    return mask_skull, mask_brain, t_local, thickness_mean


# ==============================================================================
# 8) Transductor fijo: r_T fijo + R fijo = 30 mm, foco geométrico
# ==============================================================================

def sample_transducer_T_and_focus_geometric(rng, r_T=25.0e-3, R_const=30.0e-3):
    u_out = rng.normal(size=3).astype(np.float32)
    u_out /= (np.linalg.norm(u_out) + 1e-12)
    T = (r_T * u_out).astype(np.float32)
    n_in = (-u_out).astype(np.float32)
    F = (T + float(R_const) * n_in).astype(np.float32)
    return T, F


# ==============================================================================
# 9) reshape seguro
# ==============================================================================

def safe_reshape_pmax(p_max, Nx, Ny, Nz):
    if getattr(p_max, "ndim", 0) == 1:
        try:
            return p_max.reshape((Nx, Ny, Nz), order="F")
        except Exception:
            return p_max.reshape((Nx, Ny, Nz), order="C")
    return p_max


# ==============================================================================
# 10) Preview geometría (3 vistas)
# ==============================================================================

def preview_geometry(mask_skull, source_mask_u8, i, Nx, Ny, Nz):
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].imshow(mask_skull[cx, :, :].T, origin="lower",
                 cmap="gray", interpolation="nearest")
    ax[0].imshow(source_mask_u8[cx, :, :].T, origin="lower",
                 cmap="hot", alpha=0.6, interpolation="nearest")
    ax[0].set_title(f"Geometría (skull+source) Sagital (x={cx})")

    ax[1].imshow(mask_skull[:, cy, :].T, origin="lower",
                 cmap="gray", interpolation="nearest")
    ax[1].imshow(source_mask_u8[:, cy, :].T, origin="lower",
                 cmap="hot", alpha=0.6, interpolation="nearest")
    ax[1].set_title(f"Geometría (skull+source) Coronal (y={cy})")

    ax[2].imshow(mask_skull[:, :, cz].T, origin="lower",
                 cmap="gray", interpolation="nearest")
    ax[2].imshow(source_mask_u8[:, :, cz].T, origin="lower",
                 cmap="hot", alpha=0.6, interpolation="nearest")
    ax[2].set_title(f"Geometría (skull+source) Axial (z={cz})")

    plt.suptitle(f"Sim {i} | Geometry Preview", y=1.02)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 11) MAIN
# ==============================================================================

def main():
    OUT_DIR = "dataset_TUS_dx05_TAClike_ovoidXY_R30_thick2dx"
    os.makedirs(OUT_DIR, exist_ok=True)

    N_SIMS = 1000
    SEED = 123
    rng = np.random.default_rng(SEED)

    # Grid
    Nx, Ny, Nz = 128, 128, 128
    dx = 0.5e-3
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])

    # PML
    PML_SIZE = 10
    half_interior = 0.5 * (Nx - 2 * PML_SIZE) * dx  # ~27mm
    margin_to_pml = 2.0e-3

    # Transductor fijo (pedidos)
    APERTURE = 25e-3
    R_CONST = 30e-3           # ✅ R = 30 mm
    r_T = 25e-3
    GAP_MIN, GAP_MAX = 0.5e-3, 2.0e-3
    BOWL_THICKNESS = 2 * dx   # ✅ thickness = 2*dx

    if r_T > (half_interior - margin_to_pml):
        raise RuntimeError(
            f"r_T={r_T*1e3:.1f}mm demasiado cerca del PML. "
            f"half_interior={half_interior*1e3:.1f}mm, margen={margin_to_pml*1e3:.1f}mm."
        )
    if R_CONST < (APERTURE / 2):
        raise RuntimeError("R_CONST debe ser >= D/2.")

    # Señal
    F0 = 500e3
    N_CYCLES = 10
    A_MIN = 0.3e6
    A_MAX = 1.2e6

    # Cráneo (más proporcional)
    TMIN, TMAX = 2.0e-3, 5.0e-3

    # Medio
    c_water, rho_water, alpha_water = 1482.0, 994.0, 0.0126
    c_skull, rho_skull, alpha_skull = 2800.0, 1850.0, 15.0
    c_brain, rho_brain, alpha_brain = 1546.0, 1046.0, 0.5
    c_max = max(c_skull, c_brain, c_water)

    # Tiempo
    domain_diag = np.sqrt((Nx*dx)**2 + (Ny*dx)**2 + (Nz*dx)**2)
    c_min = min(c_water, c_brain)
    t_end = float(2.0 * domain_diag / c_min)
    CFL = 0.1

    ok_time, how = try_make_time(kgrid, c_max, CFL)
    print(
        f"🕒 makeTime: {how}" if ok_time else "🕒 makeTime no aplicó. Usaré fallback dt/Nt.")

    dt, Nt, _ = get_dt_Nt_or_fallback(kgrid, dx, c_max, CFL, t_end)
    try:
        kgrid.dt = dt
        kgrid.Nt = Nt
    except Exception:
        pass
    print(f"🕒 Tiempo: dt={dt:.3e} s | Nt={Nt}")

    sim_opts = SimulationOptions(
        save_to_disk=True,
        pml_inside=False,
        pml_size=PML_SIZE,
        data_cast="single",
    )
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=True)

    print(f"\n📦 Generando dataset ({N_SIMS} sims) → {OUT_DIR}/")
    print(
        f"🔧 Transductor: D={APERTURE*1e3:.1f}mm | R={R_CONST*1e3:.1f}mm | thickness={BOWL_THICKNESS*1e3:.1f}mm")
    print(
        f"🔧 Cráneo: grosor [{TMIN*1e3:.1f}, {TMAX*1e3:.1f}] mm | gap [{GAP_MIN*1e3:.1f}, {GAP_MAX*1e3:.1f}] mm | p_xy=2.25")

    for i in range(N_SIMS):
        print(f"\n=== Sim {i:03d}/{N_SIMS-1:03d} ===")

        # 1) ejes ovoid tipo TAC con gap garantizado (ratios suaves)
        a_out, b_out, c_out = sample_skull_axes_for_fixed_rT_gap_ovoid(
            rng=rng,
            r_T=r_T,
            gap_min=GAP_MIN,
            gap_max=GAP_MAX,
            ap_ratio=(1.08, 1.18),
            si_ratio=(0.95, 1.05),
        )
        skull_outer_max = max(a_out, b_out, c_out)
        gap_real = float(r_T - skull_outer_max)

        # 2) cráneo TAC-like (aplanado SOLO XY)
        mask_skull, mask_brain, t_local, t_mean = create_skull_masks_procedural_TAC(
            kgrid=kgrid,
            a_out=a_out, b_out=b_out, c_out=c_out,
            thickness_min=TMIN, thickness_max=TMAX,
            thickness_var=0.6e-3,
            rough_amp=0.18e-3,
            shape_amp=0.4e-3,
            flatten_strength=0.0e-3,
            base_cut_strength=0.0e-3,
            p_xy=2.25,
            center_shift_mm=0.8,
            rng=rng
        )

        # 3) transductor fijo + foco geométrico
        source_pos, focus_pos = sample_transducer_T_and_focus_geometric(
            rng=rng, r_T=r_T, R_const=R_CONST
        )

        # 4) máscara bowl (thickness=2*dx) y sin solapar hueso
        source_mask_bool = create_oriented_bowl(
            kgrid=kgrid,
            focus_pos=focus_pos,
            transducer_center=source_pos,
            radius_curvature=float(R_CONST),
            aperture_diameter=APERTURE,
            thickness=BOWL_THICKNESS
        )
        source_mask_bool = source_mask_bool & (~mask_skull)
        n_src_pts = int(np.sum(source_mask_bool))

        if n_src_pts <= 0:
            ok_src = False
            for _ in range(120):
                source_pos, focus_pos = sample_transducer_T_and_focus_geometric(
                    rng, r_T=r_T, R_const=R_CONST)
                source_mask_bool = create_oriented_bowl(
                    kgrid, focus_pos, source_pos, float(R_CONST), APERTURE, BOWL_THICKNESS)
                source_mask_bool = source_mask_bool & (~mask_skull)
                n_src_pts = int(np.sum(source_mask_bool))
                if n_src_pts > 0:
                    ok_src = True
                    break
            if not ok_src:
                raise RuntimeError(
                    "No pude generar source_mask válida. Revisa r_T/gap/PML.")

        source_mask_u8 = source_mask_bool.astype(np.uint8)

        # 5) medio
        sound_speed = np.ones((Nx, Ny, Nz), dtype=np.float32) * c_water
        density = np.ones((Nx, Ny, Nz), dtype=np.float32) * rho_water
        alpha_coeff = np.ones((Nx, Ny, Nz), dtype=np.float32) * alpha_water

        sound_speed[mask_brain] = c_brain
        density[mask_brain] = rho_brain
        alpha_coeff[mask_brain] = alpha_brain

        sound_speed[mask_skull] = c_skull
        density[mask_skull] = rho_skull
        alpha_coeff[mask_skull] = alpha_skull

        medium = kWaveMedium(
            sound_speed=sound_speed,
            density=density,
            alpha_coeff=alpha_coeff,
            alpha_power=1.1
        )

        # 6) señal
        A_source = float(rng.uniform(A_MIN, A_MAX))
        sig = make_tone_burst(dt, Nt, F0, N_CYCLES, A_source)
        source_p = np.tile(sig[None, :], (n_src_pts, 1)).astype(np.float32)

        source = kSource()
        source.p_mask = source_mask_u8
        source.p = source_p
        source = make_source_compatible_for_time_signal(source, source_mask_u8)

        sensor = kSensor()
        sensor.mask = np.ones((Nx, Ny, Nz), dtype=np.uint8)
        sensor.record = ["p_max"]

        print(
            f"skull_axes(mm) a={a_out*1e3:.2f} b={b_out*1e3:.2f} c={c_out*1e3:.2f} | "
            f"gap={gap_real*1e3:.2f} | t_mean={t_mean*1e3:.2f} | src_pts={n_src_pts}"
        )

        # 7) sim
        data = kspaceFirstOrder3D(
            kgrid=kgrid,
            medium=medium,
            source=source,
            sensor=sensor,
            simulation_options=sim_opts,
            execution_options=exec_opts
        )

        p_max = safe_reshape_pmax(data["p_max"], Nx, Ny, Nz).astype(np.float32)
        p_max_norm = (p_max / A_source).astype(np.float32)

        # 8) guardar
        out_path = os.path.join(OUT_DIR, f"sample_{i:04d}.npz")
        np.savez_compressed(
            out_path,
            # inputs
            mask_skull=mask_skull.astype(np.uint8),
            mask_brain=mask_brain.astype(np.uint8),
            sound_speed=sound_speed,
            density=density,
            alpha_coeff=alpha_coeff,

            # transducer + geometry
            source_pos=source_pos.astype(np.float32),
            focus_pos=focus_pos.astype(np.float32),
            aperture=np.float32(APERTURE),
            radius_curvature=np.float32(R_CONST),
            r_T=np.float32(r_T),
            gap=np.float32(gap_real),
            source_mask=source_mask_u8,

            # TUS params
            f0_hz=np.float32(F0),
            n_cycles=np.int32(N_CYCLES),
            A_source=np.float32(A_source),
            dt=np.float32(dt),
            Nt=np.int32(Nt),
            cfl=np.float32(CFL),

            # skull params
            a_out=np.float32(a_out), b_out=np.float32(b_out), c_out=np.float32(c_out),
            skull_thickness_min=np.float32(TMIN),
            skull_thickness_max=np.float32(TMAX),
            skull_thickness_mean=np.float32(t_mean),
            p_xy=np.float32(2.25),

            # outputs
            p_max=p_max,
            p_max_norm=p_max_norm,

            # grid
            dx=np.float32(dx),
            Nx=np.int32(Nx), Ny=np.int32(Ny), Nz=np.int32(Nz),
            pml_size=np.int32(PML_SIZE),
        )
        print(f"💾 Guardado: {out_path}")

        # preview primeras 2
        if i < 2:
            preview_geometry(mask_skull, source_mask_u8, i, Nx, Ny, Nz)

    print("\n✅ Dataset generado.")
    print("   IA entrena con p_max_norm; para volver a Pa: p_max_pred = p_max_norm_pred * A_source.")


if __name__ == "__main__":
    main()
