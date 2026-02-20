# dataset_oval3D_best_clinical_roi.py
# "Mejor dataset posible" (práctico y realista) para U-Net 3D surrogate k-Wave (TUS transcraneal)
#
# ✅ Foco SIEMPRE en cerebro y BALANCEADO por profundidad real (distance transform)
# ✅ Transductor SOLO "arriba y lateral" (NO 360°): muestreo en casquete del hemisferio superior + excluye zona inferior
# ✅ Transductor pegado a la superficie local (radio exacto del elipsoide en esa dirección) + gel
# ✅ Eje del bowl apuntando al foco (condicionamiento clínico)
# ✅ Chequeo de colisión transductor-cráneo
# ✅ Variación por muestra (c/rho/alpha) + heterogeneidad espacial suave (aberración más real)
# ✅ Thickness > 1 voxel (source_mask estable)
# ✅ ROI alrededor del foco guardada en .npz (para weighted loss, NO como canal)
#
# Requisitos:
#   pip install scipy
#
# Output: .npz con keys:
#   inputs: mask_skull, mask_brain, sound_speed, density, alpha_coeff, source_mask, source_pos, focus_pos, roi_focus
#   outputs: p_max, p_max_norm
#   meta: depth_class, depth_mm, roi_radius_mm, etc.

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions


# ==============================================================================
# 1) GEOMETRÍA: ELIPSOIDE + BOWL ORIENTADO
# ==============================================================================

def create_ellipsoid_mask(kgrid, a, b, c, center=(0.0, 0.0, 0.0), warp=None):
    """
    warp opcional para deformación suave del elipsoide (evita geometría demasiado perfecta).
    warp={"amp":0.5e-3, "freq":2, "seed":123}
    """
    cx, cy, cz = center
    X, Y, Z = np.meshgrid(kgrid.x_vec, kgrid.y_vec, kgrid.z_vec, indexing="ij")
    Xc, Yc, Zc = X - cx, Y - cy, Z - cz

    if warp is not None:
        amp = float(warp.get("amp", 0.0))
        freq = int(warp.get("freq", 2))
        seed = int(warp.get("seed", 0))
        rng = np.random.default_rng(seed)
        ph1, ph2, ph3 = rng.uniform(0, 2*np.pi, size=3)

        disp = amp * (
            np.sin(freq * (Xc / (a + 1e-12)) * np.pi + ph1) +
            np.sin(freq * (Yc / (b + 1e-12)) * np.pi + ph2) +
            np.sin(freq * (Zc / (c + 1e-12)) * np.pi + ph3)
        ) / 3.0

        a_eff = a + disp
        b_eff = b + disp
        c_eff = c + disp
    else:
        a_eff, b_eff, c_eff = a, b, c

    eq = (Xc**2 / (a_eff**2 + 1e-24)) + \
        (Yc**2 / (b_eff**2 + 1e-24)) + (Zc**2 / (c_eff**2 + 1e-24))
    return eq <= 1.0


def create_oriented_bowl(kgrid, focus_pos, transducer_center, radius_curvature, aperture_diameter, thickness):
    F = np.asarray(focus_pos, dtype=np.float32).reshape(3,)
    T = np.asarray(transducer_center, dtype=np.float32).reshape(3,)

    axis_vec = T - F
    axis_len = float(np.linalg.norm(axis_vec))
    axis_dir = axis_vec / (axis_len + 1e-12)   # shape (3,)

    X, Y, Z = np.meshgrid(kgrid.x_vec, kgrid.y_vec, kgrid.z_vec, indexing="ij")

    dist_to_focus = np.sqrt((X - F[0])**2 + (Y - F[1])**2 + (Z - F[2])**2)
    shell_mask = np.abs(dist_to_focus - float(radius_curvature)
                        ) <= (float(thickness) / 2)

    vx, vy, vz = X - F[0], Y - F[1], Z - F[2]
    dot = vx * axis_dir[0] + vy * axis_dir[1] + vz * axis_dir[2]
    cos_theta = dot / (dist_to_focus + 1e-12)

    half_ap = float(aperture_diameter) / 2
    sin_val = np.clip(half_ap / (float(radius_curvature) + 1e-12), -1, 1)
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
# 3) SEÑAL: tone burst Hann
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
# 4) Compat wrapper + reshape
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


def safe_reshape_pmax(p_max, Nx, Ny, Nz):
    if getattr(p_max, "ndim", 0) == 1:
        try:
            return p_max.reshape((Nx, Ny, Nz), order="F")
        except Exception:
            return p_max.reshape((Nx, Ny, Nz), order="C")
    return p_max


# ==============================================================================
# 5) Sampling de cráneo, foco por profundidad, transductor clínico
# ==============================================================================

def random_skull_params(rng):
    a_in = rng.uniform(8.5e-3, 12.0e-3)
    b_in = rng.uniform(11.0e-3, 15.5e-3)
    c_in = rng.uniform(8.5e-3, 12.0e-3)
    grosor = rng.uniform(2.0e-3, 8.0e-3)
    return a_in, b_in, c_in, grosor


def add_spatial_heterogeneity(rng, base_map, mask, rel_std=0.03, smooth_iters=3):
    noise = rng.normal(loc=0.0, scale=rel_std,
                       size=base_map.shape).astype(np.float32)
    field = (1.0 + noise)

    for _ in range(smooth_iters):
        f = field
        field = (
            f +
            np.roll(f, 1, 0) + np.roll(f, -1, 0) +
            np.roll(f, 1, 1) + np.roll(f, -1, 1) +
            np.roll(f, 1, 2) + np.roll(f, -1, 2)
        ) / 7.0

    out = base_map.copy().astype(np.float32)
    out[mask] = (base_map[mask] * field[mask]).astype(np.float32)
    return out


def pick_focus_by_depth(rng, mask_brain, depth_mm, kgrid,
                        target="mix",
                        shallow=(2.0, 6.0),
                        mid=(6.0, 12.0),
                        deep=(12.0, 30.0),
                        margin_vox=3):
    """
    Escoge focus_pos dentro del cerebro, balanceado por profundidad real (mm).
    depth_mm: distance_transform_edt(mask_brain)*dx*1e3
    """
    Nx, Ny, Nz = mask_brain.shape
    coords = np.argwhere(mask_brain > 0)
    if coords.shape[0] == 0:
        raise RuntimeError("mask_brain vacío")

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ok = (
        (x >= margin_vox) & (x < Nx - margin_vox) &
        (y >= margin_vox) & (y < Ny - margin_vox) &
        (z >= margin_vox) & (z < Nz - margin_vox)
    )
    coords2 = coords[ok]
    if coords2.shape[0] > 0:
        coords = coords2

    if target == "mix":
        target = rng.choice(["shallow", "mid", "deep"], p=[0.33, 0.34, 0.33])

    lo, hi = {"shallow": shallow, "mid": mid, "deep": deep}[target]
    d = depth_mm[coords[:, 0], coords[:, 1], coords[:, 2]]
    band = coords[(d >= lo) & (d < hi)]
    if band.shape[0] == 0:
        band = coords

    idx = band[rng.integers(0, band.shape[0])]
    ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])
    focus = np.array([kgrid.x_vec[ix], kgrid.y_vec[iy],
                     kgrid.z_vec[iz]], dtype=np.float32)
    return focus, target, float(depth_mm[ix, iy, iz])


def get_ellipsoid_radius_in_direction(u, a, b, c):
    """
    Radio exacto del elipsoide externo en dirección u (unitaria).
    """
    ux, uy, uz = u
    denom = (ux/a)**2 + (uy/b)**2 + (uz/c)**2
    return 1.0 / np.sqrt(denom + 1e-24)


def sample_surface_direction_upper_lateral(rng,
                                           min_z=0.25,
                                           max_polar_deg=70.0,
                                           min_azimuth_sep_deg=15.0):
    """
    Devuelve un vector unitario u en un casquete del hemisferio superior (z>=min_z),
    con polar angle <= max_polar_deg respecto a +Z.
    Además fuerza "lateral" (no solo coronario puro) usando min_azimuth_sep_deg
    (evita concentrarse en un azimut estrecho).
    """
    max_polar = np.deg2rad(max_polar_deg)

    for _ in range(200):
        # muestreo uniforme en esfera y luego restringimos
        u = rng.normal(size=3).astype(np.float32)
        u /= (np.linalg.norm(u) + 1e-12)

        # hemisferio superior
        if u[2] < 0:
            u[2] *= -1

        # mínimo z (evita demasiado lateral extremo tipo oreja-oreja)
        if u[2] < min_z:
            continue

        # polar angle constraint
        polar = np.arccos(np.clip(u[2], -1.0, 1.0))
        if polar > max_polar:
            continue

        # "lateral": evita azimut demasiado cercano a 0 siempre (opcional suave)
        # azimut = atan2(y,x)
        az = np.arctan2(u[1], u[0])
        # forzamos que no caiga siempre cerca de 0; esto solo diversifica un poco.
        if np.abs(np.rad2deg(az)) < min_azimuth_sep_deg and rng.random() < 0.5:
            continue

        return u

    # fallback
    u = np.array([0.3, 0.3, 0.9], dtype=np.float32)
    u /= np.linalg.norm(u)
    return u


def place_transducer_clinical(rng, focus_pos, a_out, b_out, c_out,
                              dist_from_skin_range=(0.5e-3, 5.0e-3),
                              max_polar_deg=75.0,
                              min_z=0.20):
    """
    Transductor en superficie externa (elipsoide) en "arriba y lateral".
    - Elegimos punto en cráneo (dirección u) con restricciones clínicas.
    - Lo colocamos a (R_surface + gel) * u
    - Así queda fuera del cráneo, pegado, y el bowl apunta naturalmente al foco al usar create_oriented_bowl.
    """
    u = sample_surface_direction_upper_lateral(
        rng,
        min_z=min_z,
        max_polar_deg=max_polar_deg,
        min_azimuth_sep_deg=10.0
    )
    R_surf = float(get_ellipsoid_radius_in_direction(u, a_out, b_out, c_out))
    gel = float(rng.uniform(dist_from_skin_range[0], dist_from_skin_range[1]))
    source_pos = (u * (R_surf + gel)).astype(np.float32)
    return source_pos.astype(np.float32), np.array(focus_pos, dtype=np.float32), u, R_surf, gel


# ==============================================================================
# 6) ROI (para loss)
# ==============================================================================

def make_roi_sphere(kgrid, center_pos, radius_m):
    cx, cy, cz = map(float, center_pos)
    X, Y, Z = np.meshgrid(kgrid.x_vec, kgrid.y_vec, kgrid.z_vec, indexing="ij")
    dist2 = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
    roi = (dist2 <= (radius_m**2))
    return roi.astype(np.uint8)


def make_roi_brain_focus(kgrid, focus_pos, mask_brain, radius_m):
    roi = make_roi_sphere(kgrid, focus_pos, radius_m)
    roi = (roi.astype(bool) & mask_brain.astype(bool)).astype(np.uint8)
    return roi


# ==============================================================================
# 7) MAIN
# ==============================================================================

def main():
    OUT_DIR = "dataset_oval3D_best_clinical_roi"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ----- Dataset size -----
    N_TARGET = 600           # recomendado: 600–1200 (según tiempo)
    SEED = 999
    rng = np.random.default_rng(SEED)

    # ----- Grid -----
    Nx, Ny, Nz = 128, 128, 128
    # 1 mm → cubo de 128 mm de lado (típico tamaño cabeza humana en k-Wave) con 128^3 voxeles
    dx = 1e-3
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])

    # ----- Transductor -----
    APERTURE = 25e-3
    THICKNESS = 3 * dx

    # ----- ROI -----
    ROI_RADIUS_MM = 10.0     # recomendado: 8–12 mm
    ROI_RADIUS_M = ROI_RADIUS_MM * 1e-3

    # ----- TUS -----
    F0 = 500e3
    N_CYCLES = 10
    A_MIN = 0.3e6
    A_MAX = 1.2e6

    # ----- Medio base -----
    c_water, rho_water, alpha_water = 1482.0, 994.0, 0.0126
    c_brain_mu, rho_brain_mu, alpha_brain_mu = 1546.0, 1046.0, 0.5

    # ----- Tiempo -----
    c_max = 3100.0
    domain_diag = np.sqrt((Nx*dx)**2 + (Ny*dx)**2 + (Nz*dx)**2)
    c_min = min(c_water, c_brain_mu)
    t_end = float(2.0 * domain_diag / c_min)
    CFL = 0.1

    ok_time, how = try_make_time(kgrid, c_max, CFL)
    print(
        f"🕒 makeTime: {how}" if ok_time else "🕒 makeTime no aplicado. Fallback.")

    dt, Nt, _ = get_dt_Nt_or_fallback(kgrid, dx, c_max, CFL, t_end)
    try:
        kgrid.dt = dt
        kgrid.Nt = Nt
    except Exception:
        pass
    print(f"🕒 Tiempo: dt={dt:.3e} | Nt={Nt}")

    sim_opts = SimulationOptions(
        save_to_disk=True, pml_inside=False, pml_size=10, data_cast="single")
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=True)

    # ----- Estadísticas -----
    saved = 0
    skipped = 0

    print(
        f"\n📦 Generando dataset BEST-CLINICAL+ROI (target={N_TARGET}) → {OUT_DIR}/")

    # bucle hasta guardar N_TARGET
    i = 0
    while saved < N_TARGET:
        print(f"\n=== Attempt {i:05d} | saved {saved}/{N_TARGET} ===")
        i += 1

        # ---- cráneo ----
        a_in, b_in, c_in, grosor = random_skull_params(rng)
        a_out, b_out, c_out = a_in + grosor, b_in + grosor, c_in + grosor

        warp = {
            "amp": float(rng.uniform(0.0e-3, 0.5e-3)),
            "freq": int(rng.integers(1, 3)),
            "seed": int(rng.integers(0, 1e9))
        }

        mask_inner = create_ellipsoid_mask(kgrid, a_in, b_in, c_in, warp=warp)
        mask_outer = create_ellipsoid_mask(
            kgrid, a_out, b_out, c_out, warp=warp)
        mask_brain = mask_inner
        mask_skull = mask_outer & (~mask_inner)

        if np.sum(mask_brain) < 1000 or np.sum(mask_skull) < 1000:
            skipped += 1
            continue

        # ---- profundidad real (mm) dentro del cerebro ----
        dist_vox = distance_transform_edt(mask_brain.astype(np.uint8))
        depth_mm_map = dist_vox * (dx * 1e3)

        # ---- foco balanceado por profundidad ----
        focus_pos, depth_class, depth_val_mm = pick_focus_by_depth(
            rng, mask_brain, depth_mm_map, kgrid,
            target="mix",
            shallow=(2, 6),
            mid=(6, 12),
            deep=(12, 30),
            margin_vox=3
        )

        # ---- ROI ----
        roi_focus = make_roi_brain_focus(
            kgrid, focus_pos, mask_brain, ROI_RADIUS_M)

        # ---- materiales por muestra (cráneo variado, cerebro leve variación) ----
        c_skull_now = float(rng.uniform(2600.0, 3100.0))
        rho_skull_now = float(rng.uniform(1700.0, 2100.0))
        alpha_skull_now = float(rng.uniform(12.0, 18.0))

        c_brain = float(rng.normal(c_brain_mu, 25.0))
        rho_brain = float(rng.normal(rho_brain_mu, 25.0))
        alpha_brain = float(
            np.clip(rng.normal(alpha_brain_mu, 0.08), 0.25, 0.9))

        sound_speed = np.ones((Nx, Ny, Nz), dtype=np.float32) * c_water
        density = np.ones((Nx, Ny, Nz), dtype=np.float32) * rho_water
        alpha_coeff = np.ones((Nx, Ny, Nz), dtype=np.float32) * alpha_water

        sound_speed[mask_brain] = c_brain
        density[mask_brain] = rho_brain
        alpha_coeff[mask_brain] = alpha_brain

        sound_speed[mask_skull] = c_skull_now
        density[mask_skull] = rho_skull_now
        alpha_coeff[mask_skull] = alpha_skull_now

        # ---- heterogeneidad espacial ----
        sound_speed = add_spatial_heterogeneity(
            rng, sound_speed, mask_skull, rel_std=0.02, smooth_iters=3)
        density = add_spatial_heterogeneity(
            rng, density,     mask_skull, rel_std=0.03, smooth_iters=3)
        alpha_coeff = add_spatial_heterogeneity(
            rng, alpha_coeff, mask_skull, rel_std=0.15, smooth_iters=3)

        sound_speed = add_spatial_heterogeneity(
            rng, sound_speed, mask_brain, rel_std=0.005, smooth_iters=2)
        density = add_spatial_heterogeneity(
            rng, density,     mask_brain, rel_std=0.005, smooth_iters=2)
        alpha_coeff = add_spatial_heterogeneity(
            rng, alpha_coeff, mask_brain, rel_std=0.04,  smooth_iters=2)

        medium = kWaveMedium(sound_speed=sound_speed, density=density,
                             alpha_coeff=alpha_coeff, alpha_power=1.1)

        # ---- transductor clínico (arriba + lateral), pegado a superficie, apuntando al foco ----
        # Mezcla controlada de dificultad: mayor max_polar => más oblicuo (pero aún clínico)
        max_polar_deg = float(rng.uniform(45.0, 75.0))  # 45–75°
        ok = False
        for attempt in range(80):
            source_pos, focus_pos2, u_surf, R_surf, gel = place_transducer_clinical(
                rng, focus_pos, a_out, b_out, c_out,
                dist_from_skin_range=(0.5e-3, 5.0e-3),
                max_polar_deg=max_polar_deg,
                min_z=0.20
            )

            R = float(np.linalg.norm(source_pos - focus_pos2))

            source_mask_bool = create_oriented_bowl(
                kgrid=kgrid,
                focus_pos=focus_pos2,
                transducer_center=source_pos,
                radius_curvature=R,
                aperture_diameter=APERTURE,
                thickness=THICKNESS
            )

            if np.sum(source_mask_bool) < 20:
                continue

            # colisión: el transductor no debe atravesar hueso
            if np.any(np.logical_and(source_mask_bool, mask_skull)):
                continue

            ok = True
            break

        if not ok:
            skipped += 1
            continue

        source_mask_u8 = source_mask_bool.astype(np.uint8)

        # ---- amplitud + señal ----
        A_source = float(rng.uniform(A_MIN, A_MAX))
        sig = make_tone_burst(dt, Nt, F0, N_CYCLES, A_source)

        src_idx = np.argwhere(source_mask_u8 > 0)
        n_src_pts = int(src_idx.shape[0])
        if n_src_pts == 0:
            skipped += 1
            continue

        source_p = np.tile(sig[None, :], (n_src_pts, 1)).astype(np.float32)

        source = kSource()
        source.p_mask = source_mask_u8
        source.p = source_p
        source = make_source_compatible_for_time_signal(source, source_mask_u8)

        sensor = kSensor()
        sensor.mask = np.ones((Nx, Ny, Nz), dtype=np.uint8)
        sensor.record = ["p_max"]

        # ---- sim ----
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

        # ---- guardar ----
        out_path = os.path.join(OUT_DIR, f"sample_{saved:04d}.npz")
        depth_class_int = {"shallow": 0, "mid": 1, "deep": 2}[depth_class]

        np.savez_compressed(
            out_path,

            # inputs
            mask_skull=mask_skull.astype(np.uint8),
            mask_brain=mask_brain.astype(np.uint8),
            sound_speed=sound_speed.astype(np.float32),
            density=density.astype(np.float32),
            alpha_coeff=alpha_coeff.astype(np.float32),

            source_pos=source_pos.astype(np.float32),
            focus_pos=focus_pos2.astype(np.float32),
            aperture=np.float32(APERTURE),
            radius_curvature=np.float32(R),
            source_mask=source_mask_u8.astype(np.uint8),

            # ROI para loss
            roi_focus=roi_focus.astype(np.uint8),
            roi_radius_mm=np.float32(ROI_RADIUS_MM),

            # TUS params
            f0_hz=np.float32(F0),
            n_cycles=np.int32(N_CYCLES),
            A_source=np.float32(A_source),
            dt=np.float32(dt),
            Nt=np.int32(Nt),
            cfl=np.float32(CFL),

            # outputs
            p_max=p_max.astype(np.float32),
            p_max_norm=p_max_norm.astype(np.float32),

            # grid
            dx=np.float32(dx),
            Nx=np.int32(Nx), Ny=np.int32(Ny), Nz=np.int32(Nz),

            # skull params
            a_in=np.float32(a_in), b_in=np.float32(b_in), c_in=np.float32(c_in),
            grosor=np.float32(grosor),

            # meta útil
            depth_class=np.int32(depth_class_int),
            depth_mm=np.float32(depth_val_mm),
            max_polar_deg=np.float32(max_polar_deg),
            gel_mm=np.float32(gel * 1e3),
            c_skull_now=np.float32(c_skull_now),
            rho_skull_now=np.float32(rho_skull_now),
            alpha_skull_now=np.float32(alpha_skull_now),
            warp_amp_mm=np.float32(warp["amp"] * 1e3),
            warp_freq=np.int32(warp["freq"]),
        )

        print(f"💾 Saved {saved}: {out_path}")
        print(
            f"    depth={depth_class} ({depth_val_mm:.1f} mm) | max_polar={max_polar_deg:.1f}° | gel={gel*1e3:.2f} mm | src_pts={n_src_pts}")
        saved += 1

        # preview primeras 2
        if saved <= 2:
            slice_z = Nz // 2
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(mask_skull[:, :, slice_z].T,
                       cmap="gray", origin="lower")
            plt.imshow(source_mask_u8[:, :, slice_z].T,
                       cmap="hot", alpha=0.5, origin="lower")
            plt.title(f"Saved {saved-1} | Skull + Source (Z={slice_z})")

            plt.subplot(1, 2, 2)
            vmax = np.percentile(p_max_norm, 99.5)
            plt.imshow(p_max_norm[:, :, slice_z].T,
                       cmap="jet", origin="lower", vmin=0, vmax=vmax)
            plt.colorbar(label="p_max_norm")
            plt.contour(roi_focus[:, :, slice_z].T,
                        colors="white", linewidths=0.7, alpha=0.85)
            plt.title(f"Saved {saved-1} | p_max_norm + ROI")
            plt.tight_layout()
            plt.show()

    print(
        f"\n✅ DONE. Guardados: {saved} | Skips: {skipped} | Skip rate ~ {100.0*skipped/max(1,(saved+skipped)):.1f}%")
    print("ROI está guardada para usarla en el loss ponderado (NO como canal).")


if __name__ == "__main__":
    main()
