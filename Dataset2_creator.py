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
# 1) TRANSDUCER CREATION (precomputed XYZ version)
# ==============================================================================

def create_oriented_bowl_xyz(X, Y, Z, focus_pos, transducer_center,
                             radius_curvature, aperture_diameter, thickness):
    F = np.array(focus_pos, dtype=np.float32)
    T = np.array(transducer_center, dtype=np.float32)

    axis_vec = T - F
    axis_len = np.linalg.norm(axis_vec)
    axis_dir = axis_vec / (axis_len + 1e-12)

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
# 2) ROBUST TIME
# ==============================================================================

def try_make_time(kgrid, c_max, cfl):
    candidates = [
        ("makeTime(c_max, cfl=...)", lambda: kgrid.makeTime(c_max, cfl=cfl)),
        ("makeTime(c_max, CFL=...)", lambda: kgrid.makeTime(c_max, CFL=cfl)),
        ("makeTime(c_max, cfl) positional", lambda: kgrid.makeTime(c_max, cfl)),
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
        raise RuntimeError(f"Invalid time: dt={dt}, Nt={Nt}")

    t = np.arange(Nt, dtype=np.float32) * np.float32(dt)
    return dt, Nt, t


# ==============================================================================
# 3) TUS-LIKE SIGNAL
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
# 4) kSource COMPATIBILITY PATCH
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
# 5) SMOOTH 3D NOISE (FFT low-pass)
# ==============================================================================

def _smooth_noise_3d_fft(shape, rng, corr_len_vox=12.0, amplitude=1.0):
    n = rng.standard_normal(shape).astype(np.float32)

    Nx, Ny, Nz = shape
    kx = np.fft.fftfreq(Nx).astype(np.float32)[:, None, None]
    ky = np.fft.fftfreq(Ny).astype(np.float32)[None, :, None]
    kz = np.fft.fftfreq(Nz).astype(np.float32)[None, None, :]

    k2 = (kx * kx + ky * ky + kz * kz)
    sigma = float(corr_len_vox) / 2.0
    filt = np.exp(-(2.0 * (np.pi**2)) * (sigma**2) * k2).astype(np.float32)

    s = np.fft.ifftn(np.fft.fftn(n) * filt).real.astype(np.float32)
    s -= s.mean()
    s /= (np.max(np.abs(s)) + 1e-8)
    return amplitude * s


# ==============================================================================
# 6) HUMAN-SCALED SKULL AXES
# ==============================================================================

def sample_human_scaled_skull_axes(rng):
    b_out = float(rng.uniform(50.0e-3, 52.0e-3))       # AP semiaxis
    a_out = float(b_out * rng.uniform(0.74, 0.78))     # LR semiaxis
    c_out = float(b_out * rng.uniform(0.74, 0.78))     # SI semiaxis
    return a_out, b_out, c_out


# ==============================================================================
# 7) PROCEDURAL HUMAN-SCALED SKULL
# ==============================================================================

def create_skull_masks_procedural_human_scaled(
    X, Y, Z,
    kgrid,
    a_out, b_out, c_out,
    thickness_min=3.0e-3,
    thickness_max=8.0e-3,
    rough_amp=0.20e-3,
    rough_corr_len_vox=18.0,
    shape_amp=0.80e-3,
    shape_corr_len_vox=40.0,
    flatten_strength=0.40e-3,
    base_cut_strength=0.15e-3,
    p_xy=2.35,
    center_shift_mm=0.8,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    Nx, Ny, Nz = X.shape

    shift = rng.uniform(
        -center_shift_mm * 1e-3,
        center_shift_mm * 1e-3,
        size=3
    ).astype(np.float32)

    Xc = (X - shift[0]).astype(np.float32)
    Yc = (Y - shift[1]).astype(np.float32)
    Zc = (Z - shift[2]).astype(np.float32)

    R = np.sqrt(Xc * Xc + Yc * Yc + Zc * Zc) + 1e-12
    ux = (Xc / R).astype(np.float32)
    uy = (Yc / R).astype(np.float32)
    uz = (Zc / R).astype(np.float32)

    q = rng.normal(size=4).astype(np.float32)
    q /= (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    Rm = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)

    uxr = (Rm[0, 0] * ux + Rm[0, 1] * uy + Rm[0, 2] * uz).astype(np.float32)
    uyr = (Rm[1, 0] * ux + Rm[1, 1] * uy + Rm[1, 2] * uz).astype(np.float32)
    uzr = (Rm[2, 0] * ux + Rm[2, 1] * uy + Rm[2, 2] * uz).astype(np.float32)

    t_noise = _smooth_noise_3d_fft(
        (Nx, Ny, Nz), rng, corr_len_vox=20.0, amplitude=1.0)
    surf_noise = _smooth_noise_3d_fft(
        (Nx, Ny, Nz), rng, corr_len_vox=rough_corr_len_vox, amplitude=1.0)
    shape_noise = _smooth_noise_3d_fft(
        (Nx, Ny, Nz), rng, corr_len_vox=shape_corr_len_vox, amplitude=1.0)

    thickness_mean = float(rng.uniform(4.8e-3, 6.5e-3))

    lateral = np.abs(uxr).astype(np.float32)
    anterior = np.clip(uyr, 0.0, 1.0).astype(np.float32)
    posterior = np.clip(-uyr, 0.0, 1.0).astype(np.float32)
    superior = np.clip(uzr, 0.0, 1.0).astype(np.float32)
    inferior = np.clip(-uzr, 0.0, 1.0).astype(np.float32)

    t_regional = (
        thickness_mean
        - 1.5e-3 * lateral
        + 0.7e-3 * posterior
        + 0.4e-3 * anterior
        + 0.6e-3 * superior
        - 0.2e-3 * inferior
    ).astype(np.float32)

    t_local = (t_regional + 0.45e-3 * t_noise).astype(np.float32)
    t_local = np.clip(t_local, thickness_min, thickness_max).astype(np.float32)

    surf_disp = (rough_amp * surf_noise).astype(np.float32)
    shape_disp = (shape_amp * shape_noise).astype(np.float32)

    z_norm = (Zc / (np.max(np.abs(np.asarray(kgrid.z_vec).reshape(-1))
                           ) + 1e-12)).astype(np.float32)
    flatten = (-flatten_strength * (z_norm ** 2)).astype(np.float32)
    base_cut = (
        -base_cut_strength
        * (z_norm < -0.35).astype(np.float32)
        * ((-z_norm - 0.35) / 0.65)
    ).astype(np.float32)

    p_xy = float(p_xy)
    r_xy = (
        (np.abs(Xc) / (a_out + 1e-12)) ** p_xy
        + (np.abs(Yc) / (b_out + 1e-12)) ** p_xy
    ) ** (1.0 / p_xy)

    r_z = np.abs(Zc) / (c_out + 1e-12)
    r = np.sqrt(r_xy * r_xy + r_z * r_z).astype(np.float32)

    Rref = float((a_out + b_out + c_out) / 3.0)
    d_in = (1.0 - r) * Rref + surf_disp + shape_disp + flatten + base_cut

    mask_head = (d_in >= 0.0)
    mask_skull = mask_head & (d_in <= t_local)
    mask_brain = mask_head & (d_in > t_local)

    return mask_skull, mask_brain, t_local, thickness_mean, shift


# ==============================================================================
# 8) GRID / INDEX HELPERS
# ==============================================================================

def _coords_to_indices(x, y, z, kgrid):
    dx = float(kgrid.dx)

    x_vec = np.asarray(kgrid.x_vec).reshape(-1)
    y_vec = np.asarray(kgrid.y_vec).reshape(-1)
    z_vec = np.asarray(kgrid.z_vec).reshape(-1)

    x0 = float(x_vec[0])
    y0 = float(y_vec[0])
    z0 = float(z_vec[0])

    ix = np.round((x - x0) / dx).astype(np.int32)
    iy = np.round((y - y0) / dx).astype(np.int32)
    iz = np.round((z - z0) / dx).astype(np.int32)

    ix = np.clip(ix, 0, len(x_vec) - 1)
    iy = np.clip(iy, 0, len(y_vec) - 1)
    iz = np.clip(iz, 0, len(z_vec) - 1)
    return ix, iy, iz


def mask_touches_pml(mask, pml_size):
    p = int(pml_size)
    if p <= 0:
        return False
    if np.any(mask[:p, :, :]) or np.any(mask[-p:, :, :]):
        return True
    if np.any(mask[:, :p, :]) or np.any(mask[:, -p:, :]):
        return True
    if np.any(mask[:, :, :p]) or np.any(mask[:, :, -p:]):
        return True
    return False


def safe_reshape_pmax(p_max, Nx, Ny, Nz):
    if getattr(p_max, "ndim", 0) == 1:
        try:
            return p_max.reshape((Nx, Ny, Nz), order="F")
        except Exception:
            return p_max.reshape((Nx, Ny, Nz), order="C")
    return p_max


# ==============================================================================
# 9) TARGET DEPTH STRATEGY
# ==============================================================================

DEPTH_BINS = [
    ("superficial", 8.0, 14.0),
    ("intermediate", 14.0, 24.0),
    ("deep", 24.0, 34.0),
]


def build_brain_depth_map(mask_brain, dx):
    return distance_transform_edt(mask_brain) * dx


def sample_target_from_depth_bin(mask_brain, depth_map, kgrid, rng,
                                 depth_bin, a_out, b_out, c_out):
    name, dmin_mm, dmax_mm = depth_bin
    dmin = dmin_mm * 1e-3
    dmax = dmax_mm * 1e-3

    valid_mask = (
        (mask_brain > 0)
        & (depth_map >= dmin)
        & (depth_map < dmax)
    )

    valid = np.argwhere(valid_mask)
    if valid.shape[0] == 0:
        raise RuntimeError(f"No valid targets found for depth bin '{name}'.")

    x_vec = np.asarray(kgrid.x_vec).reshape(-1)
    y_vec = np.asarray(kgrid.y_vec).reshape(-1)
    z_vec = np.asarray(kgrid.z_vec).reshape(-1)

    x = x_vec[valid[:, 0]]
    y = y_vec[valid[:, 1]]
    z = z_vec[valid[:, 2]]

    region_ok = (
        (np.abs(x) <= 0.95 * a_out)
        & (np.abs(y) <= 0.95 * b_out)
        & (z >= -0.55 * c_out)
        & (z <= 0.75 * c_out)
    )
    region_ok = np.asarray(region_ok).reshape(-1)

    valid = valid[region_ok]
    if valid.shape[0] == 0:
        raise RuntimeError(
            f"No anatomically plausible targets found for depth bin '{name}'.")

    idx = int(rng.integers(valid.shape[0]))
    ix, iy, iz = valid[idx]

    F = np.array([
        x_vec[ix],
        y_vec[iy],
        z_vec[iz]
    ], dtype=np.float32)

    depth_m = float(depth_map[ix, iy, iz])
    return F, (int(ix), int(iy), int(iz)), depth_m, name


# ==============================================================================
# 10) SURFACE SEARCH FROM TARGET ALONG A DIRECTION
# ==============================================================================

def estimate_outer_skull_surface_from_focus(mask_skull, kgrid, focus_pos, u_dir,
                                            max_dist=None, dr=None):
    if max_dist is None:
        max_dist = 0.08
    if dr is None:
        dr = float(kgrid.dx) / 2.0

    s = np.arange(0.0, max_dist + dr, dr, dtype=np.float32)

    xs = focus_pos[0] + s * np.float32(u_dir[0])
    ys = focus_pos[1] + s * np.float32(u_dir[1])
    zs = focus_pos[2] + s * np.float32(u_dir[2])

    ix, iy, iz = _coords_to_indices(xs, ys, zs, kgrid)
    vals = mask_skull[ix, iy, iz]

    hit = np.where(vals > 0)[0]
    if hit.size == 0:
        return None, None

    last = hit[-1]
    surf_point = np.array([xs[last], ys[last], zs[last]], dtype=np.float32)
    surf_dist = float(s[last])

    return surf_point, surf_dist


# ==============================================================================
# 11) REALISTIC APPROACH DIRECTION SAMPLER
# ==============================================================================

def sample_approach_direction(rng):
    mode = rng.choice(["lateral", "frontback", "superior"],
                      p=[0.40, 0.30, 0.30])

    if mode == "lateral":
        sign = rng.choice([-1.0, 1.0])
        v = np.array([
            sign * rng.uniform(0.8, 1.2),
            rng.normal(0.0, 0.30),
            abs(rng.normal(0.25, 0.20))
        ], dtype=np.float32)

    elif mode == "frontback":
        sign = rng.choice([-1.0, 1.0])
        v = np.array([
            rng.normal(0.0, 0.30),
            sign * rng.uniform(0.8, 1.2),
            abs(rng.normal(0.25, 0.20))
        ], dtype=np.float32)

    else:
        v = np.array([
            rng.normal(0.0, 0.35),
            rng.normal(0.0, 0.35),
            abs(rng.normal(1.0, 0.25))
        ], dtype=np.float32)

    v /= (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float32), mode


# ==============================================================================
# 12) TARGET-DRIVEN TRANSDUCER PLACEMENT
# ==============================================================================

def place_transducer_for_target(mask_skull, X, Y, Z, kgrid, focus_pos, rng,
                                radius_curvature, aperture_diameter,
                                bowl_thickness, gap_min, gap_max,
                                pml_size, n_tries=220):

    R = float(radius_curvature)
    t = float(bowl_thickness)

    for _ in range(n_tries):
        u_out, approach_mode = sample_approach_direction(rng)

        surf_point, surf_dist = estimate_outer_skull_surface_from_focus(
            mask_skull=mask_skull,
            kgrid=kgrid,
            focus_pos=focus_pos,
            u_dir=u_out,
            max_dist=0.08,
            dr=float(kgrid.dx) / 2.0
        )
        if surf_point is None:
            continue

        gap_real = (R - 0.5 * t) - surf_dist
        if not (gap_min <= gap_real <= gap_max):
            continue

        T = (focus_pos + R * u_out).astype(np.float32)

        source_mask_pre = create_oriented_bowl_xyz(
            X=X, Y=Y, Z=Z,
            focus_pos=focus_pos,
            transducer_center=T,
            radius_curvature=R,
            aperture_diameter=float(aperture_diameter),
            thickness=t
        )

        n_pre = int(np.sum(source_mask_pre))
        if n_pre <= 0:
            continue

        if mask_touches_pml(source_mask_pre, pml_size):
            continue

        source_mask_post = source_mask_pre & (~mask_skull)
        n_post = int(np.sum(source_mask_post))
        if n_post <= 0:
            continue

        frac_kept = n_post / max(n_pre, 1)
        if frac_kept < 0.995:
            continue

        r_T = float(np.linalg.norm(T))

        return (
            T,
            source_mask_post.astype(np.uint8),
            float(gap_real),
            surf_point.astype(np.float32),
            float(surf_dist),
            u_out.astype(np.float32),
            approach_mode,
            r_T
        )

    raise RuntimeError(
        "Could not place a valid target-driven near-contact transducer.")


# ==============================================================================
# 13) TARGET SELECTION + PLACEMENT PIPELINE
# ==============================================================================

def choose_depth_bin_balanced(sim_idx):
    return DEPTH_BINS[sim_idx % len(DEPTH_BINS)]


def sample_target_and_place_transducer(mask_skull, mask_brain, depth_map,
                                       X, Y, Z, kgrid, rng,
                                       a_out, b_out, c_out,
                                       sim_idx,
                                       radius_curvature, aperture_diameter,
                                       bowl_thickness, gap_min, gap_max,
                                       pml_size,
                                       max_target_tries=80):

    preferred_bin = choose_depth_bin_balanced(sim_idx)
    ordered_bins = [preferred_bin] + \
        [b for b in DEPTH_BINS if b != preferred_bin]

    last_error = None

    for depth_bin in ordered_bins:
        for _ in range(max_target_tries):
            try:
                focus_pos, focus_idx, focus_depth_m, depth_name = sample_target_from_depth_bin(
                    mask_brain=mask_brain,
                    depth_map=depth_map,
                    kgrid=kgrid,
                    rng=rng,
                    depth_bin=depth_bin,
                    a_out=a_out,
                    b_out=b_out,
                    c_out=c_out
                )
            except Exception as e:
                last_error = e
                break

            try:
                (
                    source_pos,
                    source_mask_u8,
                    gap_real,
                    surf_point,
                    surf_dist,
                    beam_dir,
                    approach_mode,
                    r_T
                ) = place_transducer_for_target(
                    mask_skull=mask_skull,
                    X=X, Y=Y, Z=Z,
                    kgrid=kgrid,
                    focus_pos=focus_pos,
                    rng=rng,
                    radius_curvature=radius_curvature,
                    aperture_diameter=aperture_diameter,
                    bowl_thickness=bowl_thickness,
                    gap_min=gap_min,
                    gap_max=gap_max,
                    pml_size=pml_size,
                    n_tries=220
                )

                return {
                    "focus_pos": focus_pos,
                    "focus_idx": focus_idx,
                    "focus_depth_m": float(focus_depth_m),
                    "focus_depth_bin": depth_name,
                    "source_pos": source_pos,
                    "source_mask_u8": source_mask_u8,
                    "gap_real": float(gap_real),
                    "surface_point": surf_point,
                    "surface_dist": float(surf_dist),
                    "beam_dir": beam_dir,
                    "approach_mode": approach_mode,
                    "r_T": float(r_T),
                }

            except Exception as e:
                last_error = e
                continue

    raise RuntimeError(
        f"Failed to sample target + transducer placement. Last error: {last_error}")


# ==============================================================================
# 14) PREVIEW
# ==============================================================================

def preview_geometry(mask_skull, mask_brain, source_mask_u8, focus_idx, i):
    Nx, Ny, Nz = mask_skull.shape
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    fx, fy, fz = focus_idx

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].imshow(mask_brain[cx, :, :].T, origin="lower",
                 cmap="Blues", alpha=0.35, interpolation="nearest")
    ax[0].imshow(mask_skull[cx, :, :].T, origin="lower",
                 cmap="gray", alpha=0.9, interpolation="nearest")
    ax[0].imshow(source_mask_u8[cx, :, :].T, origin="lower",
                 cmap="hot", alpha=0.65, interpolation="nearest")
    if fx == cx:
        ax[0].plot(fy, fz, "co", markersize=6)
    ax[0].set_title(f"Sagittal (x={cx})")

    ax[1].imshow(mask_brain[:, cy, :].T, origin="lower",
                 cmap="Blues", alpha=0.35, interpolation="nearest")
    ax[1].imshow(mask_skull[:, cy, :].T, origin="lower",
                 cmap="gray", alpha=0.9, interpolation="nearest")
    ax[1].imshow(source_mask_u8[:, cy, :].T, origin="lower",
                 cmap="hot", alpha=0.65, interpolation="nearest")
    if fy == cy:
        ax[1].plot(fx, fz, "co", markersize=6)
    ax[1].set_title(f"Coronal (y={cy})")

    ax[2].imshow(mask_brain[:, :, cz].T, origin="lower",
                 cmap="Blues", alpha=0.35, interpolation="nearest")
    ax[2].imshow(mask_skull[:, :, cz].T, origin="lower",
                 cmap="gray", alpha=0.9, interpolation="nearest")
    ax[2].imshow(source_mask_u8[:, :, cz].T, origin="lower",
                 cmap="hot", alpha=0.65, interpolation="nearest")
    if fz == cz:
        ax[2].plot(fx, fy, "co", markersize=6)
    ax[2].set_title(f"Axial (z={cz})")

    plt.suptitle(f"Sim {i} | Geometry preview", y=1.02)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 15) MAIN
# ==============================================================================

def main():
    OUT_DIR = "dataset_TUS_dx05_256_humanScaled_A58_R54_targetDepthBins_PMLinside"
    os.makedirs(OUT_DIR, exist_ok=True)

    N_SIMS = 1000
    N_PREVIEW = 2
    SEED = 123
    rng = np.random.default_rng(SEED)

    Nx, Ny, Nz = 256, 256, 256
    dx = 0.5e-3
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])

    x_vec = np.asarray(kgrid.x_vec).reshape(-1).astype(np.float32)
    y_vec = np.asarray(kgrid.y_vec).reshape(-1).astype(np.float32)
    z_vec = np.asarray(kgrid.z_vec).reshape(-1).astype(np.float32)

    X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec, indexing="ij")

    PML_SIZE = 10
    half_interior = (Nx / 2 - PML_SIZE) * dx
    margin_to_pml = 2.0e-3

    APERTURE = 58e-3
    R_CONST = 54e-3
    BOWL_THICKNESS = 2 * dx

    GAP_MIN = 2.0e-3
    GAP_MAX = 3.0e-3

    TMIN, TMAX = 3.0e-3, 8.0e-3

    c_water, rho_water, alpha_water = 1482.0, 994.0, 0.0126
    c_skull, rho_skull, alpha_skull = 2800.0, 1850.0, 15.0
    c_brain, rho_brain, alpha_brain = 1546.0, 1046.0, 0.5
    c_max = max(c_skull, c_brain, c_water)

    F0 = 500e3
    N_CYCLES = 10
    A_MIN = 0.3e6
    A_MAX = 1.2e6

    if 52e-3 + 2.0e-3 > (half_interior - margin_to_pml):
        raise RuntimeError(
            f"Skull too large for the interior domain. half_interior={half_interior * 1e3:.1f} mm"
        )

    domain_diag = np.sqrt((Nx * dx) ** 2 + (Ny * dx) ** 2 + (Nz * dx) ** 2)
    c_min = min(c_water, c_brain)
    t_end = float(2.0 * domain_diag / c_min)
    CFL = 0.1

    ok_time, how = try_make_time(kgrid, c_max, CFL)
    print(
        f"=R makeTime: {how}" if ok_time else "=R makeTime not applied. Using fallback dt/Nt.")

    dt, Nt, _ = get_dt_Nt_or_fallback(kgrid, dx, c_max, CFL, t_end)
    try:
        kgrid.dt = dt
        kgrid.Nt = Nt
    except Exception:
        pass
    print(f"=R Time: dt={dt:.3e} s | Nt={Nt}")

    sim_opts = SimulationOptions(
        save_to_disk=True,
        pml_inside=True,
        pml_size=PML_SIZE,
        data_cast="single",
    )
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=True)

    print(f"\n=� Generating dataset ({N_SIMS} sims) � {OUT_DIR}/")
    print(
        f"=' Grid: {Nx}x{Ny}x{Nz} | dx={dx * 1e3:.2f} mm | PML inside={PML_SIZE}")
    print(
        f"=' Transducer: aperture={APERTURE * 1e3:.1f} mm | ROC={R_CONST * 1e3:.1f} mm | thickness={BOWL_THICKNESS * 1e3:.1f} mm")
    print(f"=' Coupling gap: {GAP_MIN * 1e3:.1f}-{GAP_MAX * 1e3:.1f} mm")
    print("=' Depth bins (mm from nearest skull): superficial 8-14 | intermediate 14-24 | deep 24-34")

    for i in range(N_SIMS):
        print(f"\n=== Sim {i:04d}/{N_SIMS - 1:04d} ===")

        a_out, b_out, c_out = sample_human_scaled_skull_axes(rng)

        mask_skull, mask_brain, t_local, t_mean, skull_shift = create_skull_masks_procedural_human_scaled(
            X=X, Y=Y, Z=Z,
            kgrid=kgrid,
            a_out=a_out, b_out=b_out, c_out=c_out,
            thickness_min=TMIN, thickness_max=TMAX,
            rough_amp=0.20e-3,
            rough_corr_len_vox=18.0,
            shape_amp=0.80e-3,
            shape_corr_len_vox=40.0,
            flatten_strength=0.40e-3,
            base_cut_strength=0.15e-3,
            p_xy=2.35,
            center_shift_mm=0.8,
            rng=rng
        )

        depth_map = build_brain_depth_map(mask_brain, dx)

        placement = sample_target_and_place_transducer(
            mask_skull=mask_skull,
            mask_brain=mask_brain,
            depth_map=depth_map,
            X=X, Y=Y, Z=Z,
            kgrid=kgrid,
            rng=rng,
            a_out=a_out, b_out=b_out, c_out=c_out,
            sim_idx=i,
            radius_curvature=R_CONST,
            aperture_diameter=APERTURE,
            bowl_thickness=BOWL_THICKNESS,
            gap_min=GAP_MIN,
            gap_max=GAP_MAX,
            pml_size=PML_SIZE,
            max_target_tries=80
        )

        focus_pos = placement["focus_pos"]
        focus_idx = placement["focus_idx"]
        focus_depth_m = placement["focus_depth_m"]
        focus_depth_bin = placement["focus_depth_bin"]
        source_pos = placement["source_pos"]
        source_mask_u8 = placement["source_mask_u8"]
        gap_real = placement["gap_real"]
        surface_point = placement["surface_point"]
        surface_dist = placement["surface_dist"]
        beam_dir = placement["beam_dir"]
        approach_mode = placement["approach_mode"]
        r_T = placement["r_T"]

        n_src_pts = int(np.sum(source_mask_u8))
        if n_src_pts <= 0:
            raise RuntimeError(
                "Invalid source mask after target-driven placement.")

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
            f"skull_axes(mm): a={a_out * 1e3:.2f}, b={b_out * 1e3:.2f}, c={c_out * 1e3:.2f} | "
            f"t_mean={t_mean * 1e3:.2f} | "
            f"target={focus_depth_bin} ({focus_depth_m * 1e3:.2f} mm) | "
            f"gap={gap_real * 1e3:.2f} | "
            f"surface_dist={surface_dist * 1e3:.2f} | "
            f"r_T={r_T * 1e3:.2f} | "
            f"approach={approach_mode} | "
            f"src_pts={n_src_pts}"
        )

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

        out_path = os.path.join(OUT_DIR, f"sample_{i:04d}.npz")

        save_dict = {
            "mask_skull": mask_skull.astype(np.uint8),
            "mask_brain": mask_brain.astype(np.uint8),
            "source_mask": source_mask_u8,

            "sound_speed": sound_speed.astype(np.float32),
            "density": density.astype(np.float32),
            "alpha_coeff": alpha_coeff.astype(np.float32),

            "source_pos": source_pos.astype(np.float32),
            "focus_pos": focus_pos.astype(np.float32),
            "focus_idx": np.array(focus_idx, dtype=np.int32),
            "surface_point": surface_point.astype(np.float32),
            "beam_dir": beam_dir.astype(np.float32),

            "aperture": np.float32(APERTURE),
            "radius_curvature": np.float32(R_CONST),
            "bowl_thickness": np.float32(BOWL_THICKNESS),
            "r_T": np.float32(r_T),
            "gap": np.float32(gap_real),
            "surface_dist": np.float32(surface_dist),

            "focus_depth_m": np.float32(focus_depth_m),
            "focus_depth_mm": np.float32(focus_depth_m * 1e3),
            "focus_depth_bin": np.array(focus_depth_bin),
            "approach_mode": np.array(approach_mode),

            "f0_hz": np.float32(F0),
            "n_cycles": np.int32(N_CYCLES),
            "A_source": np.float32(A_source),
            "dt": np.float32(dt),
            "Nt": np.int32(Nt),
            "cfl": np.float32(CFL),

            "a_out": np.float32(a_out),
            "b_out": np.float32(b_out),
            "c_out": np.float32(c_out),
            "skull_shift": skull_shift.astype(np.float32),
            "skull_thickness_min": np.float32(TMIN),
            "skull_thickness_max": np.float32(TMAX),
            "skull_thickness_mean": np.float32(t_mean),
            "p_xy": np.float32(2.35),

            "p_max": p_max.astype(np.float32),
            "p_max_norm": p_max_norm.astype(np.float32),

            "dx": np.float32(dx),
            "Nx": np.int32(Nx),
            "Ny": np.int32(Ny),
            "Nz": np.int32(Nz),
            "pml_size": np.int32(PML_SIZE),
            "pml_inside": np.int32(1),
        }

        np.savez_compressed(out_path, **save_dict)
        print(f"=� Saved: {out_path}")

        if i < N_PREVIEW:
            preview_geometry(
                mask_skull=mask_skull,
                mask_brain=mask_brain,
                source_mask_u8=source_mask_u8,
                focus_idx=focus_idx,
                i=i
            )

    print("\n Dataset generated.")
    print("   The AI should train with p_max_norm.")
    print("   To recover Pa: p_max_pred = p_max_norm_pred * A_source.")


if __name__ == "__main__":
    main()