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
# 1) GEOMETRÍA DEL TRANSDUCTOR
# ==============================================================================


def create_oriented_bowl(kgrid, focus_pos, transducer_center, radius_curvature, aperture_diameter, thickness):
    F = np.array(focus_pos, dtype=float)
    T = np.array(transducer_center, dtype=float)
    axis_vec = T - F
    axis_dir = axis_vec / (np.linalg.norm(axis_vec) + 1e-12)


    X, Y, Z = np.meshgrid(kgrid.x_vec, kgrid.y_vec, kgrid.z_vec, indexing="ij")
    dist_to_focus = np.sqrt((X - F[0])**2 + (Y - F[1])**2 + (Z - F[2])**2)
    shell_mask = np.abs(dist_to_focus - radius_curvature) <= (thickness / 2)


    vx, vy, vz = X - F[0], Y - F[1], Z - F[2]
    dot = vx * axis_dir[0] + vy * axis_dir[1] + vz * axis_dir[2]
    cos_theta = dot / (dist_to_focus + 1e-12)


    half_ap = aperture_diameter / 2.0
    min_cos = np.cos(np.arcsin(np.clip(half_ap / (radius_curvature + 1e-12), -1, 1)))
    return shell_mask & (cos_theta >= min_cos)


# ==============================================================================
# 2) CRÁNEO PROCEDURAL Y RUIDO
# ==============================================================================


def sample_skull_axes_for_fixed_rT_gap_ovoid(rng, r_T, gap_min, gap_max):
    skull_outer_max = float(rng.uniform(r_T - gap_max, r_T - gap_min))
    r_ap = float(rng.uniform(1.08, 1.18))
    r_si = float(rng.uniform(0.95, 1.05))
    b_out = skull_outer_max
    a_out = b_out / r_ap
    c_out = a_out * r_si
    return a_out, b_out, c_out


def _smooth_noise_3d_fft(shape, rng, corr_len_vox=12.0, amplitude=1.0):
    n = rng.standard_normal(shape).astype(np.float32)
    Nx, Ny, Nz = shape
    kx = np.fft.fftfreq(Nx)[:, None, None]
    ky = np.fft.fftfreq(Ny)[None, :, None]
    kz = np.fft.fftfreq(Nz)[None, None, :]
    k2 = (kx**2 + ky**2 + kz**2)
    filt = np.exp(-(2.0 * (np.pi**2)) * ((corr_len_vox/2.0)**2) * k2)
    s = np.fft.ifftn(np.fft.fftn(n) * filt).real
    s -= s.mean()
    s /= (np.max(np.abs(s)) + 1e-8)
    return (amplitude * s).astype(np.float32)


def create_skull_masks_procedural_TAC(kgrid, a_out, b_out, c_out, t_min, t_max, rng):
    X, Y, Z = np.meshgrid(kgrid.x_vec, kgrid.y_vec, kgrid.z_vec, indexing="ij")
    p_xy = 2.25
    r_xy = ((np.abs(X)/a_out)**p_xy + (np.abs(Y)/b_out)**p_xy)**(1.0/p_xy)
    r_z = np.abs(Z)/c_out
    r = np.sqrt(r_xy**2 + r_z**2)
    Rref = (a_out + b_out + c_out) / 3.0


    t_noise = _smooth_noise_3d_fft(X.shape, rng, corr_len_vox=20.0)
    thickness_mean = float(rng.uniform(t_min, t_max))
    t_local = np.clip(thickness_mean + 1.2e-3 * t_noise, t_min, t_max)
    
    surf_disp = 0.35e-3 * _smooth_noise_3d_fft(X.shape, rng, 16.0)
    d_in = (1.0 - r) * Rref + surf_disp
    
    mask_head = (d_in >= 0.0)
    return (mask_head & (d_in <= t_local)), (mask_head & (d_in > t_local)), t_local, thickness_mean


# ==============================================================================
# 3) UTILIDADES FÍSICAS Y K-WAVE
# ==============================================================================


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
        try: delattr(source, "p0")
        except: pass
    return source


def safe_reshape_pmax(p_max, Nx, Ny, Nz):
    try: return p_max.reshape((Nx, Ny, Nz), order="F")
    except: return p_max.reshape((Nx, Ny, Nz), order="C")


# ==============================================================================
# 4) VISUALIZACIÓN DIAGNÓSTICA (JET MAP)
# ==============================================================================


def preview_results(mask_skull, source_mask, p_max_norm, i, dx, Nx, Ny, Nz):
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    ext = [-Nx*dx*500, Nx*dx*500, -Ny*dx*500, Ny*dx*500] 
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    
    slices_geo = [mask_skull[cx,:,:].T, mask_skull[:,cy,:].T, mask_skull[:,:,cz].T]
    slices_src = [source_mask[cx,:,:].T, source_mask[:,cy,:].T, source_mask[:,:,cz].T]
    slices_p = [p_max_norm[cx,:,:].T, p_max_norm[:,cy,:].T, p_max_norm[:,:,cz].T]
    titles = ["Sagital (X=0)", "Coronal (Y=0)", "Axial (Z=0)"]


    for j in range(3):
        ax[0, j].imshow(slices_geo[j], extent=ext, origin="lower", cmap="gray")
        ax[0, j].imshow(slices_src[j], extent=ext, origin="lower", cmap="hot", alpha=0.4)
        ax[0, j].set_title(f"Geometría {titles[j]}")
        ax[0, j].set_ylabel("mm")
        
        im = ax[1, j].imshow(slices_p[j], extent=ext, origin="lower", cmap="jet", vmax=np.max(p_max_norm)*0.8)
        ax[1, j].set_title(f"Presión {titles[j]}")
        ax[1, j].set_xlabel("mm")
        ax[1, j].set_ylabel("mm")
        plt.colorbar(im, ax=ax[1, j], shrink=0.6)
        
    plt.suptitle(f"Simulación {i:04d} | dx={dx*1e3}mm | f={350}kHz", fontsize=14)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 5) MAIN - EJECUCIÓN DEL DATASET
# ==============================================================================


def main():
    OUT_DIR = "dataset_TUS_dx10_f350_R60"
    os.makedirs(OUT_DIR, exist_ok=True)


    # Configuración Física Escalada
    N_SIMS = 1000
    SEED = 123
    rng = np.random.default_rng(SEED)


    dx = 1.0e-3
    Nx, Ny, Nz = 128, 128, 128
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])


    F0 = 400e3 
    N_CYCLES = 10
    R_CONST = 60e-3
    APERTURE = 50e-3
    r_T = 50e-3
    BOWL_THICKNESS = 1.0 * dx
    
    GAP_MIN, GAP_MAX = 1.0e-3, 4.0e-3
    TMIN, TMAX = 3.0e-3, 6e-3 
    A_MIN, A_MAX = 0.3e6, 1.2e6
    CFL = 0.1
    PML_SIZE = 10
    
    # Medios (Agua, Hueso, Cerebro)
    props = {
        "water": (1482.0, 994.0, 0.0126),
        "skull": (2800.0, 1850.0, 15.0),
        "brain": (1546.0, 1046.0, 0.5)
    }


    # Tiempo
    domain_diag = np.sqrt(3) * Nx * dx
    t_end = 2.0 * domain_diag / props["water"][0]
    dt, Nt, _ = get_dt_Nt_or_fallback(kgrid, dx, props["skull"][0], CFL, t_end)
    
    try:
        kgrid.dt = dt
        kgrid.Nt = Nt
    except Exception:
        pass


    sim_opts = SimulationOptions(save_to_disk=True, pml_inside=True, pml_size=PML_SIZE, data_cast="single")
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=True)


    print(f"📡 Iniciando Dataset: {N_SIMS} sims | f={F0/1e3}kHz | dx={dx*1e3}mm")


    for i in range(N_SIMS):
        # 1. Geometría Ovoidal
        a, b, c = sample_skull_axes_for_fixed_rT_gap_ovoid(rng, r_T, GAP_MIN, GAP_MAX)
        m_skull, m_brain, t_local, t_mean = create_skull_masks_procedural_TAC(kgrid, a, b, c, TMIN, TMAX, rng)
        gap_real = r_T - max(a, b, c)


        # 2. Transductor y Foco
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        s_pos = (r_T * u).astype(np.float32)
        f_pos = (s_pos - R_CONST * u).astype(np.float32)
        
        s_mask_bool = create_oriented_bowl(kgrid, f_pos, s_pos, R_CONST, APERTURE, BOWL_THICKNESS)
        s_mask_u8 = (s_mask_bool & (~m_skull)).astype(np.uint8)
        n_pts = int(np.sum(s_mask_u8))


        if n_pts == 0: 
            print(f"⚠️ Sim {i} saltada: El transductor quedó dentro del cráneo/PML.")
            continue


        # 3. Fuente y Medio
        A_source = rng.uniform(A_MIN, A_MAX)
        source = kSource()
        source.p = np.tile(make_tone_burst(dt, Nt, F0, N_CYCLES, A_source)[None, :], (n_pts, 1)).astype(np.float32)
        source = make_source_compatible_for_time_signal(source, s_mask_u8)


        medium = kWaveMedium(
            sound_speed = np.where(m_skull, props["skull"][0], np.where(m_brain, props["brain"][0], props["water"][0])).astype(np.float32),
            density = np.where(m_skull, props["skull"][1], np.where(m_brain, props["brain"][1], props["water"][1])).astype(np.float32),
            alpha_coeff = np.where(m_skull, props["skull"][2], np.where(m_brain, props["brain"][2], props["water"][2])).astype(np.float32),
            alpha_power = 1.1
        )


        sensor = kSensor()
        sensor.mask = np.ones((Nx, Ny, Nz), dtype=np.uint8)
        sensor.record = ["p_max"]


        # 4. Ejecutar
        print(f"\n>> Sim {i:04d}/{N_SIMS-1} | P_pts: {n_pts} | Grosor Medio: {t_mean*1e3:.1f}mm")
        data = kspaceFirstOrder3D(kgrid=kgrid, medium=medium, source=source, sensor=sensor, 
                                  simulation_options=sim_opts, execution_options=exec_opts)


        # 5. Guardar (CON TODOS LOS METADATOS RESTAURADOS)
        p_max = safe_reshape_pmax(data["p_max"], Nx, Ny, Nz).astype(np.float32)
        p_max_norm = (p_max / A_source).astype(np.float32)
        
        out_path = os.path.join(OUT_DIR, f"sample_{i:04d}.npz")
        np.savez_compressed(
            out_path,
            # Arrays
            mask_skull=m_skull.astype(np.uint8),
            p_max_norm=p_max_norm,
            source_mask=s_mask_u8,
            
            # Geometría y Transductor
            source_pos=s_pos,
            focus_pos=f_pos,
            aperture=np.float32(APERTURE),
            radius_curvature=np.float32(R_CONST),
            r_T=np.float32(r_T),
            gap=np.float32(gap_real),
            
            # Cráneo
            a_out=np.float32(a), b_out=np.float32(b), c_out=np.float32(c),
            skull_thickness_mean=np.float32(t_mean),
            
            # Parámetros físicos
            A_source=np.float32(A_source),
            f0_hz=np.float32(F0),
            n_cycles=np.int32(N_CYCLES),
            dx=np.float32(dx),
            dt=np.float32(dt),
            Nt=np.int32(Nt)
        )


        # 6. Preview de las primeras 3
        if i < 3: 
            preview_results(m_skull, s_mask_u8, p_max_norm, i, dx, Nx, Ny, Nz)


if __name__ == "__main__":
    main()
