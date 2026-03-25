import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN
# =========================================================
# Cambia esta ruta a la carpeta donde generaste los datos
DATASET_DIR = r"dataset_TUS_dx10_f400_R60"  

# True: Muestra escala 0 a 1 (ideal para comparar foco relativo)
# False: Muestra la presión en Pascales reales (p_max_norm * A_source)
USE_NORMALIZED = True  
# =========================================================

def pick_random_npz(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"No encontré archivos .npz en {folder}")
    return os.path.join(folder, np.random.choice(files))

def show_3views(volume, title_prefix="", cmap="gray", vmin=None, vmax=None, mark_xyz=None):
    """
    volume shape (Nx,Ny,Nz)
    mark_xyz: (ix,iy,iz) para marcar el máximo
    """
    Nx, Ny, Nz = volume.shape
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sagital: x=cx => (y,z)
    im0 = axes[0].imshow(volume[cx, :, :].T, origin="lower",
                         cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix} Sagital (X={cx})")
    axes[0].set_xlabel("Eje Y (vóxeles)")
    axes[0].set_ylabel("Eje Z (vóxeles)")

    # Coronal: y=cy => (x,z)
    im1 = axes[1].imshow(volume[:, cy, :].T, origin="lower",
                         cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title_prefix} Coronal (Y={cy})")
    axes[1].set_xlabel("Eje X (vóxeles)")
    axes[1].set_ylabel("Eje Z (vóxeles)")

    # Axial: z=cz => (x,y)
    im2 = axes[2].imshow(volume[:, :, cz].T, origin="lower",
                         cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"{title_prefix} Axial (Z={cz})")
    axes[2].set_xlabel("Eje X (vóxeles)")
    axes[2].set_ylabel("Eje Y (vóxeles)")

    # Marcar punto (máximo) en las 3 vistas
    if mark_xyz is not None:
        ix, iy, iz = mark_xyz
        axes[0].plot(iy, iz, marker="+", markersize=14, markeredgewidth=2, color="white")
        axes[1].plot(ix, iz, marker="+", markersize=14, markeredgewidth=2, color="white")
        axes[2].plot(ix, iy, marker="+", markersize=14, markeredgewidth=2, color="white")

    plt.tight_layout()
    return fig, axes, (im0, im1, im2)

def main():
    npz_path = pick_random_npz(DATASET_DIR)
    d = np.load(npz_path)

    # 1. Extraer booleanos y metadatos
    is_water = bool(d["is_water_only"])
    skull = d["mask_skull"].astype(bool)
    src_mask = d["source_mask"].astype(bool)
    
    A_source = float(d["A_source"])
    dx = float(d["dx"])
    f0 = float(d["f0_hz"])
    gap = float(d["gap"])
    t_mean = float(d["skull_thickness_mean"])

    # 2. Reconstruir la salida según la preferencia
    p_norm = d["p_max_norm"].astype(np.float32)
    
    if USE_NORMALIZED:
        p = p_norm
        key = "p_max_norm"
        visual_vmax = max(1.0, float(np.max(p))) # A veces el foco difractado supera 1.0 ligeramente
    else:
        p = p_norm * A_source
        key = "Presión Máxima (Pa)"
        visual_vmax = float(np.max(p))

    # 3. Calcular el vóxel del máximo global
    idx_flat = int(np.argmax(p))
    ix, iy, iz = np.unravel_index(idx_flat, p.shape)

    # 4. Imprimir la información en consola
    tipo_caso = "💧 SOLO AGUA (Baseline)" if is_water else "💀 CASO CON CRÁNEO"
    
    print(f"\n📂 Archivo: {os.path.basename(npz_path)}")
    print(f"=====================================================")
    print(f" TIPO DE SIMULACIÓN : {tipo_caso}")
    print(f"=====================================================")
    print(f" Parámetros Físicos:")
    print(f"  • Frecuencia     : {f0/1e3:.1f} kHz")
    print(f"  • Presión Fuente : {A_source/1e6:.2f} MPa")
    if not is_water:
        print(f"  • Gap (Agua)     : {gap*1e3:.1f} mm")
        print(f"  • Grosor Cráneo  : {t_mean*1e3:.2f} mm (Media)")
    
    x_mm = (ix - p.shape[0]//2) * dx * 1e3
    y_mm = (iy - p.shape[1]//2) * dx * 1e3
    z_mm = (iz - p.shape[2]//2) * dx * 1e3
    
    print(f"\n Resultados del Foco Acústico:")
    print(f"  • Valor Máximo   : {p[ix,iy,iz]:.3e} {key.replace('p_max_norm', 'Adimensional')}")
    print(f"  • Vóxel del Pico : (X:{ix}, Y:{iy}, Z:{iz})")
    print(f"  • Posición Real  : ({x_mm:.1f} mm, {y_mm:.1f} mm, {z_mm:.1f} mm) desde el centro")
    print(f"=====================================================\n")

    # ----------------------------
    # A) Mostrar Geometría
    # ----------------------------
    geom = np.zeros_like(p, dtype=np.float32)
    geom[skull] = 0.5  # Gris para el cráneo
    geom[src_mask] = 1.0 # Blanco brillante para el transductor

    # Añadimos un pequeño margen para que el título no pise el gráfico
    show_3views(geom, title_prefix="Geometría", cmap="magma", vmin=0, vmax=1)
    plt.suptitle(f"{tipo_caso} - Vista Geométrica", fontsize=14, y=1.05)
    plt.show()

    # ----------------------------
    # B) Mostrar Mapa de Presión
    # ----------------------------
    fig, axes, ims = show_3views(
        p,
        title_prefix=key,
        cmap="jet",
        vmin=0.0,
        vmax=visual_vmax,
        mark_xyz=(ix, iy, iz)
    )
    plt.suptitle(f"{tipo_caso} - Mapa de Presión (Jet)", fontsize=14, y=1.05)
    cbar = fig.colorbar(ims[2], ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label(key)
    plt.show()

if __name__ == "__main__":
    main()
