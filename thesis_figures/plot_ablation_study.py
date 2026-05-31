import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# =========================================================
# CONFIG
# =========================================================

CSV_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/Ablation_table.csv"

OUT_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/figures"
OUT_NAME = "ablation_ssim_vs_focal_error_clean"

SAVE_PNG = True
SAVE_PDF = False
DPI = 600

FIGSIZE = (7.0, 4.8)

# Orden exacto de las filas del CSV
MODEL_LABELS_BY_ROW = [
    "U-Net Base",
    "U-Net Full Loss",
    "cGAN Base",
    "cGAN D 1:2",
    "cGAN D LR 0.5",
    "cGAN Full 100e",
    "cGAN Full 220e",
]

# Destacar modelos sí/no
USE_HIGHLIGHT = True
HIGHLIGHT_MODELS = [
    "U-Net Full Loss",
    "cGAN Full 220e",
]

# Barras de error opcionales
USE_ERRORBARS = False

# Sin título = más paper-like
SHOW_TITLE = False

# Límites manuales opcionales
XLIM = None
YLIM = None
# XLIM = (0.71, 0.82)
# YLIM = (1.78, 3.25)


# =========================================================
# STYLE
# =========================================================

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11


# =========================================================
# HELPERS
# =========================================================

def normalize_col_name(name):
    return (
        str(name)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )


def find_col(df, candidates):
    normalized = {normalize_col_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalize_col_name(cand)
        if key in normalized:
            return normalized[key]

    raise KeyError(
        "No encontré ninguna de estas columnas:\n"
        f"{candidates}\n\n"
        "Columnas disponibles en tu CSV:\n"
        f"{list(df.columns)}"
    )


def try_find_col(df, candidates):
    try:
        return find_col(df, candidates)
    except KeyError:
        return None


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Columna SSIM global
    ssim_col = find_col(df, [
        "mean_ssim_global",
        "mean SSIM global",
        "SSIM global",
        "ssim_global",
        "mean_ssim",
        "SSIM",
    ])

    # Columna focal localization error
    loc_col = find_col(df, [
        "mean_peak_loc_err_mm_brain",
        "mean peak loc error mm",
        "mean_peak_loc_error_mm_brain",
        "mean_peak_loc_err_mm",
        "peak_loc_err_mm_brain",
        "peak_loc_error_mm_brain",
        "mean_peak_location_error_mm",
        "focal_localization_error_mm",
    ])

    plot_df = df[[ssim_col, loc_col]].copy()
    plot_df = plot_df.rename(columns={
        ssim_col: "ssim_global",
        loc_col: "focal_loc_error_mm",
    })

    plot_df["ssim_global"] = pd.to_numeric(plot_df["ssim_global"], errors="coerce")
    plot_df["focal_loc_error_mm"] = pd.to_numeric(plot_df["focal_loc_error_mm"], errors="coerce")

    # Std opcionales
    std_ssim_col = try_find_col(df, [
        "std_ssim_global",
        "std SSIM global",
        "ssim_global_std",
        "std_ssim",
    ])

    std_loc_col = try_find_col(df, [
        "std_peak_loc_err_mm_brain",
        "std peak loc error mm",
        "std_peak_loc_error_mm_brain",
        "std_focal_localization_error_mm",
    ])

    if std_ssim_col is not None:
        plot_df["std_ssim_global"] = pd.to_numeric(df[std_ssim_col], errors="coerce")
    else:
        plot_df["std_ssim_global"] = np.nan

    if std_loc_col is not None:
        plot_df["std_focal_loc_error_mm"] = pd.to_numeric(df[std_loc_col], errors="coerce")
    else:
        plot_df["std_focal_loc_error_mm"] = np.nan

    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["ssim_global", "focal_loc_error_mm"]).reset_index(drop=True)

    if len(plot_df) != len(MODEL_LABELS_BY_ROW):
        raise ValueError(
            f"El CSV tiene {len(plot_df)} filas válidas, pero MODEL_LABELS_BY_ROW tiene "
            f"{len(MODEL_LABELS_BY_ROW)} labels. Deben coincidir exactamente."
        )

    plot_df["model"] = MODEL_LABELS_BY_ROW
    return plot_df


def get_point_style(model):
    if USE_HIGHLIGHT and model in HIGHLIGHT_MODELS:
        return {
            "s": 80,
            "marker": "o",
            "facecolor": "black",
            "edgecolor": "black",
            "linewidth": 0.9,
        }

    return {
        "s": 55,
        "marker": "o",
        "facecolor": "white",
        "edgecolor": "black",
        "linewidth": 1.1,
    }

def get_label_offsets():
    return {
        "U-Net Base":      (6, -10),
        "U-Net Full Loss": (-10, 10),
        "cGAN D 1:2":      (12, -16),
        "cGAN D LR 0.5":   (2, 16),
        "cGAN Base":       (6, 8),
        "cGAN Full 100e":  (-12, -10),
        "cGAN Full 220e":  (8, 8),
        "cGAN Full 300e":  (8, -10),
    }


def print_summary(df):
    print("\n================ DATA USED FOR PLOT ================")
    print(df[["model", "ssim_global", "focal_loc_error_mm"]].to_string(index=False))
    print("====================================================\n")


# =========================================================
# PLOT
# =========================================================

def make_plot(df):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    offsets = get_label_offsets()

    # -----------------------------------------------------
    # Draw points + labels
    # -----------------------------------------------------
    for _, row in df.iterrows():
        model = row["model"]
        x = row["ssim_global"]
        y = row["focal_loc_error_mm"]

        style = get_point_style(model)

        # Error bars opcionales
        if (
            USE_ERRORBARS
            and not np.isnan(row["std_ssim_global"])
            and not np.isnan(row["std_focal_loc_error_mm"])
        ):
            ax.errorbar(
                x,
                y,
                xerr=row["std_ssim_global"],
                yerr=row["std_focal_loc_error_mm"],
                fmt="none",
                ecolor="0.72",
                elinewidth=0.8,
                capsize=2,
                alpha=0.6,
                zorder=1,
            )

        # Punto
        ax.scatter(
            x,
            y,
            s=style["s"],
            marker=style["marker"],
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=style["linewidth"],
            zorder=3,
        )

        # Label sin caja blanca
        dx, dy = offsets.get(model, (6, 6))

        txt = ax.annotate(
            model,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color="0.22",
            ha="left" if dx >= 0 else "right",
            va="center",
            zorder=4,
            clip_on=False,
        )

        # Contorno blanco fino para legibilidad
        txt.set_path_effects([
            pe.withStroke(linewidth=2.8, foreground="white")
        ])

    # -----------------------------------------------------
    # Axes labels
    # -----------------------------------------------------
    ax.set_xlabel("Global SSIM")
    ax.set_ylabel(r"$\Delta F$ (mm)")

    if SHOW_TITLE:
        ax.set_title(
            "Model comparison: global similarity vs focal accuracy",
            fontsize=12,
            pad=10,
        )

    # -----------------------------------------------------
    # Limits
    # -----------------------------------------------------
    if XLIM is not None:
        ax.set_xlim(XLIM)
    else:
        x_min, x_max = df["ssim_global"].min(), df["ssim_global"].max()
        x_pad = 0.12 * (x_max - x_min + 1e-8)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

    if YLIM is not None:
        ax.set_ylim(YLIM)
    else:
        y_min, y_max = df["focal_loc_error_mm"].min(), df["focal_loc_error_mm"].max()
        y_pad = 0.10 * (y_max - y_min + 1e-8)
        ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)

    # -----------------------------------------------------
    # Paper-like style
    # -----------------------------------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.tick_params(axis="both", which="major", width=0.8, length=3)

    ax.grid(
        True,
        linestyle="-",
        linewidth=0.35,
        alpha=0.16,
    )

    fig.tight_layout()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    png_path = os.path.join(OUT_DIR, f"{OUT_NAME}.png")
    pdf_path = os.path.join(OUT_DIR, f"{OUT_NAME}.pdf")

    if SAVE_PNG:
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        print(f"PNG guardado en: {png_path}")

    if SAVE_PDF:
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"PDF guardado en: {pdf_path}")

    plt.show()


# =========================================================
# MAIN
# =========================================================

def main():
    df = load_data(CSV_PATH)
    print_summary(df)
    make_plot(df)


if __name__ == "__main__":
    main()