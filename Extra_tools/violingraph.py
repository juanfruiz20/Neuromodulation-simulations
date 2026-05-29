import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

CSV_PATH = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/test_per_sample_3metrics_dice90.csv"

OUT_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/figures"
OUT_NAME = "overall_test_metrics_violin_paperlike"

SAVE_PNG = True
SAVE_PDF = True
DPI = 600

FIGSIZE = (10.8, 4.2)

# Etiqueta global del eje x
MODEL_LABEL = "cGAN Full Loss (220e)"

# Límites opcionales
YLIM_PEAK_ERR = (0, 15)
YLIM_DICE90 = (0, 100)
YLIM_LOC_ERR = None   # por ejemplo: (0, 25)

# Tipografía / estilo general
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


# =========================================================
# LOAD AND PREPARE DATA
# =========================================================

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No encontré ninguna de estas columnas: {candidates}")


def prepare_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    peak_col = find_col(df, [
        "peak_rel_err_brain_percent",
        "peak_rel_err_percent",
        "peak_error_percent",
        "peak_rel_err_brain",
    ])

    dice_col = find_col(df, [
        "dice_focus_brain_thr90_percent",
        "dice90_percent",
        "dice_focus_brain_thr90",
        "dice90",
    ])

    loc_col = find_col(df, [
        "peak_loc_err_mm_brain",
        "peak_loc_error_mm_brain",
        "peak_loc_err_mm",
        "peak_location_error_mm",
    ])

    plot_df = df[[peak_col, dice_col, loc_col]].copy()

    plot_df = plot_df.rename(columns={
        peak_col: "Peak relative error [%]",
        dice_col: "DSC A90 [%]",
        loc_col: "Peak location error [mm]",
    })

    # Si alguna métrica viene en escala 0-1, convertir a %
    if plot_df["Peak relative error [%]"].dropna().max() <= 1.5:
        plot_df["Peak relative error [%]"] *= 100.0

    if plot_df["DSC A90 [%]"].dropna().max() <= 1.5:
        plot_df["DSC A90 [%]"] *= 100.0

    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

    return plot_df


# =========================================================
# SUMMARY
# =========================================================

def print_summary(df):
    metrics = [
        "Peak relative error [%]",
        "DSC A90 [%]",
        "Peak location error [mm]",
    ]

    print("\n================ OVERALL SUMMARY ================")

    rows = []

    for metric in metrics:
        vals = df[metric].dropna().values

        row = {
            "metric": metric,
            "n": len(vals),
            "mean": np.mean(vals),
            "median": np.median(vals),
            "std": np.std(vals),
            "min": np.min(vals),
            "p25": np.percentile(vals, 25),
            "p75": np.percentile(vals, 75),
            "max": np.max(vals),
        }

        rows.append(row)

        print(f"\n{metric}")
        print(f"  n:      {row['n']}")
        print(f"  Mean:   {row['mean']:.4f}")
        print(f"  Median: {row['median']:.4f}")
        print(f"  Std:    {row['std']:.4f}")
        print(f"  Min:    {row['min']:.4f}")
        print(f"  P25:    {row['p25']:.4f}")
        print(f"  P75:    {row['p75']:.4f}")
        print(f"  Max:    {row['max']:.4f}")

    print("=================================================\n")

    return pd.DataFrame(rows)


def save_summary_csv(summary_df):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{OUT_NAME}_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"Summary CSV guardado en: {out_path}")


# =========================================================
# PLOT HELPERS
# =========================================================

def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)

    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", width=0.8, length=3)

    ax.grid(False)


def plot_single_violin_box(
    ax,
    values,
    ylabel,
    panel_letter,
    violin_color,
    ylim=None,
):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        raise ValueError(f"No hay valores válidos para {ylabel}")

    data = [values]
    pos = [1]

    # -----------------------------------------------------
    # Violin
    # -----------------------------------------------------
    violin = ax.violinplot(
        data,
        positions=pos,
        widths=0.62,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in violin["bodies"]:
        body.set_facecolor(violin_color)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.45)

    # -----------------------------------------------------
    # Boxplot interno
    # -----------------------------------------------------
    ax.boxplot(
        data,
        positions=pos,
        widths=0.24,
        patch_artist=True,
        whis=1.5,
        showmeans=True,

        boxprops=dict(
            facecolor="white",
            edgecolor="black",
            linewidth=0.85,
        ),

        whiskerprops=dict(
            color="black",
            linewidth=0.8,
        ),

        capprops=dict(
            color="black",
            linewidth=0.8,
        ),

        medianprops=dict(
            color="red",
            linewidth=1.3,
        ),

        meanprops=dict(
            marker="x",
            markeredgecolor="green",
            markerfacecolor="green",
            markersize=6.5,
            markeredgewidth=1.4,
        ),

        flierprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=3.0,
            linestyle="none",
            markeredgewidth=0.6,
        ),
    )

    # -----------------------------------------------------
    # Format
    # -----------------------------------------------------
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_xlim(0.45, 1.55)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)

    style_axis(ax)

    # Panel letter
    ax.text(
        -0.16,
        1.02,
        panel_letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


# =========================================================
# MAKE FIGURE
# =========================================================

def make_figure(df):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=FIGSIZE,
    )

    plot_single_violin_box(
        ax=axes[0],
        values=df["Peak relative error [%]"],
        ylabel=r"$\Delta P_R$ [%]",
        panel_letter="a",
        violin_color="#9ecae1",
        ylim=YLIM_PEAK_ERR,
    )

    plot_single_violin_box(
        ax=axes[1],
        values=df["DSC A90 [%]"],
        ylabel=r"$DSC_{A90}$ [%]",
        panel_letter="b",
        violin_color="#fcbba1",
        ylim=YLIM_DICE90,
    )

    plot_single_violin_box(
        ax=axes[2],
        values=df["Peak location error [mm]"],
        ylabel=r"$\Delta F$ [mm]",
        panel_letter="c",
        violin_color="#a1d99b",
        ylim=YLIM_LOC_ERR,
    )

    # Label global eje X
    fig.supxlabel(MODEL_LABEL, y=0.02, fontsize=11)

    plt.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.90,
        bottom=0.18,
        wspace=0.42,
    )

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
    df = prepare_dataframe(CSV_PATH)

    print(f"CSV cargado: {CSV_PATH}")
    print(f"Samples válidos: {len(df)}")

    summary_df = print_summary(df)
    save_summary_csv(summary_df)
    make_figure(df)


if __name__ == "__main__":
    main()