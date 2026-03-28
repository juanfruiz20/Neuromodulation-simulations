import pandas as pd

df = pd.read_csv("checkpoint_ranking.csv")

cols = [
    "checkpoint",
    "epoch",
    "mean_mse_global",
    "mean_mae_global",
    "mean_ssim_global",
    "mean_mse_brain",
    "mean_mae_brain",
    "mean_ssim_brain",
    "mean_peak_loc_err_mm_brain",
    "mean_peak_rel_err_brain",
    "mean_dice_focus_brain_thr50",
    "mean_dice_focus_brain_thr70",
]

df = df[cols]

df = df.rename(columns={
    "checkpoint": "Checkpoint",
    "epoch": "Epoch",
    "mean_mse_global": "MSE Global",
    "mean_mae_global": "MAE Global",
    "mean_ssim_global": "SSIM Global",
    "mean_mse_brain": "MSE Brain",
    "mean_mae_brain": "MAE Brain",
    "mean_ssim_brain": "SSIM Brain",
    "mean_peak_loc_err_mm_brain": "PeakLocErrBrain (mm)",
    "mean_peak_rel_err_brain": "PeakRelErrBrain",
    "mean_dice_focus_brain_thr50": "Mean Dice Brain Thr50",
    "mean_dice_focus_brain_thr70": "Mean Dice Brain Thr70",
})

latex_table = df.to_latex(
    index=False,
    float_format="%.4f",
    caption="Checkpoint comparison on the test set.",
    label="tab:checkpoint_ranking",
    column_format="lccccccccccc"
)

with open("checkpoint_ranking_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)

print("Tabla guardada en checkpoint_ranking_table.tex")