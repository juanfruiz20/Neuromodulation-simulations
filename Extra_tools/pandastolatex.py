import pandas as pd
import glob
import os


# 1. Define aquí manualmente la lista de tus archivos (rutas completas o relativas)
mis_archivos = [
   r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_baseline_basic/test_metric_dice/best_summary.csv",
   r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_expDexpB/test_metrics_brain/epoch_030_summary.csv",
   r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_base01_globalL1/test_metrics_brain/best_summary.csv",
   r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_exp02_D_every2/test_metrics_brain/best_summary.csv",
   r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_exp03_lrdhalf_300epoch/test_metrics_brain/best_summary.csv",
   r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_TFG/test_metric_dice/epoch_100_summary.csv",
   r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_TFG/test_metric_dice/epoch_220_summary.csv",
]

# 2. Leer y combinar directamente
# Usamos una lista por comprensión para leer cada archivo de la lista
df_final = pd.concat([pd.read_csv(f) for f in mis_archivos], ignore_index=True)

# 3. Guardar el resultado
nombre_salida = 'master_TFG.csv'
df_final.to_csv(nombre_salida, index=False)

print(f"¡Listo! Se han unido {len(mis_archivos)} archivos en '{nombre_salida}'.")