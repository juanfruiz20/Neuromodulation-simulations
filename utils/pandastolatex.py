import pandas as pd
import glob
import os

# 1. Define the directory path once
# folder_path = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_exp04_300epoch/test_metrics_brain"

# 2. Get all summary files (excluding 'last' if you want)
# This finds every file ending in _summary.csv
# all_files = glob.glob(os.path.join(folder_path, "*_summary.csv"))

# Filter out 'last' if it exists in the list
# files = [f for f in all_files if 'last_summary' not in f]

# 3. Sort them (optional, but keeps things organized)
# files.sort()

# 4. Read and combine
# combined_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
# combined_df.to_csv('master_results_cGANexp4_1_300.csv', index=False)
# import pandas as pd

# 1. Define aquí manualmente la lista de tus archivos (rutas completas o relativas)
mis_archivos = [
    r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_expDexpB/test_metrics_brain/epoch_030_summary.csv",
    r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_base01_globalL1/test_metrics_brain/best_summary.csv",
    r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_exp01_fromscratch_tb/test_metrics_brain/epoch_100_summary.csv",
    r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_cgan_exp04_300epoch/test_metric_dice/epoch_290_summary.csv"
]

# 2. Leer y combinar directamente
# Usamos una lista por comprensión para leer cada archivo de la lista
df_final = pd.concat([pd.read_csv(f) for f in mis_archivos], ignore_index=True)

# 3. Guardar el resultado
nombre_salida = 'master_results_nuevo.csv'
df_final.to_csv(nombre_salida, index=False)

print(f"¡Listo! Se han unido {len(mis_archivos)} archivos en '{nombre_salida}'.")