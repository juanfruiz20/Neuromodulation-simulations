import pandas as pd
import glob
import os

# 1. Define the directory path once
folder_path = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_ExpC_V2/test_metrics_brain_expC"

# 2. Get all summary files (excluding 'last' if you want)
# This finds every file ending in _summary.csv
all_files = glob.glob(os.path.join(folder_path, "*_summary.csv"))

# Filter out 'last' if it exists in the list
files = [f for f in all_files if 'last_summary' not in f]

# 3. Sort them (optional, but keeps things organized)
files.sort()

# 4. Read and combine
combined_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
combined_df.to_csv('master_results_expc2.csv', index=False)