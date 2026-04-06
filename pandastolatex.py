import pandas as pd
import glob

# 1. Get a list of your 5 files
# You can use a wildcard like 'results_*.csv' or list them manually
files = [r'/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_V2_expB/test_metrics_brain_v2/epoch_070_summary.csv', r'/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_V2_expB/test_metrics_brain_v2/epoch_080_summary.csv', r'/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_V2_expB/test_metrics_brain_v2/epoch_090_summary.csv', r'/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_V2_expB/test_metrics_brain_v2/epoch_100_summary.csv', r'/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/checkpoints_unet_V2_expB/test_metrics_brain_v2/best_summary.csv']

# 2. Read and combine them
combined_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# 3. Save the master CSV
combined_df.to_csv('master_results.csv', index=False)