import numpy as np

path = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/test/sample_0005.npz"

with np.load(path) as d:
    print(d.files)