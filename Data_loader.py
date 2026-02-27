# dataset_loader.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------
# 1) Utils: listar archivos
# ---------------------------


def list_npz_files(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if len(files) == 0:
        raise RuntimeError(f"No encontré .npz en: {data_dir}")
    return files


# ---------------------------
# 2) Split reproducible
#    Train / Val / Test
# ---------------------------
def split_files(files, seed=123, train_ratio=0.85, val_ratio=0.10):
    """
    Devuelve (train_files, val_files, test_files).
    test_ratio = 1 - train_ratio - val_ratio (lo que sobra).
    """
    if train_ratio <= 0 or val_ratio < 0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError(
            "Ratios inválidos: requiere train>0, val>=0 y train+val<1.")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)

    n = len(files)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))

    # Asegura que no te quedes sin muestras por redondeos
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    test_files = [files[i] for i in test_idx]

    return train_files, val_files, test_files


# ---------------------------
# 3) Stats para normalización
# ---------------------------
def compute_stats(train_files, max_samples=128, seed=0):
    """
    Stats SOLO sobre TRAIN (evita leakage).
    Calcula mean/std por canal:
      - sound_speed (z-score)
      - density      (z-score)
      - alpha_log    (log + z-score)
    Nota: source_mask es binaria, NO se normaliza.
    """
    rng = np.random.default_rng(seed)
    if len(train_files) == 0:
        raise RuntimeError("train_files vacío: no puedo calcular stats.")

    if len(train_files) > max_samples:
        pick = list(rng.choice(train_files, size=max_samples, replace=False))
    else:
        pick = train_files

    # acumuladores de media de la media y media de cuadrados (aprox. global)
    sums = {"sound_speed": 0.0, "density": 0.0, "alpha_log": 0.0}
    sqs = {"sound_speed": 0.0, "density": 0.0, "alpha_log": 0.0}
    count = 0

    for p in pick:
        d = np.load(p)
        c = d["sound_speed"].astype(np.float32)
        rho = d["density"].astype(np.float32)
        a = d["alpha_coeff"].astype(np.float32)
        a = np.log(a + 1e-6).astype(np.float32)

        # medias por volumen
        for name, arr in [("sound_speed", c), ("density", rho), ("alpha_log", a)]:
            m = float(arr.mean())
            m2 = float((arr * arr).mean())
            sums[name] += m
            sqs[name] += m2

        count += 1

    stats = {}
    for name in sums:
        mean = sums[name] / count
        mean2 = sqs[name] / count
        var = max(mean2 - mean * mean, 1e-12)
        std = float(np.sqrt(var))
        stats[name] = {"mean": float(mean), "std": std}

    return stats


# ---------------------------
# 4) Dataset PyTorch
# ---------------------------
class TusDataset(Dataset):
    """
    Devuelve:
      X: float32 [4, 128,128,128]   (c, rho, log(alpha), source_mask)
      y: float32 [1, 128,128,128]   (p_max_norm)

    Nota: aquí NO aplicamos log al target (según tu decisión actual).
    """

    def __init__(self, files, stats=None, normalize=True, expected_shape=(128, 128, 128)):
        self.files = files
        self.stats = stats
        self.normalize = normalize
        self.expected_shape = tuple(expected_shape)

        if len(self.files) == 0:
            raise RuntimeError("TusDataset recibió lista vacía de archivos.")

        if self.normalize and self.stats is None:
            raise RuntimeError(
                "normalize=True pero stats=None. Calcula stats con compute_stats(train_files).")

    def __len__(self):
        return len(self.files)

    def _zscore(self, x, key):
        if not self.normalize:
            return x
        s = self.stats[key]
        return (x - s["mean"]) / (s["std"] + 1e-8)

    def __getitem__(self, i):
        path = self.files[i]
        d = np.load(path)

        c = d["sound_speed"].astype(np.float32)
        rho = d["density"].astype(np.float32)
        a = d["alpha_coeff"].astype(np.float32)
        src = d["source_mask"].astype(np.float32)
        y = d["p_max_norm"].astype(np.float32)

        # checks de shape (evita bugs silenciosos)
        if c.shape != self.expected_shape or rho.shape != self.expected_shape or src.shape != self.expected_shape or y.shape != self.expected_shape:
            raise RuntimeError(
                f"Shape inválida en {os.path.basename(path)} | "
                f"c:{c.shape} rho:{rho.shape} src:{src.shape} y:{y.shape} | "
                f"esperado:{self.expected_shape}"
            )

        # normalización inputs
        c = self._zscore(c, "sound_speed")
        rho = self._zscore(rho, "density")

        # alpha: log + z-score (estándar en tu caso)
        a = np.log(a + 1e-6).astype(np.float32)
        a = self._zscore(a, "alpha_log")

        # stack canales: [4,128,128,128]
        X = np.stack([c, rho, a, src], axis=0).astype(np.float32)

        # target: [1,128,128,128]
        y = y[None, ...].astype(np.float32)

        return torch.from_numpy(X), torch.from_numpy(y)


# ---------------------------
# 5) Sanity check rápido
# ---------------------------
if __name__ == "__main__":
    DATA_DIR = r"dataset_TUS_dx05_TAClike_ovoidXY_R30_thick2dx"

    files = list_npz_files(DATA_DIR)
    train_files, val_files, test_files = split_files(
        files, seed=123, train_ratio=0.85, val_ratio=0.10)

    stats = compute_stats(train_files, max_samples=128, seed=0)

    print("Total sims:", len(files))
    print("Split train/val/test:", len(train_files),
          len(val_files), len(test_files))
    print("Stats:", stats)

    ds = TusDataset(train_files, stats=stats, normalize=True)
    X, y = ds[0]
    print("X:", tuple(X.shape), X.dtype, "| y:", tuple(y.shape), y.dtype)
    print("X mean/std:", float(X.mean()), float(X.std()),
          "| y mean/std:", float(y.mean()), float(y.std()))
