import numpy as np
import torch
from torch.utils.data import Dataset


class UltrasoundDataset(Dataset):
    def __init__(self, files, transform=None):
        """
        files: Lista de rutas completas a los archivos .npz
        """
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Cargar el archivo .npz usando la ruta de la lista
        filepath = self.files[idx]
        data = np.load(filepath)

        # --- A. PREPARAR LOS 4 CANALES FÍSICOS (INPUTS) ---

        # Canal 0: Densidad (Reflexión)
        # Normalizamos: (Valor - 1000) / 1000.
        rho = (data['density'].astype(np.float32) - 1000.0) / 1000.0

        # Canal 1: Velocidad del Sonido (Refracción)
        # Normalizamos: (Valor - 1500) / 1500.
        c = (data['sound_speed'].astype(np.float32) - 1500.0) / 1500.0

        # Canal 2: Coeficiente de Absorción (Atenuación)
        # Normalizamos dividiendo por un máximo razonable (ej. 15).
        alpha = data['alpha_coeff'].astype(np.float32) / 15.0

        # Canal 3: Fuente (Geometría)
        # Es binaria (0 o 1).
        source = data['source_mask'].astype(np.float32)

        # --- JUNTAR LOS 4 CANALES ---
        # Creamos un bloque de [4, 128, 128, 128]
        input_combined = np.stack([rho, c, alpha, source], axis=0)
        input_tensor = torch.from_numpy(input_combined).float()

        # --- B. EL TARGET (PRESIÓN FINAL) ---
        target = data['p_max'].astype(np.float32) / 1e6
        target_tensor = torch.from_numpy(
            target).float().unsqueeze(0)  # [1, D, H, W]

        # --- C. MÁSCARA CLÍNICA (SOLO PARA LA LOSS) ---
        brain_mask = data['mask_brain'].astype(np.float32)
        brain_tensor = torch.from_numpy(brain_mask).float().unsqueeze(0)

        # Devolvemos 3 cosas: Input(4ch), Target(1ch), BrainMask(1ch)
        return input_tensor, target_tensor, brain_tensor
