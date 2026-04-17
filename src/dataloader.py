import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TusDataset(Dataset):
    """
    Dataset optimizado para simulación TUS (Transcranial Ultrasound Stimulation).
    Lee directamente los archivos .npz desde las carpetas (train/val/test).
    
    Entrada (X) : Tensor float32 [2, 128, 128, 128] -> (source_mask, mask_skull)
    Salida  (y) : Tensor float32 [1, 128, 128, 128] -> (p_max_norm)
    """

    def __init__(self, data_dir: str, expected_shape=(128, 128, 128)):
        super().__init__()
        self.data_dir = data_dir
        self.expected_shape = tuple(expected_shape)
        
        # Buscar todos los .npz en la carpeta proporcionada
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        if len(self.files) == 0:
            raise RuntimeError(f"❌ No se encontraron archivos .npz en: {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        
        # Usamos np.load en un bloque 'with' (buena práctica para cerrar el archivo rápido)
        with np.load(path) as d:
            # 1. Extraer máscaras (0s y 1s)
            # Convertimos a float32 porque los pesos de las redes neuronales operan en decimales
            src = d["source_mask"].astype(np.float32)
            skull = d["mask_skull"].astype(np.float32)
            
            # 2. Extraer el mapa de presión (Target)
            y = d["p_max_norm"].astype(np.float32)

        # 3. Verificación de seguridad (evita que un archivo corrupto rompa el entrenamiento horas después)
        if src.shape != self.expected_shape or skull.shape != self.expected_shape or y.shape != self.expected_shape:
            raise RuntimeError(
                f"Shape inválida en {os.path.basename(path)} | "
                f"src:{src.shape} skull:{skull.shape} y:{y.shape} | "
                f"esperado:{self.expected_shape}"
            )

        # 4. Apilar canales de entrada: [2, 128, 128, 128]
        # Canal 0: Máscara del Transductor | Canal 1: Máscara del Cráneo
        X = np.stack([src, skull], axis=0)

        # 5. Ajustar dimensión del target para PyTorch: [1, 128, 128, 128]
        y = y[np.newaxis, ...]

        return torch.from_numpy(X), torch.from_numpy(y)


# =========================================================
# BLOQUE DE PRUEBA (Sanity Check)
# =========================================================
if __name__ == "__main__":
    # Cambia esto por la ruta real a tu carpeta de Train
    TRAIN_DIR = r"/data/home/agustin/Documents/oslo/TFG Juanfe/Neuromodulation-simulations/dataset_TUS_SplitV1/train"

    try:
        # 1. Instanciar el Dataset
        ds = TusDataset(TRAIN_DIR)
        print(f"✅ Dataset cargado con éxito. Total de muestras: {len(ds)}")

        # 2. Extraer un elemento individual
        X, y = ds[0]
        print(f"\n--- Muestra Individual ---")
        print(f"Entrada X : {tuple(X.shape)} | Tipo: {X.dtype} | Min: {float(X.min())}, Max: {float(X.max())}")
        print(f"Salida y  : {tuple(y.shape)} | Tipo: {y.dtype} | Min: {float(y.min())}, Max: {float(y.max()):.4f}")
        
        # 3. Probar el DataLoader (Simulando un batch de entrenamiento)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 para probar rápido en Windows
        batch_X, batch_y = next(iter(loader))
        
        print(f"\n--- Batch de Entrenamiento ---")
        print(f"Batch X shape: {tuple(batch_X.shape)}") # Debería ser (4, 2, 128, 128, 128)
        print(f"Batch y shape: {tuple(batch_y.shape)}") # Debería ser (4, 1, 128, 128, 128)
        print("✅ DataLoader funcionando perfectamente.")
        
    except Exception as e:
        print(f"\n❌ Error al probar el DataLoader: {e}")