import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random  # <--- Necesario para mezclar la lista manualmente

# IMPORTANTE: Usamos los archivos NUEVOS
from dataset_phys import UltrasoundDataset
from model_phys import UNet3D

# --- CONFIGURACIÓN ---
# <--- REVISA QUE ESTE NOMBRE SEA CORRECTO
DATA_DIR = "dataset123_oval3D_150try"
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
EPOCHS = 15
SAVE_EVERY = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints_phys"

os.makedirs(SAVE_DIR, exist_ok=True)

# --- FUNCIÓN DE PÉRDIDA CLÍNICA ---


def clinical_loss(prediction, target, brain_mask):
    """
    Loss v12: 'The Smart Limiter' (Física Realista)

    Filosofía:
    1. FOCO: Si falta energía -> Empuja fuerte (x500).
    2. HUESO (Clave): NO castiga por pintar en el hueso.
       Castiga SOLO si la IA pinta MÁS energía de la que dice el Ground Truth.
       (Evita la "explosión" en la entrada del cráneo, pero respeta la física).
    3. FONDO: Limpieza suave para definir bordes.
    """

    # Diferencia base al cuadrado (MSE)
    diff = prediction - target
    mse = diff ** 2

    # Pesos base
    weights = torch.ones_like(prediction)

    # --- MÁSCARAS ---
    skull_zone = 1.0 - brain_mask               # Zona Hueso + Aire
    has_real_signal = (target > 0.05).float()   # Donde hay sonido real

    # --- 1. RECUPERAR EL HAZ (Undershooting) ---
    # Si hay señal real y la IA se queda corta.
    # Empujamos para que pinte el foco y el paso por el hueso.
    is_undershooting = (prediction < target).float()

    # x500 es suficiente para subir sin explotar (probado que x3000 es mucho)
    signal_boost = has_real_signal * is_undershooting * 500.0
    weights = weights + signal_boost

    # --- 2. EL LIMITADOR ÓSEO (La corrección clave) ---
    # Aquí está la magia:
    # NO castigamos "skull_zone" a secas.
    # Castigamos "skull_zone * is_overshooting".
    # Traduccción: "Si el hueso brilla en el Target, está bien.
    # Pero si tú (IA) lo haces brillar MÁS que el Target, te castigo".

    is_overshooting = (prediction > target).float()

    # Castigo x1000 para frenar la "explosión" en la entrada
    skull_limit = skull_zone * is_overshooting * 1000.0
    weights = weights + skull_limit

    # --- 3. LIMPIEZA DE FONDO (Anti-Blur) ---
    # En el cerebro sano (donde no hay haz), si te pasas, castigo suave.
    # Ayuda a que el foco sea nítido y no una nube.
    background_clean = (1.0 - has_real_signal) * \
        brain_mask * is_overshooting * 200.0
    weights = weights + background_clean

    # Cálculo final
    loss = torch.mean(mse * weights)

    return loss
# --- BUCLE PRINCIPAL ---


def train():
    print(f"🚀 Iniciando entrenamiento FÍSICO (4 Canales) en {DEVICE}...")

    # 1. Cargar Datos y Verificar
    if not os.path.exists(DATA_DIR):
        print(f"❌ ERROR CRÍTICO: No encuentro la carpeta '{DATA_DIR}'")
        # Imprimir dónde está buscando Python para que lo veas
        print(f"   (Buscando en: {os.path.abspath(DATA_DIR)})")
        return

    # Obtener lista de archivos .npz
    all_files = [os.path.join(DATA_DIR, f)
                 for f in os.listdir(DATA_DIR) if f.endswith('.npz')]

    if len(all_files) == 0:
        print("❌ ERROR: La carpeta existe pero NO tiene archivos .npz")
        return

    # --- CAMBIO AQUÍ: División Manual (Más seguro que random_split) ---
    random.shuffle(all_files)  # Mezclamos la lista

    split_idx = int(0.8 * len(all_files))  # Calculamos el corte del 80%
    train_files = all_files[:split_idx]   # Primera parte para entrenar
    val_files = all_files[split_idx:]     # Resto para validar

    print(
        f"📂 Datos cargados: {len(train_files)} entrenamiento | {len(val_files)} validación")

    # Ahora sí, train_files es una LISTA pura de strings, no un Subset raro
    train_dataset = UltrasoundDataset(train_files)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 1. Creas la arquitectura (Vacía)
    model = UNet3D(in_channels=4, out_channels=1).to(DEVICE)

    # --- 2. INYECTAS LA SABIDURÍA (Cargar Epoch 30) ---
    checkpoint_path = "checkpoints_phys/phys_model_epoch_30.pth"

    if os.path.exists(checkpoint_path):
        print(f"🔄 CARGANDO CEREBRO PREVIO: {checkpoint_path}")
        # Cargamos los pesos entrenados
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("✅ Carga exitosa. Continuamos con el Fine-Tuning.")
    else:
        print(
            f"⚠️ ¡PELIGRO! No encontré {checkpoint_path}. Si sigues, empezarás de CERO.")
    # ----------------------------------------------------

    # 3. Creas el Optimizador (Nuevo, para adaptarse a la nueva Loss v9)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -----------------------------------------------------------------------
    # 3. Bucle de Entrenamiento (CORREGIDO)
    # -----------------------------------------------------------------------
    print(f"🚀 Arrancando Fine-Tuning desde Epoch 31 hasta 45...")

    # Range (31, 46) significa: 31, 32, ... hasta 45.
    for epoch in range(31, 46):
        model.train()
        epoch_loss = 0

        for batch_idx, (inputs, targets, brain_masks) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            brain_masks = brain_masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Llamada a tu Loss v9
            loss = clinical_loss(outputs, targets, brain_masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                # Quitamos el EPOCHS del print porque "15" ya no es el total real, confunde.
                print(
                    f"   [Epoch {epoch}] Batch {batch_idx}/{len(train_loader)} -> Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"📉 >>> FIN EPOCH {epoch} | Loss Promedio: {avg_loss:.4f} <<<")

        # Guardado: Usamos 'epoch' directamente, no 'epoch+1'
        if epoch % SAVE_EVERY == 0:
            save_path = f"{SAVE_DIR}/phys_model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 CHECKPOINT GUARDADO: {save_path}\n")

    print("🎉 ¡FINE-TUNING COMPLETADO! Ahora revisa los resultados. 🎉")


if __name__ == "__main__":
    train()
