import os
import shutil
import numpy as np

# =========================================================
# CONFIGURACIÓN
# =========================================================
SOURCE_DIR = "dataset_TUS_dx10_f400_R60"  # Tu carpeta original
OUTPUT_DIR = "dataset_TUS_SplitV1"          # Nueva carpeta donde se guardará la división

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 # El resto
SEED = 42 # Semilla para que la división sea reproducible
# =========================================================

def main():
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"❌ No se encontró la carpeta original: {SOURCE_DIR}")

    # 1. Crear las carpetas de destino
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    # 2. Escanear y clasificar los archivos rápidamente
    print(f"🔍 Escaneando archivos en '{SOURCE_DIR}'...")
    water_cases = []
    skull_cases = []

    archivos_npz = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".npz")]

    for file in archivos_npz:
        filepath = os.path.join(SOURCE_DIR, file)
        
        # Leemos SOLO la variable 'is_water_only', evitando cargar los mapas 3D a la RAM
        with np.load(filepath) as data:
            if data["is_water_only"]:
                water_cases.append(filepath)
            else:
                skull_cases.append(filepath)

    print(f"📊 Total encontrados: {len(skull_cases)} casos de Cráneo | {len(water_cases)} casos de Solo Agua.")

    # 3. Barajar aleatoriamente para evitar sesgos de orden de generación
    np.random.seed(SEED)
    np.random.shuffle(water_cases)
    np.random.shuffle(skull_cases)

    # 4. Función auxiliar para calcular índices y repartir
    def get_splits(file_list):
        total = len(file_list)
        n_train = int(total * TRAIN_RATIO)
        n_val = int(total * VAL_RATIO)
        
        return (
            file_list[:n_train], 
            file_list[n_train : n_train + n_val], 
            file_list[n_train + n_val:]
        )

    # Repartimos cada grupo por separado para garantizar homogeneidad
    water_train, water_val, water_test = get_splits(water_cases)
    skull_train, skull_val, skull_test = get_splits(skull_cases)

    # 5. Agrupamos los diccionarios de copiado
    copy_plan = {
        "train": water_train + skull_train,
        "val": water_val + skull_val,
        "test": water_test + skull_test
    }

    # 6. Copiar los archivos
    print("\n🚀 Iniciando el copiado estructurado a 'train', 'val' y 'test'...")
    
    for split_name, files in copy_plan.items():
        dest_folder = os.path.join(OUTPUT_DIR, split_name)
        
        # Opcional: Barajar una vez más dentro del split para que agua y cráneo queden mezclados
        np.random.shuffle(files)
        
        for i, f in enumerate(files):
            filename = os.path.basename(f)
            dest_path = os.path.join(dest_folder, filename)
            
            # shutil.copy2 copia el archivo y preserva los metadatos de fecha/hora
            shutil.copy2(f, dest_path)
            
        # Imprimir resumen de la carpeta
        n_water = sum(1 for f in files if f in water_cases)
        n_skull = sum(1 for f in files if f in skull_cases)
        print(f"   ✅ {split_name.upper():<6} -> Total: {len(files):<4} (Cráneo: {n_skull:<3} | Agua: {n_water:<3})")

    print(f"\n🎉 ¡División estratificada completada con éxito en '{OUTPUT_DIR}'!")

if __name__ == "__main__":
    main()
