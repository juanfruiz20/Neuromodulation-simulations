import os
import numpy as np
import shutil
from pathlib import Path


def split_dataset_stratified(
    source_dir="dataset_TUS_dx1_TAC_35mm_100_water_only",
    output_dir="dataset_split",
    train_ratio=0.8,
    val_ratio=0.1,
    seed=123
):
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # 1. Obtener todos los archivos .npz
    npz_files = list(source_path.glob("sample_*.npz"))
    if not npz_files:
        print(f"No se encontraron archivos .npz en {source_dir}")
        return

    print(
        f"[INFO] Analizando {len(npz_files)} archivos para separar por clase...")

    water_cases = []
    skull_cases = []

    # 2. Leer cada archivo para clasificarlo
    for f in npz_files:
        try:
            data = np.load(f)
            is_water = data['water_only'].item()

            if is_water == 1:
                water_cases.append(f)
            else:
                skull_cases.append(f)
        except Exception as e:
            print(f"[WARN] Error leyendo {f.name}: {e}")

    print(
        f"[INFO] Encontrados {len(water_cases)} casos 'solo agua' y {len(skull_cases)} casos 'con cráneo'.")

    # 3. Barajar aleatoriamente usando una semilla fija
    rng = np.random.default_rng(seed)
    rng.shuffle(water_cases)
    rng.shuffle(skull_cases)

    # 4. Función auxiliar para calcular los índices de corte
    def get_splits(items):
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_items = items[:n_train]
        val_items = items[n_train:n_train+n_val]
        test_items = items[n_train+n_val:]

        return train_items, val_items, test_items

    train_w, val_w, test_w = get_splits(water_cases)
    train_s, val_s, test_s = get_splits(skull_cases)

    # 5. Combinar listas
    splits = {
        "train": train_w + train_s,
        "val": val_w + val_s,
        "test": test_w + test_s
    }

    # 6. Copiar los archivos a las nuevas carpetas
    for split_name, files in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] Copiando {len(files)} archivos a {split_dir}...")
        for i, f in enumerate(files):
            shutil.copy2(f, split_dir / f.name)

            if (i + 1) % 100 == 0 or (i + 1) == len(files):
                print(f"  -> {i + 1}/{len(files)} copiados.")

    # 7. Copiar el archivo de metadatos a la raíz del nuevo dataset
    metadata_src = source_path / "dataset_metadata.json"
    if metadata_src.exists():
        print(f"\n[INFO] Copiando archivo de metadatos: {metadata_src.name}")
        shutil.copy2(metadata_src, output_path / metadata_src.name)
    else:
        print(
            f"\n[WARN] No se encontró {metadata_src.name} en la carpeta original.")

    print("\n[DONE] ¡Partición completada con éxito!")
    print(f"Distribución final de 'solo agua' / 'con cráneo':")
    print(f" - Train : {len(train_w)} / {len(train_s)}")
    print(f" - Val   : {len(val_w)} / {len(val_s)}")
    print(f" - Test  : {len(test_w)} / {len(test_s)}")


if __name__ == "__main__":
    split_dataset_stratified(
        source_dir="dataset_TUS_dx1_TAC_35mm_100_water_only",
        output_dir="dataset_TUS_split",
        train_ratio=0.8,
        val_ratio=0.1,
        seed=123
    )
