import re
import numpy as np
import pandas as pd
import os
import h5py

FLOUR_ORDER = ["T", "E", "C", "A", "Z"]  # ADATTA questa lista al tuo caso reale


def parse_mix_code(mix_code: str, flour_order):
    """
    mix_code: es '30A70Z', '50A50A', '20E80C'
    flour_order: es ['T','E','C','A','Z']
    ritorna: vettore z normalizzato in [0,1]
    """

    # Trova coppie (numero,lettera)
    parts = re.findall(r'(\d+)([A-Za-z])', mix_code)
    # Esempio: '50A50A' → [('50','A'), ('50','A')]

    # Somma ripetizioni
    values = {}
    for num_str, letter in parts:
        v = int(num_str)
        if letter in values:
            values[letter] += v  # somma se ripetuta
        else:
            values[letter] = v

    # Costruisci vettore z
    z = np.zeros(len(flour_order), dtype=np.float32)
    for i, letter in enumerate(flour_order):
        if letter in values:
            z[i] = values[letter]

    # Normalizza
    total = z.sum()
    if total > 0:
        z = z / total

    return z


excel_path = "/home/acp/datasets/MuestrasEtiquetas1-200v1.0.xlsx"  # path al tuo file

df = pd.read_excel(excel_path)

# Controllo i nomi delle colonne (devono essere esattamente come hai scritto)
# 'N muestra' e 'Codigo'
id_to_z = {}

for _, row in df.iterrows():
    n_muestra = row["Nº MUESTRA"]  # es. 1, 2, 3, ...
    codigo = str(row["CÓDIGO"])  # es. 'B30A70ZF001'

    # Estrai la parte miscela: nel tuo caso è '30A70Z'
    # Se la struttura è sempre B + miscela + Fxxx,
    # prendi la parte tra la prima cifra e l'ultima lettera prima di 'F'
    # Se la cosa è più semplice (es. codice finisce sempre con 'F###'),
    # puoi fare:
    #   mix_code = codigo[ codigo.find("0") : codigo.rfind("F") ]
    # ma qui andiamo più robusti con una regex:
    mix_part_match = re.search(r'(\d+[A-Za-z](\d+[A-Za-z])*)', codigo)
    if not mix_part_match:
        raise ValueError(f"Non riesco a estrarre la miscela da '{codigo}'")
    mix_code = mix_part_match.group(1)  # es. '30A70Z'

    z = parse_mix_code(mix_code, FLOUR_ORDER)
    id_to_z[n_muestra] = z


input_dir = "/home/acp/datasets/SSD1/abundance"      # cartella dove hai le 224x224x121
output_bags_dir = "/home/acp/datasets/SSD1/abundance_train/bags"
output_labels_dir = "/home/acp/datasets/SSD1/abundance_train/labels"

os.makedirs(output_bags_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

PATCH_H = 16
PATCH_W = 16


def extract_patches_16x16(img):
    """
    img: np.array [L, H, W] con L=121, H=W=224
    ritorna: lista di patch [L,16,16]
    """
    L, H, W = img.shape
    patches = []
    coords = []

    for y in range(0, H, PATCH_H):
        for x in range(0, W, PATCH_W):
            patch = img[:, y:y + PATCH_H, x:x + PATCH_W]  # <--- CORRETTO PER TE
            if patch.shape == (L, PATCH_H, PATCH_W):
                patches.append(patch)
                coords.append((y, x))

    return patches, coords


for n_muestra, z in id_to_z.items():
    # Costruisci nome file immagine grande
    # Adatta questo alla tua convenzione:
    # es. "patch_000001.h5", "patch_000002.h5", ...
    fname = f"patch_{int(n_muestra):06d}.h5"
    fpath = os.path.join(input_dir, fname)

    if not os.path.exists(fpath):
        print(f"File non trovato per N muestra={n_muestra}: {fpath}")
        continue

    # Leggi immagine grande [224,224,121]
    with h5py.File(fpath, "r") as f:
        # Adatta 'data' al nome reale del dataset dentro l'h5
        img = f["cube"][...]  # np.array [121,224,224]

    patches, coords = extract_patches_16x16(img)

    for k, patch in enumerate(patches):
        y, x = coords[k]

        # Nome patch: includo id + indice patch o coordinate
        patch_name = f"patch_{int(n_muestra):06d}_y{y:03d}_x{x:03d}.h5"
        patch_path = os.path.join(output_bags_dir, patch_name)

        # Salva patch
        with h5py.File(patch_path, "w") as pf:
            pf.create_dataset("data", data=patch, compression="gzip")

        # Salva label z con stesso nome in labels/
        label_path = os.path.join(output_labels_dir, patch_name.replace(".h5", ".npy"))
        np.save(label_path, z)
