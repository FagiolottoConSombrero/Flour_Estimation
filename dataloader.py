import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HSILLPDataset(Dataset):
    """
    Dataset per LLP su patch HSI 16x16x121.
    Ogni elemento:
      - X: tensor [16,16,121] (bag = patch)
      - z: tensor [K]         (proporzioni farine)
    """
    def __init__(self, root_dir, bag_key="data", dtype=torch.float32):
        """
        root_dir: cartella root del dataset (contiene train/val/test)
        split:    'train', 'val' oppure 'test'
        bag_key:  nome del dataset dentro l'h5 (default: 'data')
        dtype:    tipo dei tensori per X
        """
        self.root_dir = Path(root_dir)
        self.bag_key = bag_key
        self.dtype = dtype

        self.bag_dir = self.root_dir / "bags"
        self.label_dir = self.root_dir / "labels"

        if not self.bag_dir.exists():
            raise FileNotFoundError(f"Cartella bags non trovata: {self.bag_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Cartella labels non trovata: {self.label_dir}")

        self.files = sorted([f for f in os.listdir(self.bag_dir) if f.endswith(".h5")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]  # es. 'patch_000001_y000_x000.h5'

        # ---- Leggi patch (bag) ----
        bag_path = self.bag_dir / fname
        with h5py.File(bag_path, "r") as f:
            X = f[self.bag_key][...]     # np.array [16,16,121]

        # ---- Leggi label z ----
        label_name = fname.replace(".h5", ".npy")
        label_path = self.label_dir / label_name
        if not label_path.exists():
            raise FileNotFoundError(f"Label mancante per {fname}: {label_path}")

        z = np.load(label_path)          # np.array [K]

        # ---- Converti in tensori ----
        X = torch.tensor(X, dtype=self.dtype)      # [16,16,121]
        z = torch.tensor(z, dtype=torch.float32)   # [K]

        return X, z
