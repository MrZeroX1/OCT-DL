from __future__ import annotations
from pathlib import Path
from typing import Sequence
import pandas as pd
import torch
from torch.utils.data import Dataset
from .preprocessing import build_transform, load_oct_image

class OCTBiomarkerDataset(Dataset):
    """CSV-backed dataset whose NaN labels remain explicitly unobserved."""
    def __init__(self, csv_path: str | Path, biomarkers: Sequence[str], image_size: int, training: bool = False):
        self.frame = pd.read_csv(csv_path)
        self.biomarkers = list(biomarkers)
        missing = {"image_path", *self.biomarkers} - set(self.frame.columns)
        if missing: raise ValueError(f"Dataset CSV misses columns: {sorted(missing)}")
        self.transform = build_transform(image_size, training)
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = self.transform(load_oct_image(row.image_path))
        values = pd.to_numeric(row[self.biomarkers], errors="coerce").to_numpy(dtype="float32")
        mask = ~torch.isnan(torch.tensor(values))
        target = torch.nan_to_num(torch.tensor(values), nan=0.0)
        return {"image": image, "target": target, "mask": mask, "patient_id": str(row.patient_id), "image_id": str(row.image_id)}
