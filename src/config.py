from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_dir: Path = Path(".")
    data_dir: Path = project_dir / 'data'
    dish_csv: Path = data_dir / 'dish_new.csv'
    ingredients_csv: Path = data_dir / 'ingredients.csv'
    images_dir: Path = data_dir / 'images'

    save_dir: Path = project_dir / 'model'
    save_path: Path = save_dir / 'best_model.pt'

    seed: int = 42
    num_workers: int = 0 
    valid_size: float = 0.15

    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    img_size: int = 224
    backbone: str = "efficientnet_b0" 
    ingr_emb_dim: int = 128
    mlp_hidden: int = 512
    dropout: float = 0.1

    device: str = "cpu"
