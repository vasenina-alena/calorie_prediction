from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def build_ingredient_vocab(ingredients_csv: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
   
    df = pd.read_csv(ingredients_csv)

    raw2idx = {}
    idx2raw = {}

    idx = 1  # 0 is PAD
    for raw_id in df["id"].astype(str).tolist():
        raw2idx[raw_id] = idx
        idx2raw[idx] = raw_id
        idx += 1

    return raw2idx, idx2raw


def parse_ingredients(ingredients_str: str) -> List[str]:
   
    if not isinstance(ingredients_str, str) or ingredients_str.strip() == "":
        return []
    return ingredients_str.split(";")


class DishDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        raw2idx: Dict[str, int],
        transform: transforms.Compose,
        return_target: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.raw2idx = raw2idx
        self.transform = transform
        self.return_target = return_target

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, dish_id: str) -> Tuple[torch.Tensor, float]:
        img_path = self.images_dir / str(dish_id) / "rgb.png"

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            has_image = 1.0
        else:
            # Fallback (на будущее, хотя в датасете сейчас все с картинками)
            img = torch.zeros(3, self.transform.size, self.transform.size)
            has_image = 0.0

        return img, has_image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Image
        image, has_image = self._load_image(row["dish_id"])

        # Ingredients
        raw_ingredients = parse_ingredients(row["ingredients"])
        ingr_ids = [
            self.raw2idx[raw_id]
            for raw_id in raw_ingredients
            if raw_id in self.raw2idx
        ]
        ingr_ids = torch.tensor(ingr_ids, dtype=torch.long)

        # Mass
        mass = torch.tensor([row["total_mass"]], dtype=torch.float32)

        sample = {
            "image": image,
            "has_image": torch.tensor([has_image], dtype=torch.float32),
            "ingredients": ingr_ids,
            "mass": mass,
        }

        if self.return_target:
            target = torch.tensor([row["total_calories"]], dtype=torch.float32)
            sample["target"] = target

        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    
    images = torch.stack([b["image"] for b in batch], dim=0)
    has_image = torch.stack([b["has_image"] for b in batch], dim=0)
    mass = torch.stack([b["mass"] for b in batch], dim=0)

    # Ingredients padding
    lengths = [b["ingredients"].numel() for b in batch]
    max_len = max(lengths) if lengths else 0

    if max_len > 0:
        ingredients = torch.zeros(len(batch), max_len, dtype=torch.long)
        ingr_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

        for i, b in enumerate(batch):
            l = b["ingredients"].numel()
            if l > 0:
                ingredients[i, :l] = b["ingredients"]
                ingr_mask[i, :l] = True
    else:
        ingredients = torch.zeros(len(batch), 1, dtype=torch.long)
        ingr_mask = torch.zeros(len(batch), 1, dtype=torch.bool)

    collated = {
        "image": images,
        "has_image": has_image,
        "ingredients": ingredients,
        "ingr_mask": ingr_mask,
        "mass": mass,
    }

    if "target" in batch[0]:
        targets = torch.stack([b["target"] for b in batch], dim=0)
        collated["target"] = targets

    return collated


def get_transforms(img_size: int, is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(img_size + 32),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )