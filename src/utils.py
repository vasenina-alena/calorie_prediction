from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import (
    DishDataset,
    collate_fn,
    get_transforms,
    build_ingredient_vocab,
)
from src.model import CalorieRegressor
from src.config import Config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        for k in batch:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad()
        preds = model(batch)
        loss = mae(preds, batch["target"])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        for k in batch:
            batch[k] = batch[k].to(device)

        preds = model(batch)
        loss = mae(preds, batch["target"])

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(cfg: Config) -> Dict[str, float]:
    
    set_seed(cfg.seed)

    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    dish_df = pd.read_csv(cfg.dish_csv)

    train_df = dish_df[dish_df["split"] == "train"].reset_index(drop=True)

    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg.valid_size,
        random_state=cfg.seed,
    )

    
    raw2idx, _ = build_ingredient_vocab(cfg.ingredients_csv)
    num_ingredients = len(raw2idx)

    train_tfms = get_transforms(cfg.img_size, is_train=True)
    val_tfms = get_transforms(cfg.img_size, is_train=False)

    train_ds = DishDataset(
        train_df,
        images_dir=cfg.images_dir,
        raw2idx=raw2idx,
        transform=train_tfms,
        return_target=True,
    )
    val_ds = DishDataset(
        val_df,
        images_dir=cfg.images_dir,
        raw2idx=raw2idx,
        transform=val_tfms,
        return_target=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    model = CalorieRegressor(
        num_ingredients=num_ingredients,
        ingr_emb_dim=cfg.ingr_emb_dim,
        backbone=cfg.backbone,
        mlp_hidden=cfg.mlp_hidden,
        dropout=cfg.dropout,
    ).to(Config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val = float("inf")

    # Списки для хранения значений потерь
    train_losses = []
    val_losses = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, Config.device)
        val_loss = validate(model, val_loader, Config.device)

        # Сохраняем потери для графиков
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train MAE: {train_loss:.2f} | valid MAE: {val_loss:.2f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_mae": best_val,
                    "config": cfg,
                },
                cfg.save_path,
            )

    # Визуализация графиков потерь
    plt.figure(figsize=(10, 5))
    # plt.plot(range(1, cfg.epochs + 1), train_losses, label='Train MAE', marker='s')
    plt.plot(range(1, cfg.epochs + 1), val_losses, label='Valid MAE', marker='o')
    
    plt.title('Valid MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid()
    plt.show()  # Отображение графика

    return {
        "best_val_mae": best_val,
        "save_path": str(cfg.save_path),
    }