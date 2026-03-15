from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import timm


class IngredientsEncoder(nn.Module):
   
    def __init__(self, num_ingredients: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_ingredients + 1,
            embedding_dim=emb_dim,
            padding_idx=0,
        )

    def forward(self, ingredients: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        emb = self.embedding(ingredients)  # [B, L, D]
        mask = mask.unsqueeze(-1)           # [B, L, 1]

        emb = emb * mask
        summed = emb.sum(dim=1)              # [B, D]
        denom = mask.sum(dim=1).clamp(min=1) # [B, 1]

        return summed / denom


class ImageEncoder(nn.Module):
    
    def __init__(self, backbone: str, pretrained: bool):
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,
        )
        self.out_features = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CalorieRegressor(nn.Module):
    
    def __init__(
        self,
        num_ingredients: int,
        ingr_emb_dim: int,
        backbone: str,
        mlp_hidden: int,
        dropout: float
    ):
        super().__init__()

        feat_dim = 0
        
        self.ingr_encoder = IngredientsEncoder(
            num_ingredients=num_ingredients,
            emb_dim=ingr_emb_dim,
        )
        feat_dim += ingr_emb_dim

        self.img_encoder = ImageEncoder(
            backbone=backbone,
            pretrained=True,
        )
        feat_dim += self.img_encoder.out_features

        feat_dim += 1

        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []

        img_feat = self.img_encoder(batch["image"])
        img_feat = img_feat * batch["has_image"]
        features.append(img_feat)

        text_feat = self.ingr_encoder(
            batch["ingredients"],
            batch["ingr_mask"],
        )
        features.append(text_feat)

        features.append(batch["mass"])

        x = torch.cat(features, dim=1)
        out = self.regressor(x)

        return out