"""
ColorInvariantSareeEncoder Neural Architecture.
Encodes saree textile images into a color-invariant metric embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any


class ColorInvariantSareeEncoder(nn.Module):
    """
    Color-Invariant Saree Design Encoder.
    
    Architecture:
      Input (B, 3, H, W)
          ↓
      Pretrained ResNet Backbone (e.g. ResNet-18)
          ↓
      Global Average Pooling (B, in_features)
          ↓
      MLP Projection Head: Linear -> BatchNorm -> ReLU -> Linear (B, embedding_dim)
          ↓
      L2 Normalization (Unit hypersphere embedding)
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        embedding_dim: int = 256,
        projection_hidden_dim: int = 512,
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.projection_hidden_dim = projection_hidden_dim
        
        # Load torchvision backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            in_features = base_model.fc.in_features
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            in_features = base_model.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Extract feature representation up to global average pool
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        # Projection head: maps representations to metric learning space
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, projection_hidden_dim),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing L2-normalized embeddings.
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            Normalized embeddings of shape (B, embedding_dim) where ||z||_2 = 1
        """
        # Backbone spatial feature extraction -> (B, in_features, 1, 1)
        feat = self.backbone(x)
        # Flatten -> (B, in_features)
        feat = torch.flatten(feat, 1)
        # Project -> (B, embedding_dim)
        proj = self.projection_head(feat)
        # L2 Normalize
        normalized_emb = F.normalize(proj, p=2, dim=1)
        return normalized_emb

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward embedding extraction in eval mode."""
        return self.forward(x)

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Returns parameter count and model architecture details."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone": self.backbone_name,
            "embedding_dim": self.embedding_dim,
            "projection_hidden_dim": self.projection_hidden_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "approx_size_mb": (total_params * 4) / (1024 * 1024)
        }
