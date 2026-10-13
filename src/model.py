from __future__ import annotations
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights, efficientnet_b0, resnet18

def build_model(architecture: str, num_outputs: int, pretrained: bool = True, dropout: float = 0.3, freeze_backbone: bool = False) -> nn.Module:
    if num_outputs < 1: raise ValueError("num_outputs must be positive")
    if architecture == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_outputs))
        head = model.classifier
    elif architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_outputs))
        head = model.fc
    else: raise ValueError(f"Unsupported architecture: {architecture}")
    if freeze_backbone:
        for p in model.parameters(): p.requires_grad = False
        for p in head.parameters(): p.requires_grad = True
    return model

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__(); self.register_buffer("pos_weight", pos_weight)
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        observed = mask.to(raw.dtype)
        if observed.sum() == 0: raise ValueError("Batch has no observed biomarker labels")
        return (raw * observed).sum() / observed.sum()
