"""CNN classifiers with a global-average-pool + linear head.

With that head, Grad-CAM for the final conv block reduces exactly to CAM
(weights of the linear layer applied to the feature maps), so explanations can
be computed inside the exported ONNX graph without autograd at serving time.
"""

import torch
from torch import nn
from torchvision import models

ARCHS = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, 960),
}


class CamNet(nn.Module):
    def __init__(self, arch: str, n_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        ctor, weights, dim = ARCHS[arch]
        self.features = ctor(weights=weights if pretrained else None).features
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.features(x).mean(dim=(2, 3))))

    def forward_with_cam(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(x)
        logits = self.fc(f.mean(dim=(2, 3)))
        cam = torch.einsum("cd,bdhw->bchw", self.fc.weight, f)
        return logits, cam


class CamExport(nn.Module):
    def __init__(self, net: CamNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.forward_with_cam(x)


class FrozenEncoder(nn.Module):
    """ImageNet EfficientNet-B0 embedding: baseline features and OOD reference space."""

    def __init__(self):
        super().__init__()
        self.features = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).features

    def forward(self, x):
        return self.features(x).mean(dim=(2, 3))
