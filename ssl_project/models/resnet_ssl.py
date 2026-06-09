import torch
from torch import nn
from torchvision.models import resnet18
from types import MethodType


class ResidualAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck_dim, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(bottleneck_dim, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


def _patch_block_with_adapter(block: nn.Module, bottleneck_dim: int):
    if hasattr(block, "adapter"):
        return

    channels = block.conv2.out_channels
    block.adapter = ResidualAdapter(channels, bottleneck_dim)
    original_forward = block.forward

    def forward_with_adapter(self, x):
        out = original_forward(x)
        return out + self.adapter(out)

    block.forward = MethodType(forward_with_adapter, block)


def attach_resnet_adapters(model: nn.Module, bottleneck_dim: int):
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, layer_name)
        for block in layer:
            _patch_block_with_adapter(block, bottleneck_dim)


def build_resnet18_backbone(small_images: bool = True) -> tuple[nn.Module, int]:
    """Return a ResNet-18 encoder with the classification layer removed."""
    model = resnet18(weights=None)
    if small_images:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    feat_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feat_dim


class PretextModel(nn.Module):
    """ResNet encoder plus task-specific heads for SSL pretext tasks."""

    def __init__(self, task: str, num_classes: int, small_images: bool = True):
        super().__init__()
        self.task = task
        self.encoder, feat_dim = build_resnet18_backbone(small_images=small_images)
        if task == "relative_patch":
            self.head = nn.Sequential(
                nn.Linear(feat_dim * 2, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(512, num_classes),
            )
        else:
            self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        if self.task == "relative_patch":
            anchor, neighbor = x
            z1 = self.encoder(anchor)
            z2 = self.encoder(neighbor)
            return self.head(torch.cat([z1, z2], dim=1))
        return self.head(self.encoder(x))


class ClassificationModel(nn.Module):
    """ResNet encoder plus a linear classifier for downstream evaluation."""

    def __init__(self, num_classes: int, small_images: bool = True, adapter_dim: int = 0):
        super().__init__()
        self.encoder, feat_dim = build_resnet18_backbone(small_images=small_images)
        self.adapter_dim = adapter_dim
        if adapter_dim > 0:
            attach_resnet_adapters(self.encoder, adapter_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def enable_adapter_tuning(self):
        if self.adapter_dim <= 0:
            raise ValueError("Adapter tuning requires adapter_dim > 0.")
        self.freeze_encoder()
        for module in self.encoder.modules():
            adapter = getattr(module, "adapter", None)
            if adapter is not None:
                for param in adapter.parameters():
                    param.requires_grad = True
