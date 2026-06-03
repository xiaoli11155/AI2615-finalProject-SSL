import torch
from torch import nn
from torchvision.models import resnet18


def build_resnet18_backbone(tiny_imagenet: bool = True) -> tuple[nn.Module, int]:
    """Return a ResNet-18 encoder with the classification layer removed."""
    model = resnet18(weights=None)
    if tiny_imagenet:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    feat_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feat_dim


class PretextModel(nn.Module):
    """ResNet encoder plus task-specific heads for SSL pretext tasks."""

    def __init__(self, task: str, num_classes: int, tiny_imagenet: bool = True):
        super().__init__()
        self.task = task
        self.encoder, feat_dim = build_resnet18_backbone(tiny_imagenet=tiny_imagenet)
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

    def __init__(self, num_classes: int, tiny_imagenet: bool = True):
        super().__init__()
        self.encoder, feat_dim = build_resnet18_backbone(tiny_imagenet=tiny_imagenet)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))
