from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F


TINY_MEAN = (0.4802, 0.4481, 0.3975)
TINY_STD = (0.2302, 0.2265, 0.2262)


def _pil_loader(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def find_tiny_root(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    candidates = [root, root / "tiny-imagenet-200", root / "imagenet", root / "tiny-imagenet"]
    for candidate in candidates:
        if (candidate / "train").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find Tiny ImageNet train directory under {root}. "
        "Expected train/ with class subdirectories."
    )


class TinyImageNetVal(Dataset):
    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        ann = self.root / "val" / "val_annotations.txt"
        img_dir = self.root / "val" / "images"
        if not ann.exists():
            raise FileNotFoundError(f"Missing {ann}")
        train_classes = sorted(p.name for p in (self.root / "train").iterdir() if p.is_dir())
        self.class_to_idx = {name: idx for idx, name in enumerate(train_classes)}
        self.samples = []
        for line in ann.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1] in self.class_to_idx:
                self.samples.append((img_dir / parts[0], self.class_to_idx[parts[1]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = _pil_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, target


def train_transform(image_size: int = 64):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ]
    )


def eval_transform(image_size: int = 64):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ]
    )


def build_classification_datasets(data_dir, image_size=64):
    root = find_tiny_root(data_dir)
    train = datasets.ImageFolder(root / "train", transform=train_transform(image_size))
    val = TinyImageNetVal(root, transform=eval_transform(image_size))
    return train, val, len(train.classes)


def few_shot_subset(dataset, shots_per_class: int, seed: int):
    if shots_per_class <= 0:
        return dataset
    rng = random.Random(seed)
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [target for _, target in dataset.samples]
    by_class = {}
    for idx, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(idx)
    selected = []
    for indices in by_class.values():
        rng.shuffle(indices)
        selected.extend(indices[:shots_per_class])
    rng.shuffle(selected)
    return Subset(dataset, selected)


class RotationDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]
        label = random.randint(0, 3)
        return torch.rot90(image, k=label, dims=(1, 2)), label


class JigsawDataset(Dataset):
    def __init__(self, base_dataset, grid_size=3, num_permutations=30, seed=0):
        self.base_dataset = base_dataset
        self.grid_size = grid_size
        rng = random.Random(seed)
        identity = tuple(range(grid_size * grid_size))
        perms = {identity}
        while len(perms) < num_permutations:
            p = list(identity)
            rng.shuffle(p)
            perms.add(tuple(p))
        self.permutations = list(perms)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]
        label = random.randrange(len(self.permutations))
        return self._permute(image, self.permutations[label]), label

    def _permute(self, image, permutation):
        c, h, w = image.shape
        tile_h = h // self.grid_size
        tile_w = w // self.grid_size
        image = image[:, : tile_h * self.grid_size, : tile_w * self.grid_size]
        tiles = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                tiles.append(
                    image[
                        :,
                        row * tile_h : (row + 1) * tile_h,
                        col * tile_w : (col + 1) * tile_w,
                    ]
                )
        shuffled = [tiles[i] for i in permutation]
        rows = [
            torch.cat(shuffled[row * self.grid_size : (row + 1) * self.grid_size], dim=2)
            for row in range(self.grid_size)
        ]
        return torch.cat(rows, dim=1)


class RelativePatchDataset(Dataset):
    OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self, image_folder, patch_size=32, image_size=96):
        self.samples = image_folder.samples
        self.patch_size = patch_size
        self.image_size = image_size
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(TINY_MEAN, TINY_STD),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = _pil_loader(path)
        image = F.resize(image, [self.image_size, self.image_size])
        label = random.randrange(8)
        dy, dx = self.OFFSETS[label]
        center = self.image_size // 2 - self.patch_size // 2
        step = self.patch_size
        jitter = self.patch_size // 4
        ay = center + random.randint(-jitter, jitter)
        ax = center + random.randint(-jitter, jitter)
        by = max(0, min(self.image_size - self.patch_size, ay + dy * step))
        bx = max(0, min(self.image_size - self.patch_size, ax + dx * step))
        anchor = F.crop(image, ay, ax, self.patch_size, self.patch_size)
        neighbor = F.crop(image, by, bx, self.patch_size, self.patch_size)
        return (self.to_tensor(anchor), self.to_tensor(neighbor)), label


def build_pretext_dataset(data_dir, task, image_size=64, jigsaw_permutations=30, seed=0):
    root = find_tiny_root(data_dir)
    base = datasets.ImageFolder(root / "train", transform=train_transform(image_size))
    if task == "rotation":
        return RotationDataset(base), 4
    if task == "jigsaw":
        return JigsawDataset(base, num_permutations=jigsaw_permutations, seed=seed), jigsaw_permutations
    if task == "relative_patch":
        raw = datasets.ImageFolder(root / "train")
        return RelativePatchDataset(raw), 8
    raise ValueError(f"Unknown pretext task: {task}")
