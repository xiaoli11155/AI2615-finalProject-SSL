import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from ssl_project.data import build_pretext_dataset
from ssl_project.engine import evaluate, train_one_epoch
from ssl_project.models import PretextModel
from ssl_project.utils.checkpoint import save_checkpoint
from ssl_project.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised pretext training on Tiny ImageNet.")
    parser.add_argument("--data-dir", default="data/imagenet")
    parser.add_argument("--task", choices=["rotation", "jigsaw", "relative_patch"], default="rotation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--jigsaw-permutations", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", default="outputs/pretext")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, num_classes = build_pretext_dataset(
        args.data_dir,
        args.task,
        image_size=args.image_size,
        jigsaw_permutations=args.jigsaw_permutations,
        seed=args.seed,
    )
    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PretextModel(task=args.task, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = Path(args.output_dir) / f"{args.task}_bs{args.batch_size}_lr{args.lr}_ep{args.epochs}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.amp)
        val_metrics = evaluate(model, val_loader, criterion, device, topk=(1,))
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, row, args)
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, row, args)

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoints and history to {run_dir}")


if __name__ == "__main__":
    main()
