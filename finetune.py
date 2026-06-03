import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ssl_project.data import build_classification_datasets, few_shot_subset
from ssl_project.engine import evaluate, train_one_epoch
from ssl_project.models import ClassificationModel
from ssl_project.utils.checkpoint import load_encoder_from_pretext, save_checkpoint
from ssl_project.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune on Tiny ImageNet classification.")
    parser.add_argument("--data-dir", default="data/imagenet")
    parser.add_argument("--pretrained", default=None, help="Path to SSL checkpoint. Omit for random init.")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--shots-per-class", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", default="outputs/finetune")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, num_classes = build_classification_datasets(args.data_dir, args.image_size)
    train_set = few_shot_subset(train_set, args.shots_per_class, args.seed)
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

    model = ClassificationModel(num_classes=num_classes).to(device)
    init_name = "random"
    if args.pretrained:
        load_encoder_from_pretext(model, args.pretrained)
        init_name = Path(args.pretrained).parent.name
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    run_dir = Path(args.output_dir) / f"{init_name}_shots{args.shots_per_class}_bs{args.batch_size}_lr{args.lr}"
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.amp)
        val_metrics = evaluate(model, val_loader, criterion, device, topk=(1, 5))
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
