import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a compact SSL hyper-parameter sweep.")
    parser.add_argument("--data-dir", default="data/imagenet")
    parser.add_argument("--tasks", nargs="+", default=["rotation", "jigsaw", "relative_patch"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 3e-4])
    parser.add_argument("--pretext-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--shots-per-class", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-csv", default="outputs/experiment_summary.csv")
    return parser.parse_args()


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def maybe_amp(args):
    return ["--amp"] if args.amp else []


def best_metric(history_path, key):
    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    return max(row.get(key, 0.0) for row in history)


def main():
    args = parse_args()
    rows = []
    for task in args.tasks:
        for batch_size in args.batch_sizes:
            for lr in args.lrs:
                run(
                    [
                        sys.executable,
                        "train_pretext.py",
                        "--data-dir",
                        args.data_dir,
                        "--task",
                        task,
                        "--epochs",
                        str(args.pretext_epochs),
                        "--batch-size",
                        str(batch_size),
                        "--lr",
                        str(lr),
                        "--num-workers",
                        str(args.num_workers),
                        *maybe_amp(args),
                    ]
                )
                pretext_dir = Path("outputs/pretext") / f"{task}_bs{batch_size}_lr{lr}_ep{args.pretext_epochs}"
                ckpt = pretext_dir / "best.pt"
                run(
                    [
                        sys.executable,
                        "finetune.py",
                        "--data-dir",
                        args.data_dir,
                        "--pretrained",
                        str(ckpt),
                        "--epochs",
                        str(args.finetune_epochs),
                        "--batch-size",
                        str(batch_size),
                        "--lr",
                        str(lr),
                        "--shots-per-class",
                        str(args.shots_per_class),
                        "--num-workers",
                        str(args.num_workers),
                        *maybe_amp(args),
                    ]
                )
                ft_dir = Path("outputs/finetune") / f"{pretext_dir.name}_shots{args.shots_per_class}_bs{batch_size}_lr{lr}"
                rows.append(
                    {
                        "init": task,
                        "batch_size": batch_size,
                        "lr": lr,
                        "pretext_best_top1": best_metric(pretext_dir / "history.json", "val_top1"),
                        "finetune_best_top1": best_metric(ft_dir / "history.json", "val_top1"),
                        "finetune_best_top5": best_metric(ft_dir / "history.json", "val_top5"),
                    }
                )

    run(
        [
            sys.executable,
            "finetune.py",
            "--data-dir",
            args.data_dir,
            "--epochs",
            str(args.finetune_epochs),
            "--batch-size",
            str(args.batch_sizes[0]),
            "--lr",
            str(args.lrs[0]),
            "--shots-per-class",
            str(args.shots_per_class),
            "--num-workers",
            str(args.num_workers),
            *maybe_amp(args),
        ]
    )
    random_dir = Path("outputs/finetune") / f"random_shots{args.shots_per_class}_bs{args.batch_sizes[0]}_lr{args.lrs[0]}"
    rows.append(
        {
            "init": "random",
            "batch_size": args.batch_sizes[0],
            "lr": args.lrs[0],
            "pretext_best_top1": "",
            "finetune_best_top1": best_metric(random_dir / "history.json", "val_top1"),
            "finetune_best_top5": best_metric(random_dir / "history.json", "val_top5"),
        }
    )
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
