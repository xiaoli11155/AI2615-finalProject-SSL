import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parse_args():
    parser = argparse.ArgumentParser(description="Plot finetune validation curves across datasets and shots.")
    parser.add_argument("--finetune-dir", default="outputs/finetune")
    parser.add_argument("--tasks", nargs="+", default=["jigsaw", "rotation", "relative_patch"])
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100", "tiny-imagenet"])
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 10, 100])
    parser.add_argument("--metric", default="val_top1")
    parser.add_argument("--output-dir", default="outputs/plots")
    return parser.parse_args()


def load_history(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_series(history, metric: str):
    xs = [row["epoch"] for row in history]
    ys = [row[metric] for row in history]
    return xs, ys


def set_integer_epoch_ticks(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def find_single_run(root: Path, pattern: str):
    matches = sorted(path for path in root.glob(pattern) if (path / "history.json").exists())
    if not matches:
        return None
    return matches[0]


def load_run_series(dataset_root: Path, run_pattern: str, metric: str):
    run_dir = find_single_run(dataset_root, run_pattern)
    if run_dir is None:
        return None, None
    history = load_history(run_dir / "history.json")
    return metric_series(history, metric), run_dir.name


def plot_task(task: str, datasets: list[str], shots: list[int], finetune_root: Path, metric: str, output_dir: Path):
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    colors = {
        0: "#1f77b4",
        10: "#ff7f0e",
        100: "#2ca02c",
    }

    for ax, dataset in zip(axes, datasets):
        dataset_root = finetune_root / dataset
        for shot in shots:
            color = colors.get(shot, None)

            pre_series, _ = load_run_series(
                dataset_root,
                f"{task}_*shots{shot}_*",
                metric,
            )
            if pre_series is not None:
                x_pre, y_pre = pre_series
                ax.plot(x_pre, y_pre, color=color, linewidth=2, label=f"{task} shot={shot}")

            random_series, _ = load_run_series(
                dataset_root,
                f"random_shots{shot}_*",
                metric,
            )
            if random_series is not None:
                x_rand, y_rand = random_series
                ax.plot(
                    x_rand,
                    y_rand,
                    color=color,
                    linewidth=2,
                    linestyle="--",
                    label=f"random shot={shot}",
                )

        ax.set_title(dataset)
        ax.set_xlabel("Epoch")
        set_integer_epoch_ticks(ax)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Finetune Curves: {task}", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"finetune_{task}_{metric}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dataset(dataset: str, tasks: list[str], shots: list[int], finetune_root: Path, metric: str, output_dir: Path):
    shots = [shot for shot in shots if shot != 10]
    fig, axes = plt.subplots(1, len(shots), figsize=(6 * len(shots), 5), sharey=True)
    if len(shots) == 1:
        axes = [axes]

    colors = {
        "jigsaw": "#1f77b4",
        "rotation": "#ff7f0e",
        "relative_patch": "#2ca02c",
        "random": "#7f7f7f",
    }

    dataset_root = finetune_root / dataset
    for ax, shot in zip(axes, shots):
        for task in tasks:
            series, _ = load_run_series(
                dataset_root,
                f"{task}_*shots{shot}_*",
                metric,
            )
            if series is None:
                continue
            x, y = series
            ax.plot(x, y, color=colors.get(task), linewidth=2, label=task)

        random_series, _ = load_run_series(
            dataset_root,
            f"random_shots{shot}_*",
            metric,
        )
        if random_series is not None:
            x_rand, y_rand = random_series
            ax.plot(
                x_rand,
                y_rand,
                color=colors["random"],
                linewidth=2,
                linestyle="--",
                label="random",
            )

        ax.set_title(f"{dataset}, shot={shot}")
        ax.set_xlabel("Epoch")
        set_integer_epoch_ticks(ax)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Finetune Curves by Pretext: {dataset}", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"finetune_{dataset}_{metric}_by_pretext.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    finetune_root = Path(args.finetune_dir)
    output_dir = Path(args.output_dir)

    generated = []
    for task in args.tasks:
        out_path = plot_task(
            task=task,
            datasets=args.datasets,
            shots=args.shots,
            finetune_root=finetune_root,
            metric=args.metric,
            output_dir=output_dir,
        )
        generated.append(out_path)

    for dataset in args.datasets:
        out_path = plot_dataset(
            dataset=dataset,
            tasks=args.tasks,
            shots=args.shots,
            finetune_root=finetune_root,
            metric=args.metric,
            output_dir=output_dir,
        )
        generated.append(out_path)

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
