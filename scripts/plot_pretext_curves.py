import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pretext validation curves across tasks and datasets.")
    parser.add_argument("--pretext-dir", default="outputs/pretext")
    parser.add_argument("--tasks", nargs="+", default=["jigsaw", "rotation", "relative_patch"])
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100", "tiny-imagenet"])
    parser.add_argument("--metric", default="val_top1")
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--report-style", action="store_true")
    return parser.parse_args()


def load_history(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_series(history, metric: str):
    xs = [row["epoch"] for row in history]
    ys = [row[metric] for row in history]
    return xs, ys


def set_epoch_ticks(ax):
    ax.xaxis.set_major_locator(MultipleLocator(5))


def apply_pretext_style(report_style: bool):
    if not report_style:
        return
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )


def load_run_series(pretext_root: Path, dataset: str, task: str, metric: str):
    run_dir = pretext_root / dataset / next((p.name for p in sorted((pretext_root / dataset).glob(f"{task}_*")) if (p / "history.json").exists()), "")
    if not run_dir.name:
        return None
    history = load_history(run_dir / "history.json")
    return metric_series(history, metric)


def plot_datasets(datasets: list[str], tasks: list[str], pretext_root: Path, metric: str, output_dir: Path):
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    colors = {
        "jigsaw": "#1f77b4",
        "rotation": "#ff7f0e",
        "relative_patch": "#2ca02c",
    }

    for ax, dataset in zip(axes, datasets):
        for task in tasks:
            series = load_run_series(pretext_root, dataset, task, metric)
            if series is None:
                continue
            x, y = series
            ax.plot(x, y, color=colors.get(task), linewidth=2, label=task)

        ax.set_title(dataset)
        ax.set_xlabel("Epoch")
        set_epoch_ticks(ax)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pretext_{metric}_by_task_combined.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    apply_pretext_style(args.report_style)
    pretext_root = Path(args.pretext_dir)
    output_dir = Path(args.output_dir)

    out_path = plot_datasets(args.datasets, args.tasks, pretext_root, args.metric, output_dir)
    print(out_path)


if __name__ == "__main__":
    main()
