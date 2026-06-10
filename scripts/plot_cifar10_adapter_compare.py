import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path("outputs/finetune/cifar10")
OUT = Path("outputs/plots_paper/cifar10_rotation_jigsaw_adapter_compare_val_top1.png")


def load_history(run_name: str):
    path = ROOT / run_name / "history.json"
    return json.loads(path.read_text(encoding="utf-8"))


def series(run_name: str, metric: str = "val_top1"):
    history = load_history(run_name)
    xs = [row["epoch"] for row in history]
    ys = [row[metric] for row in history]
    return xs, ys


def main():
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 14,
        }
    )

    runs = {
        "rotation": {
            "full finetune": "rotation_bs128_lr0.001_ep20_shots100_bs128_lr0.001",
            "freeze": "rotation_bs128_lr0.001_ep20_frozen_shots100_bs128_lr0.001",
            "adapter64": "rotation_bs128_lr0.001_ep20_adapter64_shots100_bs128_lr0.001",
            "adapter128": "rotation_bs128_lr0.001_ep20_adapter128_shots100_bs128_lr0.001",
            "random": "random_shots100_bs128_lr0.001",
        },
        "jigsaw": {
            "full finetune": "jigsaw_bs128_lr0.001_ep20_shots100_bs128_lr0.001",
            "freeze": "jigsaw_bs128_lr0.001_ep20_frozen_shots100_bs128_lr0.001",
            "adapter64": "jigsaw_bs128_lr0.001_ep20_adapter64_shots100_bs128_lr0.001",
            "adapter128": "jigsaw_bs128_lr0.001_ep20_adapter128_shots100_bs128_lr0.001",
            "random": "random_shots100_bs128_lr0.001",
        },
    }
    colors = {
        "full finetune": "#1f77b4",
        "freeze": "#7f7f7f",
        "adapter64": "#ff7f0e",
        "adapter128": "#2ca02c",
        "random": "#d62728",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)

    for ax, (task, task_runs) in zip(axes, runs.items()):
        for label, run_name in task_runs.items():
            x, y = series(run_name)
            linestyle = "--" if label in {"freeze", "random"} else "-"
            ax.plot(x, y, label=label, color=colors[label], linewidth=2, linestyle=linestyle)
        ax.set_title(task)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("val_top1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("CIFAR-10 Finetune: Full vs Freeze vs Adapter", y=0.965)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
