import os
from pathlib import Path

import torch


def save_training_plot(history, progress_history, plot_path):
    if not progress_history and not history:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    progress = [item["progress"] for item in progress_history]
    progress_train_loss = [item["train_loss"] for item in progress_history]
    progress_train_acc = [item["train_acc"] for item in progress_history]
    epoch_progress = [item["progress"] for item in history]
    valid_loss = [item["valid_loss"] for item in history]
    valid_acc = [item["valid_acc"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)
    axes[0].plot(progress, progress_train_loss, label="train loss")
    axes[0].plot(epoch_progress, valid_loss, marker="o", linestyle="none", label="valid loss")
    axes[0].axhline(-torch.log(torch.tensor(1 / 3)).item(), color="gray", linestyle="--", linewidth=1, label="random baseline")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("training progress (%)")
    axes[0].set_ylabel("cross entropy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(progress, progress_train_acc, label="train acc")
    axes[1].plot(epoch_progress, valid_acc, marker="o", linestyle="none", label="valid acc")
    axes[1].axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="random baseline")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("training progress (%)")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
