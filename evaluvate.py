"""Evaluation script for the M4 HAR model."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import data
from .utils import add_common_args, compute_auc, compute_metrics, save_metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Evaluate the M4 HAR model.")
    parser = add_common_args(parser)
    parser.add_argument(
        "--model-path", default="./results/best_model.keras", help="Path to model"
    )
    return parser.parse_args()


def plot_confusion_matrix(confusion: np.ndarray, path: str) -> None:
    """Plot and save confusion matrix with annotations."""

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    for (i, j), value in np.ndenumerate(confusion):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """Main evaluation entry point."""

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.output_dir, "test_x.npy")):
        prepared = data.load_prepared_data(args.output_dir)
    else:
        prepared = data.prepare_data(
            data_path=args.data_path,
            labels_path=args.labels_path,
            subject_col=args.subject_col,
            window_len=args.window_len,
            stride=args.stride,
            test_size=args.test_size,
            random_seed=args.random_seed,
            scaler_type=args.scaler,
            output_dir=args.output_dir,
            use_smote=False,
        )

    model = tf.keras.models.load_model(args.model_path)
    probs = model.predict(prepared["test_x"], verbose=0)
    preds = np.argmax(probs, axis=1)
    true = np.argmax(prepared["test_y"], axis=1)

    metrics = compute_metrics(true, preds)
    metrics.aucs = compute_auc(true, probs)
    results_path = os.path.join(args.output_dir, "test_metrics.json")
    save_metrics(metrics, results_path)

    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    split_path = os.path.join(args.output_dir, "split_indices.json")
    test_indices = list(range(len(true)))
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as handle:
            split_info = json.load(handle)
        if split_info.get("split_type") == "window":
            test_indices = split_info.get("test_indices", test_indices)
    with open(predictions_path, "w", encoding="utf-8") as handle:
        header = "index,true_label,pred_label," + ",".join(
            [f"prob_class{i}" for i in range(probs.shape[1])]
        )
        handle.write(header + "\n")
        for idx, (t, p, prob) in zip(test_indices, zip(true, preds, probs)):
            row = [str(idx), str(t), str(p)] + [f"{v:.6f}" for v in prob]
            handle.write(",".join(row) + "\n")

    plot_confusion_matrix(
        np.array(metrics.confusion), os.path.join(args.output_dir, "confusion_matrix.png")
    )

    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"status": "completed"}, handle, indent=2)


if __name__ == "__main__":
    main()
