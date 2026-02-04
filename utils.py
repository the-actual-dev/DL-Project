"""Utility helpers for reproducibility, metrics, and CLI behavior."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class MetricSummary:
    """Container for metric values."""

    accuracy: float
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    macro_f1: float
    confusion: List[List[int]]
    aucs: List[float] | None = None


def set_seed(seed: int, seed_path: str | None = None) -> None:
    """Set random seeds for reproducibility and optionally persist to disk."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if seed_path:
        with open(seed_path, "w", encoding="utf-8") as handle:
            json.dump({"seed": seed}, handle, indent=2)


def assert_no_leakage(train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Assert that train/test indices have empty intersection and print counts."""

    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    intersection = train_set.intersection(test_set)
    if intersection:
        raise ValueError(
            f"Data leakage detected: {len(intersection)} overlapping indices."
        )
    print(
        f"No leakage detected. Train samples: {len(train_idx)}, Test samples: {len(test_idx)}"
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricSummary:
    """Compute accuracy, per-class precision/recall/F1, macro-F1, confusion."""

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]
    confusion = confusion_matrix(y_true, y_pred)
    return MetricSummary(
        accuracy=accuracy,
        per_class_precision=precision.tolist(),
        per_class_recall=recall.tolist(),
        per_class_f1=f1.tolist(),
        macro_f1=macro_f1,
        confusion=confusion.tolist(),
    )


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> List[float]:
    """Compute one-vs-rest AUC per class when probabilities are available."""

    num_classes = y_prob.shape[1]
    y_true_onehot = np.eye(num_classes)[y_true]
    aucs = []
    for class_idx in range(num_classes):
        try:
            auc = roc_auc_score(y_true_onehot[:, class_idx], y_prob[:, class_idx])
        except ValueError:
            auc = 0.0
        aucs.append(auc)
    return aucs


def save_metrics(metrics: MetricSummary, path: str) -> None:
    """Save metrics summary to JSON."""

    payload = {
        "accuracy": metrics.accuracy,
        "per_class_precision": metrics.per_class_precision,
        "per_class_recall": metrics.per_class_recall,
        "per_class_f1": metrics.per_class_f1,
        "macro_f1": metrics.macro_f1,
        "confusion_matrix": metrics.confusion,
        "auc": metrics.aucs,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add shared CLI arguments."""

    parser.add_argument("--data-path", default="./data/dataset.csv")
    parser.add_argument("--labels-path", default="./data/label.csv")
    parser.add_argument("--subject-col", default=None)
    parser.add_argument("--window-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--scaler", default="standard", choices=["standard", "minmax"])
    return parser


def resolve_device(device: str) -> None:
    """Set TensorFlow device visibility."""

    if device.lower() == "cpu":
        tf.config.set_visible_devices([], "GPU")
    else:
        # Allow TensorFlow to see GPUs if available; no-op if none.
        pass
