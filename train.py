"""Training script for the M4 HAR model."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import tensorflow as tf

from . import data
from .model import build_m4_model
from .utils import add_common_args, compute_metrics, resolve_device, set_seed


class MacroF1Callback(tf.keras.callbacks.Callback):
    """Compute macro-F1 on the validation set at epoch end."""

    def __init__(self, val_data: tuple[np.ndarray, np.ndarray]):
        super().__init__()
        self.val_x, self.val_y = val_data

    def on_epoch_end(self, epoch: int, logs: Dict | None = None) -> None:
        logs = logs or {}
        val_probs = self.model.predict(self.val_x, verbose=0)
        val_pred = np.argmax(val_probs, axis=1)
        val_true = np.argmax(self.val_y, axis=1)
        metrics = compute_metrics(val_true, val_pred)
        logs["val_macro_f1"] = metrics.macro_f1
        logs["val_accuracy"] = metrics.accuracy


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train the M4 HAR model.")
    parser = add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""

    args = parse_args()
    resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.random_seed, os.path.join(args.output_dir, "seed.json"))

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
        use_smote=args.use_smote,
    )

    split_path = os.path.join(args.output_dir, "split_indices.json")
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        print("Split summary:", json.dumps(summary, indent=2))

    model = build_m4_model(input_shape=prepared["train_x"].shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        MacroF1Callback((prepared["val_x"], prepared["val_y"])),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output_dir, "best_model.keras"),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(args.output_dir, "train_log.csv"), append=False
        ),
    ]

    model.fit(
        prepared["train_x"],
        prepared["train_y"],
        validation_data=(prepared["val_x"], prepared["val_y"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(os.path.join(args.output_dir, "final_model.keras"))


if __name__ == "__main__":
    main()
