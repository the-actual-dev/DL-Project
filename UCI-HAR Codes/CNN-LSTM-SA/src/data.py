# src/data.py
"""Data loading and preprocessing for HAR dataset (extended for UCI HAR)."""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .utils import assert_no_leakage

DEFAULT_FEATURE_COLS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "linacc_x",
    "linacc_y",
    "linacc_z",
]
ALT_FEATURE_COLS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "lacc_x",
    "lacc_y",
    "lacc_z",
]
LABEL_FALLBACKS = ("label", "activity")


def _find_label_col(df: pd.DataFrame, label_col: str) -> str | None:
    if label_col in df.columns:
        return label_col
    for candidate in LABEL_FALLBACKS:
        if candidate in df.columns:
            return candidate
    return None


def _load_csv(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def _resolve_feature_cols(
    df: pd.DataFrame,
    feature_cols: list[str] | None,
    label_col: str,
    subject_col: str | None,
) -> list[str]:
    if feature_cols is not None:
        return feature_cols
    if all(col in df.columns for col in DEFAULT_FEATURE_COLS):
        return DEFAULT_FEATURE_COLS
    if all(col in df.columns for col in ALT_FEATURE_COLS):
        return ALT_FEATURE_COLS
    excluded = set(LABEL_FALLBACKS)
    excluded.add(label_col)
    if subject_col:
        excluded.add(subject_col)
    remaining = [col for col in df.columns if col not in excluded]
    numeric_cols = (
        df[remaining].select_dtypes(include=["number"]).columns.tolist()
        if remaining
        else []
    )
    if numeric_cols:
        return numeric_cols
    raise ValueError(
        "Unable to infer feature columns. "
        f"Available columns: {df.columns.tolist()}"
    )


def _interpolate_missing(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df[feature_cols] = df[feature_cols].interpolate(method="linear", limit_direction="both")
    return df


def _window_count(num_samples: int, window_len: int, stride: int) -> int:
    if num_samples < window_len:
        return 0
    return (num_samples - window_len) // stride + 1


def _segment_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_len: int,
    stride: int,
    label_strategy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if label_strategy != "end":
        raise ValueError("label_strategy must be 'end' to match the paper.")
    windows = []
    window_labels = []
    labels = np.asarray(labels).astype(int)
    for start in range(0, len(data) - window_len + 1, stride):
        end = start + window_len
        windows.append(data[start:end])
        window_labels.append(labels[end - 1])
    return np.array(windows), np.array(window_labels)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[labels]


def _save_numpy(output_dir: str, name: str, array: np.ndarray) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.npy")
    np.save(path, array)
    return path


def _save_split_indices(output_dir: str, payload: Dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "split_indices.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _fit_scaler(
    train_x: np.ndarray, scaler_type: str
) -> Tuple[StandardScaler | MinMaxScaler, np.ndarray]:
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    shape = train_x.shape
    train_flat = train_x.reshape(-1, shape[-1])
    train_scaled = scaler.fit_transform(train_flat).reshape(shape)
    return scaler, train_scaled


def _apply_scaler(
    scaler: StandardScaler | MinMaxScaler, data: np.ndarray
) -> np.ndarray:
    shape = data.shape
    return scaler.transform(data.reshape(-1, shape[-1])).reshape(shape)


# ---------------------------
# UCI HAR loader (Inertial signals)
# ---------------------------
def _read_uci_signal_file(path: str) -> np.ndarray:
    # Each row = one window, columns = 128 timesteps (space separated)
    return np.loadtxt(path)


def _load_uci_har_folder(har_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the Inertial Signals (raw) from the UCI HAR dataset folder.
    Returns (X_all, y_all, subjects_all) where:
      - X_all shape = (N, 128, 9)
      - y_all shape = (N,) zero-based labels (0..5)
      - subjects_all shape = (N,)
    """
    # helper to find train/test paths under given root
    train_root = os.path.join(har_root, "train")
    test_root = os.path.join(har_root, "test")
    insig_rel = os.path.join("Inertial Signals")

    # list of signals in the order we want (9 channels)
    signal_files = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    ]

    def _load_split(split_root: str):
        # load each signal file and stack last axis
        arrs = []
        for sig in signal_files:
            p = os.path.join(split_root, insig_rel, f"{sig}_"+os.path.basename(split_root)+".txt")
            if not os.path.exists(p):
                # some archived datasets may use consistent names without split suffix
                alt = os.path.join(split_root, insig_rel, f"{sig}.txt")
                if os.path.exists(alt):
                    p = alt
                else:
                    raise FileNotFoundError(f"Expected {p} or {alt}")
            a = _read_uci_signal_file(p)  # shape (N_windows, 128)
            arrs.append(a[..., None])  # (N, 128, 1)
        # concatenate channels -> (N, 128, 9)
        X = np.concatenate(arrs, axis=2)
        # load labels (y_*.txt) and subject ids
        y_path = os.path.join(split_root, f"y_{os.path.basename(split_root)}.txt")
        subj_path = os.path.join(split_root, f"subject_{os.path.basename(split_root)}.txt")
        y = np.loadtxt(y_path).astype(int).squeeze()  # 1..6
        subs = np.loadtxt(subj_path).astype(int).squeeze()
        return X, y, subs

    X_train, y_train, s_train = _load_split(train_root)
    X_test, y_test, s_test = _load_split(test_root)

    # Concatenate maintaining train/test order (we'll track indices)
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    s_all = np.concatenate([s_train, s_test], axis=0)
    return X_all, y_all, s_all


# ---------------------------
# Main prepare_data (extended)
# ---------------------------
def prepare_data(
    data_path: str,
    labels_path: str | None,
    subject_col: str | None,
    window_len: int,
    stride: int,
    test_size: float,
    random_seed: int,
    scaler_type: str,
    output_dir: str,
    num_classes: int = 6,
    feature_cols: list[str] | None = None,
    label_col: str = "activity",
    label_strategy: str = "end",
    use_smote: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load, preprocess, split and save arrays.

    Special handling: if ⁠ data_path ⁠ points to the root of the UCI HAR dataset
    (a directory containing 'train/X_train.txt' and 'test/X_test.txt' or
    'UCI HAR Dataset' folder), the loader will read the Inertial Signals and
    use the provided train/test split.
    """
    if labels_path:
        print("Note: --labels-path is ignored. Using labels from dataset if available.")

    # --- Detect UCI HAR folder ---
    uci_root = None
    # If user passed the top-level zip-extracted folder, normalize it
    if os.path.isdir(data_path):
        # common layout: <path>/UCI HAR Dataset  or <path> itself contains train/
        if os.path.basename(data_path).lower().startswith("uci har"):
            uci_root = data_path
        elif os.path.exists(os.path.join(data_path, "UCI HAR Dataset")):
            uci_root = os.path.join(data_path, "UCI HAR Dataset")
        elif os.path.exists(os.path.join(data_path, "train")) and os.path.exists(os.path.join(data_path, "test")):
            # data_path is already the dataset root
            # check for inertial signals
            if os.path.exists(os.path.join(data_path, "train", "Inertial Signals")):
                uci_root = data_path

    if uci_root is not None:
        print(f"Detected UCI HAR dataset at: {uci_root}")
        X_all, y_all, subjects_all = _load_uci_har_folder(uci_root)
        # UCI labels are 1..6, convert to 0..5
        y_all = y_all.astype(int) - 1
        # Split indices: the dataset's train count is first block size
        # We loaded train then test and concatenated, so infer sizes
        # Count windows in train split from train files
        # We'll create split indices that match original train/test ordering
        # Re-create train/test indices by reading sizes:
        train_root = os.path.join(uci_root, "train")
        train_y_path = os.path.join(train_root, f"y_train.txt")
        n_train = int(np.loadtxt(train_y_path).shape[0])
        n_total = X_all.shape[0]
        train_idx = np.arange(0, n_train)
        test_idx = np.arange(n_train, n_total)
        # Now set train/test arrays
        train_x, train_y = X_all[train_idx], y_all[train_idx]
        test_x, test_y = X_all[test_idx], y_all[test_idx]

        # create a small validation split from train
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.1, random_state=random_seed
        )
        train_idx2, val_idx = next(splitter.split(train_x, train_y))
        train_x, val_x = train_x[train_idx2], train_x[val_idx]
        train_y, val_y = train_y[train_idx2], train_y[val_idx]

        # Fit scaler per-channel across timesteps + windows
        scaler, train_x = _fit_scaler(train_x, scaler_type)
        val_x = _apply_scaler(scaler, val_x)
        test_x = _apply_scaler(scaler, test_x)

        # One-hot labels
        train_y_oh = _one_hot(train_y, num_classes)
        val_y_oh = _one_hot(val_y, num_classes)
        test_y_oh = _one_hot(test_y, num_classes)

        _save_numpy(output_dir, "train_x", train_x)
        _save_numpy(output_dir, "train_y", train_y_oh)
        _save_numpy(output_dir, "val_x", val_x)
        _save_numpy(output_dir, "val_y", val_y_oh)
        _save_numpy(output_dir, "test_x", test_x)
        _save_numpy(output_dir, "test_y", test_y_oh)

        split_payload = {
            "split_type": "uci_presplit",
            "n_train": int(n_train),
            "n_test": int(n_total - n_train),
            "leakage_check": True,
        }
        _save_split_indices(output_dir, split_payload)
        return {
            "train_x": train_x,
            "train_y": train_y_oh,
            "val_x": val_x,
            "val_y": val_y_oh,
            "test_x": test_x,
            "test_y": test_y_oh,
        }

    # --- Fallback: CSV / timestep-aligned series (original behavior) ---
    df = _load_csv(data_path)
    label_col_in_data = _find_label_col(df, label_col)
    if label_col_in_data is None:
        raise ValueError(
            f"Label column '{label_col}' not found in data. Available columns: {df.columns.tolist()}"
        )
    label_col = label_col_in_data
    feature_cols = _resolve_feature_cols(df, feature_cols, label_col, subject_col)
    df = _interpolate_missing(df, feature_cols)

    labels = df[label_col].to_numpy()

    if subject_col and subject_col in df.columns:
        subjects = df[subject_col].unique()
        rng = np.random.default_rng(random_seed)
        rng.shuffle(subjects)
        split_idx = int(len(subjects) * (1 - test_size))
        train_subjects = subjects[:split_idx].tolist()
        test_subjects = subjects[split_idx:].tolist()
        subject_map = {str(subject): "train" for subject in train_subjects}
        subject_map.update({str(subject): "test" for subject in test_subjects})
        split_payload = {
            "split_type": "subject",
            "subject_split": subject_map,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "leakage_check": True,
            "train_count": len(train_subjects),
            "test_count": len(test_subjects),
        }
        _save_split_indices(output_dir, split_payload)

        train_mask = df[subject_col].isin(train_subjects).to_numpy()
        test_mask = df[subject_col].isin(test_subjects).to_numpy()
        data = df[feature_cols].values

        train_x, train_y = _segment_windows(
            data[train_mask],
            labels[train_mask],
            window_len,
            stride,
            label_strategy,
        )
        test_x, test_y = _segment_windows(
            data[test_mask],
            labels[test_mask],
            window_len,
            stride,
            label_strategy,
        )
        assert_no_leakage(np.array(train_subjects), np.array(test_subjects))
    else:
        data = df[feature_cols].values
        num_samples = len(data)
        expected_windows = _window_count(num_samples, window_len, stride)
        if expected_windows == 0:
            raise ValueError(
                "Window length is larger than available samples. "
                f"data rows={num_samples}, window_len={window_len}."
            )
        if len(labels) != num_samples:
            raise ValueError(
                "Labels length must match data rows for timestep-aligned "
                f"labels={len(labels)}, data_rows={num_samples}."
            )
        windows, window_labels = _segment_windows(
            data, labels, window_len, stride, label_strategy
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_seed
        )
        train_idx, test_idx = next(splitter.split(windows, window_labels))
        assert_no_leakage(train_idx, test_idx)
        split_payload = {
            "split_type": "window",
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "leakage_check": True,
            "train_count": len(train_idx),
            "test_count": len(test_idx),
        }
        _save_split_indices(output_dir, split_payload)
        train_x, train_y = windows[train_idx], window_labels[train_idx]
        test_x, test_y = windows[test_idx], window_labels[test_idx]

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.1, random_state=random_seed
    )
    train_idx, val_idx = next(splitter.split(train_x, train_y))
    train_x, val_x = train_x[train_idx], train_x[val_idx]
    train_y, val_y = train_y[train_idx], train_y[val_idx]

    scaler, train_x = _fit_scaler(train_x, scaler_type)
    val_x = _apply_scaler(scaler, val_x)
    test_x = _apply_scaler(scaler, test_x)

    if use_smote:
        if importlib.util.find_spec("imblearn") is None:
            raise ImportError("Install imbalanced-learn for SMOTE usage.")
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=random_seed)
        flat_train = train_x.reshape(train_x.shape[0], -1)
        train_x_res, train_y_res = smote.fit_resample(flat_train, train_y)
        train_x = train_x_res.reshape(-1, window_len, train_x.shape[-1])
        train_y = train_y_res

    train_y_oh = _one_hot(train_y, num_classes)
    val_y_oh = _one_hot(val_y, num_classes)
    test_y_oh = _one_hot(test_y, num_classes)

    _save_numpy(output_dir, "train_x", train_x)
    _save_numpy(output_dir, "train_y", train_y_oh)
    _save_numpy(output_dir, "val_x", val_x)
    _save_numpy(output_dir, "val_y", val_y_oh)
    _save_numpy(output_dir, "test_x", test_x)
    _save_numpy(output_dir, "test_y", test_y_oh)

    return {
        "train_x": train_x,
        "train_y": train_y_oh,
        "val_x": val_x,
        "val_y": val_y_oh,
        "test_x": test_x,
        "test_y": test_y_oh,
    }


def load_prepared_data(output_dir: str) -> Dict[str, np.ndarray]:
    return {
        "train_x": np.load(os.path.join(output_dir, "train_x.npy")),
        "train_y": np.load(os.path.join(output_dir, "train_y.npy")),
        "val_x": np.load(os.path.join(output_dir, "val_x.npy")),
        "val_y": np.load(os.path.join(output_dir, "val_y.npy")),
        "test_x": np.load(os.path.join(output_dir, "test_x.npy")),
        "test_y": np.load(os.path.join(output_dir, "test_y.npy")),
    }