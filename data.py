"""Data loading and preprocessing for HAR dataset."""

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


def _load_csv(data_path: str, labels_path: str | None, label_col: str) -> pd.DataFrame:
    """Load dataset and labels into a single DataFrame."""

    data_df = pd.read_csv(data_path)
    if labels_path and os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)
        if label_col not in labels_df.columns:
            labels_df.columns = [label_col]
        data_df[label_col] = labels_df[label_col].values
    return data_df


def _interpolate_missing(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Linear interpolate NaNs column-wise before segmentation."""

    df[feature_cols] = df[feature_cols].interpolate(method="linear", limit_direction="both")
    return df


def _segment_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows and majority labels."""

    windows = []
    window_labels = []
    for start in range(0, len(data) - window_len + 1, stride):
        end = start + window_len
        window = data[start:end]
        window_label = np.bincount(labels[start:end]).argmax()
        windows.append(window)
        window_labels.append(window_label)
    return np.array(windows), np.array(window_labels)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert labels to one-hot vectors."""

    return np.eye(num_classes)[labels]


def _save_numpy(output_dir: str, name: str, array: np.ndarray) -> str:
    """Save numpy array to output directory."""

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.npy")
    np.save(path, array)
    return path


def _save_split_indices(output_dir: str, payload: Dict) -> str:
    """Save split indices JSON."""

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "split_indices.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _fit_scaler(
    train_x: np.ndarray, scaler_type: str
) -> Tuple[StandardScaler | MinMaxScaler, np.ndarray]:
    """Fit scaler on training data only and transform it."""

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
    """Apply fitted scaler to data."""

    shape = data.shape
    return scaler.transform(data.reshape(-1, shape[-1])).reshape(shape)


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
    num_classes: int = 4,
    feature_cols: list[str] | None = None,
    label_col: str = "label",
    use_smote: bool = False,
) -> Dict[str, np.ndarray]:
    """Load, preprocess, split, and optionally oversample the dataset."""

    df = _load_csv(data_path, labels_path, label_col)
    if feature_cols is None:
        feature_cols = [
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
    df = _interpolate_missing(df, feature_cols)

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

        train_df = df[df[subject_col].isin(train_subjects)].copy()
        test_df = df[df[subject_col].isin(test_subjects)].copy()

        train_x, train_y = _segment_windows(
            train_df[feature_cols].values,
            train_df[label_col].values,
            window_len,
            stride,
        )
        test_x, test_y = _segment_windows(
            test_df[feature_cols].values,
            test_df[label_col].values,
            window_len,
            stride,
        )
        assert_no_leakage(np.array(train_subjects), np.array(test_subjects))
    else:
        data = df[feature_cols].values
        labels = df[label_col].values
        windows, window_labels = _segment_windows(
            data, labels, window_len, stride
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
    """Load prepared numpy arrays from disk."""

    return {
        "train_x": np.load(os.path.join(output_dir, "train_x.npy")),
        "train_y": np.load(os.path.join(output_dir, "train_y.npy")),
        "val_x": np.load(os.path.join(output_dir, "val_x.npy")),
        "val_y": np.load(os.path.join(output_dir, "val_y.npy")),
        "test_x": np.load(os.path.join(output_dir, "test_x.npy")),
        "test_y": np.load(os.path.join(output_dir, "test_y.npy")),
    }
