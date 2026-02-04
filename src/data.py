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
    """Find a label column in the dataframe."""

    if label_col in df.columns:
        return label_col
    for candidate in LABEL_FALLBACKS:
        if candidate in df.columns:
            return candidate
    return None


def _load_csv(data_path: str) -> pd.DataFrame:
    """Load dataset into a DataFrame."""

    return pd.read_csv(data_path)


def _resolve_feature_cols(
    df: pd.DataFrame,
    feature_cols: list[str] | None,
    label_col: str,
    subject_col: str | None,
) -> list[str]:
    """Resolve feature column names based on dataset schema."""

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
    """Linear interpolate NaNs column-wise before segmentation."""

    df[feature_cols] = df[feature_cols].interpolate(method="linear", limit_direction="both")
    return df


def _window_count(num_samples: int, window_len: int, stride: int) -> int:
    """Compute number of sliding windows."""

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
    """Create sliding windows and labels from timestep-aligned targets."""

    if label_strategy != "end":
        raise ValueError(
            "label_strategy must be 'end' to match the paper. "
            f"Got '{label_strategy}'."
        )

    windows = []
    window_labels = []
    labels = np.asarray(labels).astype(int)
    for start in range(0, len(data) - window_len + 1, stride):
        end = start + window_len
        windows.append(data[start:end])
        window_labels.append(labels[end - 1])
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
    label_col: str = "activity",
    label_strategy: str = "end",
    use_smote: bool = False,
) -> Dict[str, np.ndarray]:
    """Load, preprocess, split, and optionally oversample the dataset."""

    if labels_path:
        print("Note: --labels-path is ignored. Using labels from dataset.csv.")

    df = _load_csv(data_path)
    label_col_in_data = _find_label_col(df, label_col)
    if label_col_in_data is None:
        raise ValueError(
            f"Label column '{label_col}' not found in data. "
            f"Available columns: {df.columns.tolist()}"
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
                f"labels. labels={len(labels)}, data_rows={num_samples}."
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
    """Load prepared numpy arrays from disk."""

    return {
        "train_x": np.load(os.path.join(output_dir, "train_x.npy")),
        "train_y": np.load(os.path.join(output_dir, "train_y.npy")),
        "val_x": np.load(os.path.join(output_dir, "val_x.npy")),
        "val_y": np.load(os.path.join(output_dir, "val_y.npy")),
        "test_x": np.load(os.path.join(output_dir, "test_x.npy")),
        "test_y": np.load(os.path.join(output_dir, "test_y.npy")),
    }
