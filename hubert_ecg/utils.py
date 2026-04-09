from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import sklearn
import torch
from biosppy.signals.tools import filter_signal
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from loguru import logger
from scipy.signal import resample
from torcheval.metrics import MultilabelAUPRC as AUPRC
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy, MulticlassAUPRC
from torchmetrics.classification import MultilabelAUROC as AUROC
from torchmetrics.classification import MultilabelF1Score as F1_score
from torchmetrics.classification import MultilabelPrecision as Precision
from torchmetrics.classification import MultilabelRecall as Recall
from torchmetrics.classification import MultilabelSpecificity as Specificity
from torchmetrics.classification import MulticlassRecall, MulticlassSpecificity
from tqdm import tqdm


# ---------------------------------------------------------------------------
# ECG preprocessing
# ---------------------------------------------------------------------------

def apply_filter(ecg_signal: np.ndarray, filter_bandwidth, fs: int = 500) -> np.ndarray:
    """Bandpass FIR filter to remove noise and baseline wander."""
    order = int(0.3 * fs)
    ecg_signal, _, _ = filter_signal(
        signal=ecg_signal,
        ftype="FIR",
        band="bandpass",
        order=order,
        frequency=filter_bandwidth,
        sampling_rate=fs,
    )
    return ecg_signal


def scaling(seq: np.ndarray, smooth: float = 1e-8) -> np.ndarray:
    """Min-max normalisation to ``[-1, 1]`` per lead."""
    return (
        2
        * (seq - np.min(seq, axis=1)[None].T)
        / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T
        - 1
    )


def ecg_preprocessing(
    ecg_signal: np.ndarray,
    original_frequency: int,
    target_frequency: int = 100,
    band_pass: list = None,
) -> np.ndarray:
    """
    Full ECG preprocessing pipeline:

    1. Bandpass filter (default 0.05 – 47 Hz).
    2. Resample to 500 Hz.
    3. Min-max scale to ``[-1, 1]``.

    Args:
        ecg_signal: ``(12, signal_length)`` float array.
        original_frequency: Sampling rate of the input signal in Hz.
        target_frequency: Unused kept for API compatibility.
        band_pass: ``[low_hz, high_hz]`` for the bandpass filter.

    Returns:
        Preprocessed ``(12, new_length)`` float array at 500 Hz.
    """
    if band_pass is None:
        band_pass = [0.05, 47]
    assert ecg_signal.shape[0] == 12, (
        "ecg_signal must have shape (12, signal_length) for preprocessing."
    )
    ecg_signal = apply_filter(ecg_signal, band_pass, fs=original_frequency)
    ecg_signal = resample(
        ecg_signal,
        int(ecg_signal.shape[-1] * (500 / original_frequency)),
        axis=1,
    )
    return scaling(ecg_signal)


# ---------------------------------------------------------------------------
# Memmap dataset builder
# ---------------------------------------------------------------------------

def _pad_or_crop(ecg: np.ndarray, max_length: int) -> np.ndarray:
    """
    Ensure ``ecg`` (shape ``(12, L)``) has exactly ``max_length`` time steps.

    * ``L < max_length``: right-pad with zeros.
    * ``L > max_length``: centre-crop.
    * ``L == max_length``: return unchanged.
    """
    L = ecg.shape[1]
    if L == max_length:
        return ecg
    if L < max_length:
        pad_width = max_length - L
        return np.pad(ecg, ((0, 0), (0, pad_width)), mode="constant")
    # Centre-crop
    start = (L - max_length) // 2
    return ecg[:, start : start + max_length]


def build_memmap_dataset(
    csv_path: str,
    ecg_dir: str,
    output_path: str,
    max_length: int = 5000,
    n_leads: int = 12,
    dtype: str = "float32",
) -> str:
    """
    Concatenate all ECGs referenced in *csv_path* into a single memory-mapped
    file for fast, zero-overhead random access.

    Creates two files:

    * ``output_path`` — ``(N, 12, max_length)`` float32 :func:`numpy.memmap`
    * ``output_path + ".index.csv"`` — copy of the input CSV with an extra
      ``memmap_idx`` column mapping each row to its position in the memmap.

    ECGs shorter than *max_length* are right-padded with zeros; longer ones
    are centre-cropped.  The function assumes ECGs are stored as ``(12, L)``
    ``.npy`` files at ``{ecg_dir}/{filename}``.

    Args:
        csv_path: Path to the dataset CSV (must have a ``filename`` column).
        ecg_dir: Directory containing the ``.npy`` ECG files.
        output_path: Destination path for the ``.memmap`` file.
        max_length: Number of time steps per ECG (500 Hz × 10 s = 5000).
        n_leads: Number of ECG leads (default 12).
        dtype: NumPy dtype for the memmap array (default ``"float32"``).

    Returns:
        Path to the index CSV (``output_path + ".index.csv"``).

    Example::

        index_csv = build_memmap_dataset(
            "data/train.csv",
            "data/ecgs/",
            "data/train.memmap",
            max_length=5000,
        )
        # Use with ECGDataset:
        dataset = ECGDataset(index_csv, ecg_dir="", memmap_path="data/train.memmap")
    """
    df = pd.read_csv(csv_path, dtype={"filename": str})
    N = len(df)
    logger.info(
        f"Building memmap: {N} ECGs → {output_path}  "
        f"(shape {N}×{n_leads}×{max_length}, dtype={dtype})"
    )

    arr = np.memmap(output_path, dtype=dtype, mode="w+", shape=(N, n_leads, max_length))

    failed = []
    for i, row in tqdm(df.iterrows(), total=N, desc="Building memmap"):
        fname = row["filename"]
        ecg_path = fname if os.path.isfile(fname) else os.path.join(ecg_dir, fname)
        try:
            ecg = np.load(ecg_path).astype(dtype)  # (12, L)
            arr[i] = _pad_or_crop(ecg, max_length)
        except Exception as exc:
            logger.warning(f"Could not load {ecg_path}: {exc}. Filling with zeros.")
            arr[i] = 0.0
            failed.append(fname)

    arr.flush()
    del arr  # close the memmap

    if failed:
        logger.warning(f"{len(failed)} ECG(s) could not be loaded and were zero-filled.")

    df["memmap_idx"] = range(N)
    index_csv = output_path + ".index.csv"
    df.to_csv(index_csv, index=False)
    logger.info(f"Index CSV written to {index_csv}")
    return index_csv


# ---------------------------------------------------------------------------
# Dataset splitting utilities
# ---------------------------------------------------------------------------

def multilabel_split(
    path_to_csv_file: str,
    test_size: float,
    val_size: float,
    fold: str = "",
    random_state=None,
    label_start_index: int = 3,
    save: bool = False,
):
    """Stratified train / validation / test split for multi-label datasets."""
    df = pd.read_csv(path_to_csv_file)
    info = df.columns.values[:label_start_index]
    labels = df.columns.values[label_start_index:]

    X, y = df[info], df[labels]
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    for train_idx, test_idx in msss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    test = pd.concat([X_test, y_test], axis=1)
    X, y = X_train, y_train

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    for train_idx, val_idx in msss.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train = pd.concat([X_train, y_train], axis=1)
    val = pd.concat([X_val, y_val], axis=1)

    if save:
        stem = path_to_csv_file[:-4]
        train.to_csv(f"{stem}_{fold}_train.csv", index=False)
        val.to_csv(f"{stem}_{fold}_val.csv", index=False)
        test.to_csv(f"{stem}_{fold}_test.csv", index=False)

    return train, val, test


def simple_split(
    df: pd.DataFrame,
    test_size: float,
    label_start_index: int = 3,
    random_state=None,
    n_splits: int = 1,
):
    """Stratified train / test split supporting multiple folds."""
    info = df.columns.values[:label_start_index]
    labels = df.columns.values[label_start_index:]
    X, y = df[info], df[labels]

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )
    trains, tests = [], []
    for train_idx, test_idx in msss.split(X, y):
        trains.append(pd.concat([X.iloc[train_idx], y.iloc[train_idx]], axis=1))
        tests.append(pd.concat([X.iloc[test_idx], y.iloc[test_idx]], axis=1))

    return (trains[0], tests[0]) if n_splits == 1 else (trains, tests)


def label_distribution(df: pd.DataFrame, label_start: int = 3, return_counts: bool = False):
    """Compute per-label distribution (proportion or count)."""
    labels = df.columns.values[label_start:]
    dist = df[labels].sum(axis=0)
    return dist if return_counts else dist / len(df)


def ptb_splits(ptb: pd.DataFrame):
    """Split PTB-XL by strat_fold: fold 10 = test, fold 9 = val, <9 = train."""
    return ptb[ptb.strat_fold < 9], ptb[ptb.strat_fold == 9], ptb[ptb.strat_fold == 10]


def drop_nsr(df: pd.DataFrame, perc: float) -> pd.DataFrame:
    """Drop a fraction *perc* of Normal Sinus Rhythm (all-zero label) samples."""
    labels = df.columns.values[3:]
    mask = df[labels].eq(0).all(axis=1)
    to_drop = df[mask].sample(frac=perc).index
    return df.drop(to_drop)


def filter_ptb_xl(
    df: pd.DataFrame,
    min_cnt: int = 10,
    categories: list = None,
):
    """Filter PTB-XL labels by minimum occurrence count."""
    if categories is None:
        categories = [
            "label_all", "label_diag", "label_form", "label_rhythm",
            "label_diag_subclass", "label_diag_superclass",
        ]

    def select_labels(labels, min_cnt):
        lbl, cnt = np.unique(
            [item for sublist in list(labels) for item in sublist],
            return_counts=True,
        )
        return list(lbl[np.where(cnt >= min_cnt)[0]])

    df_out = df.copy()
    lbl_itos = {}
    for cat in categories:
        selected = select_labels(df_out[cat], min_cnt)
        df_out[cat + "_filtered"] = df_out[cat].apply(
            lambda x: [y for y in x if y in selected]
        )
        lbl_itos[cat] = np.array(
            list(set(x for sublist in df_out[cat + "_filtered"] for x in sublist))
        )
        lbl_stoi = {s: i for i, s in enumerate(lbl_itos[cat])}
        df_out[cat + "_filtered_numeric"] = df_out[cat + "_filtered"].apply(
            lambda x: [lbl_stoi[y] for y in x]
        )
    return df_out, lbl_itos


def prepare_data_ptb_xl(data_path: str, min_cnt: int = 1):
    """
    Prepare PTB-XL dataset with multiple label hierarchies.

    Args:
        data_path: Root directory containing ``ptbxl_database.csv`` and
            ``scp_statements.csv``.
        min_cnt: Minimum label occurrence count to retain a label.

    Returns:
        ``(ptb_all, df_ptb_xl, lbl_itos_ptb_xl)``
    """
    ptb_xl_csv = os.path.join(data_path, "ptbxl_database.csv")
    df_ptb_xl = pd.read_csv(ptb_xl_csv, index_col="ecg_id")
    df_ptb_xl.scp_codes = df_ptb_xl.scp_codes.apply(
        lambda x: eval(x.replace("nan", "np.nan"))
    )

    ptb_xl_label_df = pd.read_csv(os.path.join(data_path, "scp_statements.csv"))
    ptb_xl_label_df = ptb_xl_label_df.set_index(ptb_xl_label_df.columns[0])

    ptb_xl_label_diag = ptb_xl_label_df[ptb_xl_label_df.diagnostic > 0]
    ptb_xl_label_form = ptb_xl_label_df[ptb_xl_label_df.form > 0]
    ptb_xl_label_rhythm = ptb_xl_label_df[ptb_xl_label_df.rhythm > 0]

    diag_class_mapping, diag_subclass_mapping = {}, {}
    for id_, row in ptb_xl_label_diag.iterrows():
        if isinstance(row["diagnostic_class"], str):
            diag_class_mapping[id_] = row["diagnostic_class"]
        if isinstance(row["diagnostic_subclass"], str):
            diag_subclass_mapping[id_] = row["diagnostic_subclass"]

    df_ptb_xl["label_all"] = df_ptb_xl.scp_codes.apply(lambda x: list(x.keys()))
    df_ptb_xl["label_diag"] = df_ptb_xl.scp_codes.apply(
        lambda x: [y for y in x if y in ptb_xl_label_diag.index]
    )
    df_ptb_xl["label_form"] = df_ptb_xl.scp_codes.apply(
        lambda x: [y for y in x if y in ptb_xl_label_form.index]
    )
    df_ptb_xl["label_rhythm"] = df_ptb_xl.scp_codes.apply(
        lambda x: [y for y in x if y in ptb_xl_label_rhythm.index]
    )
    df_ptb_xl["label_diag_subclass"] = df_ptb_xl.label_diag.apply(
        lambda x: [diag_subclass_mapping[y] for y in x if y in diag_subclass_mapping]
    )
    df_ptb_xl["label_diag_superclass"] = df_ptb_xl.label_diag.apply(
        lambda x: [diag_class_mapping[y] for y in x if y in diag_class_mapping]
    )
    df_ptb_xl["dataset"] = "ptb_xl"
    df_ptb_xl, lbl_itos_ptb_xl = filter_ptb_xl(df_ptb_xl, min_cnt=min_cnt)

    drop_cols = [
        "patient_id", "height", "weight", "nurse", "site", "device",
        "recording_date", "report", "scp_codes", "heart_axis",
        "infarction_stadium1", "infarction_stadium2", "validated_by",
        "second_opinion", "initial_autogenerated_report", "validated_by_human",
        "baseline_drift", "static_noise", "burst_noise", "electrodes_problems",
        "extra_beats", "pacemaker", "filename_lr",
    ]
    df_ptb_xl = df_ptb_xl.drop(columns=[c for c in drop_cols if c in df_ptb_xl.columns])

    def _make_ptb_split(label_key):
        sub = df_ptb_xl[["filename_hr", "age", "sex", "strat_fold"]].copy()
        sub[lbl_itos_ptb_xl[label_key].tolist()] = 0
        for index, row in df_ptb_xl.iterrows():
            for label in row[label_key + "_filtered"]:
                sub.at[index, label] = 1
        sub.rename(columns={"filename_hr": "filename"}, inplace=True)
        sub["filename"] = sub["filename"].apply(
            lambda x: "HR" + os.path.basename(x).replace("_hr", ".hea.npy")
        )
        return sub

    ptb_all = _make_ptb_split("label_all")
    return ptb_all, df_ptb_xl, lbl_itos_ptb_xl


# ---------------------------------------------------------------------------
# Confidence intervals via bootstrapping
# ---------------------------------------------------------------------------

def get_CI_intervals_by_bootstrapping(
    path_to_csv_test_set: str,
    label_start_index: int = 3,
    N: int = 1000,
    task: str = "multi_label",
    average: str = "none",
    alpha: float = 0.95,
    path_to_performance: Optional[str] = None,
):
    """
    Compute 95% confidence intervals for classification metrics using
    bias-corrected bootstrapping.

    Requires pre-computed probability files in ``probs/{csv_stem}/``.

    Args:
        path_to_csv_test_set: Path to the test-set CSV.
        label_start_index: Column index where labels start.
        N: Number of bootstrap iterations.
        task: ``"multi_label"`` or ``"multi_class"``.
        average: Averaging strategy passed to torchmetrics (``"none"`` for
            per-class intervals).
        alpha: Confidence level (default 0.95 → 95% CI).
        path_to_performance: Optional path to a pre-computed performance CSV.

    Returns:
        ``dict`` mapping metric names to ``(lower, upper)`` bound arrays.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    df = pd.read_csv(path_to_csv_test_set)
    num_labels = len(df.columns[label_start_index:])

    probs_dir = os.path.join(f"probs/{os.path.basename(path_to_csv_test_set)[:-4]}")
    prob_files = sorted(
        [os.path.join(probs_dir, f) for f in os.listdir(probs_dir) if f.endswith(".npy")],
        key=os.path.getmtime,
    )
    probs = np.vstack([np.load(p) for p in prob_files])
    assert probs.shape == (len(df), num_labels)

    task2metric = {
        "multi_label": {
            "test_f1_score": F1_score(num_labels=num_labels, average=average),
            "test_recall": Recall(num_labels=num_labels, average=average),
            "test_precision": Precision(num_labels=num_labels, average=average),
            "test_specificity": Specificity(num_labels=num_labels, average=average),
            "test_auroc": AUROC(num_labels=num_labels, average=average),
            "test_auprc": AUPRC(num_labels=num_labels, average=average),
        },
        "multi_class": {
            "test_accuracy": MulticlassAccuracy(num_classes=num_labels),
            "test_auroc": MulticlassAUROC(num_classes=num_labels),
            "test_recall": MulticlassRecall(num_classes=num_labels),
            "test_specificity": MulticlassSpecificity(num_classes=num_labels),
            "test_auprc": MulticlassAUPRC(num_classes=num_labels),
        },
        "regression": {},
    }
    metrics = task2metric[task]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m in metrics.values():
        m.to(device)

    y_true = torch.tensor(df.iloc[:, label_start_index:].values, dtype=torch.long).to(device)
    y_pred = torch.tensor(probs, dtype=torch.float32).to(device)

    if path_to_performance is None:
        point_estimate = {}
        for name, m in metrics.items():
            m.reset()
            m.update(y_pred, y_true)
            point_estimate[name] = m.compute().cpu().numpy()
    else:
        pe = pd.read_csv(path_to_performance, index_col=0)
        point_estimate = (
            pe.mean(axis=1).to_dict()
            if average != "none"
            else pe.to_dict(orient="index")
        )

    bootstrapped_differences = {name: [] for name in metrics}
    for _ in range(N):
        ids = sklearn.utils.resample(
            range(len(y_true)),
            n_samples=len(y_true),
            stratify=df.iloc[:, label_start_index:].values,
        )
        for name, m in metrics.items():
            m.reset()
            m.update(y_pred[ids], y_true[ids])
            bootstrapped_differences[name].append(
                m.compute().cpu().numpy() - point_estimate[name]
            )

    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + (1.0 - alpha) / 2.0) * 100
    CI_intervals = {}
    for name in metrics:
        diffs = np.array(bootstrapped_differences[name])
        axis = 0 if average == "none" else None
        CI_intervals[name] = (
            point_estimate[name] + np.percentile(diffs, lower_p, axis=axis),
            point_estimate[name] + np.percentile(diffs, upper_p, axis=axis),
        )
    return CI_intervals
