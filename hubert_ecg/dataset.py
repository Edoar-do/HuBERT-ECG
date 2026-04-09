from __future__ import annotations

import os
from typing import Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy import signal
from torch.utils.data import Dataset

import neurokit2 as nk

SAMPLES_IN_5_SECONDS_AT_500HZ = 2500
SAMPLES_IN_10_SECONDS_AT_500HZ = 5000


class ECGDataset(Dataset):
    """
    PyTorch :class:`~torch.utils.data.Dataset` for 12-lead ECG data.

    Supports two loading backends:

    **Per-file mode (default)**
        Each :meth:`__getitem__` call opens a single ``.npy`` file.  Flexible
        but incurs filesystem overhead on large datasets.

    **Memory-mapped mode**
        All ECGs are pre-concatenated in a single ``(N, 12, max_length)``
        :func:`numpy.memmap` file built by
        :func:`~hubert_ecg.utils.build_memmap_dataset`.  Activate by passing
        ``memmap_path``; the CSV must contain a ``memmap_idx`` column (added
        automatically by the builder).

    Args:
        path_to_dataset_csv: Path to dataset CSV.  Must contain a ``filename``
            column; label columns must be the last ones.
        ecg_dir_path: Directory containing ``.npy`` ECG files.  Ignored in
            memmap mode.
        downsampling_factor: Integer decimation factor applied after loading
            (e.g., ``5`` → 500 Hz → 100 Hz).  ``None`` = no downsampling.
        features_path: Directory with pre-computed feature descriptors
            (used only when ``pretrain=True``).
        kmeans_path: Text file listing paths to K-means ``.pkl`` models, one
            per line (used only when ``pretrain=True``).
        label_start_index: CSV column index where label columns begin.
        pretrain: ``True`` when self-supervised pre-training is in progress.
        encode: ``True`` when dumping latent features (skips label loading).
        beat_based_attention_mask: Use ECG wave delineation to build the
            attention mask instead of the simple padding-based one.
        random_crop: Randomly crop a 5-second window from the ECG (useful
            during fine-tuning / test-time augmentation to avoid alignment
            artefacts with pre-computed features).
        return_full_length: Return a 10-second crop instead of 5 seconds.
            Intended for test-time augmentation.
        memmap_path: Path to a ``.memmap`` file produced by
            :func:`~hubert_ecg.utils.build_memmap_dataset`.  When provided,
            ECGs are loaded from the memory-mapped array instead of individual
            ``.npy`` files.  The CSV must contain a ``memmap_idx`` column.

    Returns from ``__getitem__``:
        * ``pretrain=True``: ``(ecg_tensor, attention_mask, labels)``
        * ``encode=True``:   ``(ecg_tensor, filename)``
        * default (fine-tune): ``(ecg_tensor, attention_mask, labels)``
    """

    def __init__(
        self,
        path_to_dataset_csv: str,
        ecg_dir_path: str,
        downsampling_factor: Optional[int] = None,
        features_path: Optional[str] = None,
        kmeans_path: Optional[str] = None,
        label_start_index: int = 3,
        pretrain: bool = True,
        encode: bool = False,
        beat_based_attention_mask: bool = False,
        random_crop: bool = False,
        return_full_length: bool = False,
        memmap_path: Optional[str] = None,
    ):
        logger.info(f"Loading dataset from {path_to_dataset_csv}...")

        self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={"filename": str})
        self.ecg_dir_path = ecg_dir_path
        self.downsampling_factor = downsampling_factor
        self.pretrain = pretrain
        self.encode = encode
        self.beat_based_attention_mask = beat_based_attention_mask
        self.random_crop = random_crop
        self.return_full_length = return_full_length

        # ------------------------------------------------------------------
        # Memory-mapped backend (optional)
        # ------------------------------------------------------------------
        self.memmap: Optional[np.memmap] = None
        if memmap_path is not None:
            N = len(self.ecg_dataframe)
            assert "memmap_idx" in self.ecg_dataframe.columns, (
                "When memmap_path is provided the CSV must contain a "
                "'memmap_idx' column (produced by build_memmap_dataset)."
            )
            # Infer max_length from file size: N * 12 * max_length * itemsize
            itemsize = np.dtype("float32").itemsize
            file_bytes = os.path.getsize(memmap_path)
            max_length = file_bytes // (N * 12 * itemsize)
            self.memmap = np.memmap(
                memmap_path, dtype="float32", mode="r", shape=(N, 12, max_length)
            )
            logger.info(
                f"Memmap mode: {N} ECGs × 12 leads × {max_length} samples "
                f"from {memmap_path}"
            )

        # ------------------------------------------------------------------
        # Pre-training setup
        # ------------------------------------------------------------------
        if pretrain:
            with open(kmeans_path, "r") as f:
                kmeans_paths = f.readlines()
            kmeans_paths = [p for p in kmeans_paths if not p.startswith("#")]
            self.ensamble_length = len(kmeans_paths)
            self.ensamble_kmeans = [joblib.load(p.strip()) for p in kmeans_paths]
            self.features_path = features_path
        elif not encode:
            self.diagnoses_cols = self.ecg_dataframe.columns.values.tolist()[label_start_index:]
            assert len(self.diagnoses_cols) > 0, "No label columns found in the dataset CSV."
            self.weights = self.compute_weights()

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def compute_weights(self) -> torch.FloatTensor:
        logger.info("Computing class weights...")
        if len(self.diagnoses_cols) > 1:
            weights = []
            for label in self.diagnoses_cols:
                count = self.ecg_dataframe[label].sum()
                weight = (len(self.ecg_dataframe) - count) / (count + 1e-9)
                weights.append(weight)
        else:
            num_labels = self.ecg_dataframe[self.diagnoses_cols[0]].max() + 1
            weights = num_labels / self.ecg_dataframe[self.diagnoses_cols].value_counts()
            weights = weights.values.tolist()
        logger.info("Class weights computed.")
        return torch.FloatTensor(weights)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ecg_dataframe)

    def __getitem__(self, idx: int):
        record = self.ecg_dataframe.iloc[idx]
        ecg_filename = record["filename"]

        # ---- load ECG -------------------------------------------------------
        if self.memmap is not None:
            memmap_idx = int(record["memmap_idx"])
            ecg_data = self.memmap[memmap_idx].copy()  # (12, max_length)
        else:
            ecg_path = (
                ecg_filename
                if os.path.isfile(ecg_filename)
                else os.path.join(self.ecg_dir_path, ecg_filename)
            )
            ecg_data = np.load(ecg_path)  # (12, any_duration)

        # ---- crop -----------------------------------------------------------
        if self.pretrain or self.encode:
            ecg_data = ecg_data[:, :SAMPLES_IN_5_SECONDS_AT_500HZ]
        elif self.random_crop:
            start = np.random.randint(
                0, ecg_data.shape[1] - SAMPLES_IN_5_SECONDS_AT_500HZ + 1
            )
            ecg_data = ecg_data[:, start : start + SAMPLES_IN_5_SECONDS_AT_500HZ]
        elif self.return_full_length:
            # Random 10-sec crop for TTA; the caller is responsible for
            # splitting it into two 5-sec crops at inference time.
            start = np.random.randint(
                0, ecg_data.shape[1] - SAMPLES_IN_10_SECONDS_AT_500HZ + 1
            )
            ecg_data = ecg_data[:, start : start + SAMPLES_IN_10_SECONDS_AT_500HZ]
        else:
            ecg_data = ecg_data[:, :SAMPLES_IN_5_SECONDS_AT_500HZ]

        # ---- NaN imputation -------------------------------------------------
        mask = np.isnan(ecg_data)
        if mask.any():
            ecg_data = np.where(mask, ecg_data[~mask].mean(), ecg_data)

        # ---- flatten + decimate ---------------------------------------------
        ecg_data = ecg_data.reshape(-1)  # (12 * length,)
        if self.downsampling_factor is not None:
            ecg_data = signal.decimate(ecg_data, self.downsampling_factor)

        # ---- attention mask -------------------------------------------------
        if not self.encode:
            if self.beat_based_attention_mask:
                attention_mask = self.compute_beat_based_attention_mask(ecg_data)
            else:
                attention_mask = self.compute_attention_mask_for_padding(ecg_data)

        # ---- return ---------------------------------------------------------
        if self.pretrain:
            feat_path = os.path.join(self.features_path, ecg_filename)
            features = np.load(feat_path, allow_pickle=True)
            labels = [kmeans.predict(features).tolist() for kmeans in self.ensamble_kmeans]
            return (
                torch.from_numpy(ecg_data.copy()).float(),
                torch.from_numpy(attention_mask.copy()).long(),
                torch.tensor(labels).long(),
            )

        elif self.encode:
            return torch.from_numpy(ecg_data.copy()).float(), ecg_filename

        else:  # fine-tuning / evaluation
            labels = record[self.diagnoses_cols].values.astype(
                float if len(self.diagnoses_cols) > 1 else int
            )
            label_tensor = (
                torch.from_numpy(labels.copy()).float()
                if len(self.diagnoses_cols) > 1
                else torch.from_numpy(labels.copy()).long()
            )
            return (
                torch.from_numpy(ecg_data.copy()).float(),
                torch.from_numpy(attention_mask.copy()).long(),
                label_tensor,
            )

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    def collate(self, batch: Tuple[Any]) -> Tuple:
        unpacked = tuple(zip(*batch))
        if self.encode and not self.pretrain:
            ecg_data = torch.stack(unpacked[0], dim=0)
            return ecg_data, unpacked[1]  # filenames are strings, not tensors
        return tuple(map(torch.stack, unpacked))

    # ------------------------------------------------------------------
    # Attention mask helpers
    # ------------------------------------------------------------------

    def compute_attention_mask_for_padding(self, array: np.ndarray) -> np.ndarray:
        """Binary mask covering the non-zero (non-padded) region of the ECG."""
        arr = array.reshape(12, -1)
        start = 0
        for i in range(arr.shape[1]):
            if np.any(arr[:, i]):
                start = i
                break
        end = arr.shape[1] - 1
        for i in range(arr.shape[1] - 1, -1, -1):
            if np.any(arr[:, i]):
                end = i
                break
        mask = np.zeros(arr.shape[1])
        mask[start : end + 1] = 1
        mask = np.repeat([mask], 12, axis=0)
        return np.concatenate(mask, axis=0)

    def compute_beat_based_attention_mask(self, ecg_data: np.ndarray) -> np.ndarray:
        """Attention mask focusing only on P wave, QRS complex, and T wave."""
        ecg_data = ecg_data.reshape(12, SAMPLES_IN_5_SECONDS_AT_500HZ)
        _, rpeaks = nk.ecg_peaks(ecg_data[1], sampling_rate=500)
        signal_dwt, _ = nk.ecg_delineate(
            ecg_data[1], rpeaks, sampling_rate=500, method="dwt",
            show=False, show_type="all"
        )
        signal_dwt["ECG_R_Peaks"] = 0
        signal_dwt["ECG_R_Peaks"].iloc[rpeaks["ECG_R_Peaks"]] = 1

        p_wave = signal_dwt["ECG_P_Onsets"] | signal_dwt["ECG_P_Offsets"]
        qrs_complex = signal_dwt["ECG_Q_Peaks"] | signal_dwt["ECG_S_Peaks"]
        t_wave = signal_dwt["ECG_T_Onsets"] | signal_dwt["ECG_T_Offsets"]

        def _fill_intervals(series, max_idx=2499):
            pts = series[series != 0].index.tolist()
            if len(pts) % 2 != 0:
                pts.append(min(pts[-1] + 1, max_idx))
            pairs = np.array(pts).reshape(-1, 2)
            for start, stop in pairs:
                series.iloc[start:stop] = 1
            return series

        p_wave = _fill_intervals(p_wave)
        qrs_complex = _fill_intervals(qrs_complex)
        t_wave = _fill_intervals(t_wave)

        attention_mask = (p_wave | t_wave | qrs_complex).to_numpy()
        attention_mask = np.repeat([attention_mask], 12, axis=0)
        return np.concatenate(attention_mask, axis=0)
