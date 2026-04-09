from __future__ import annotations

import argparse
import concurrent.futures
import os
import random

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torchaudio
from loguru import logger
from scipy import signal
from scipy.fft import fft
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ECGDataset
from .modeling import HuBERTECG, _upgrade_config

SHARD_SIZE_500 = 322
SHARD_SIZE_100 = 64
SHARD_SIZE_50 = 32

CNN_COMPRESSION_FACTOR_500 = 320
CNN_COMPRESSION_FACTOR_100 = 64
CNN_COMPRESSION_FACTOR_50 = 32


def compute_mfcc_features_and_derivatives(x: torch.Tensor, samp_rate: int) -> torch.Tensor:
    """Compute MFCC features and their first and second derivatives."""
    with torch.no_grad():
        x = x.view(1, -1)
        mfccs = torchaudio.compliance.kaldi.mfcc(
            waveform=x,
            sample_frequency=samp_rate,
            use_energy=False,
            frame_length=x.size(-1) / samp_rate * 1000,
            frame_shift=100,
        )  # (time, freq)
        mfccs = mfccs.transpose(0, 1)  # (freq, time)
        deltas = torchaudio.functional.compute_deltas(mfccs)
        ddeltas = torchaudio.functional.compute_deltas(deltas)
        concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
        return concat.transpose(0, 1).contiguous()  # (time, 39)


def get_signal_features(sig: np.ndarray) -> list:
    """Extract 16 time-domain and frequency-domain features from a signal."""
    Min = float(np.min(sig))
    Max = float(np.max(sig))
    Mean = float(np.mean(sig))
    Power = float(np.mean(sig ** 2))
    Rms = float(np.sqrt(Power))
    Var = float(np.var(sig))
    Std = float(np.std(sig))
    Peak = float(np.max(np.abs(sig)))
    P2p = float(np.ptp(sig))
    CrestFactor = float(Peak / (Rms + 1e-9))
    Skew = float(stats.skew(sig))
    Kurtosis = float(stats.kurtosis(sig))

    ft = fft(sig)
    S = np.abs(ft ** 2) / len(sig)
    Max_f = float(np.max(S))
    Sum_f = float(np.sum(S))
    Mean_f = float(np.mean(S))
    Var_f = float(np.var(S))

    return [Min, Max, Mean, Rms, Var, Std, Power, Peak, P2p, CrestFactor,
            Skew, Kurtosis, Max_f, Sum_f, Mean_f, Var_f]


def dump_ecg_features(
    record,
    in_dir: str,
    dest_dir: str,
    mfcc_only: bool,
    time_freq: bool,
    device: torch.device,
    samp_rate: int,
):
    """
    Compute and save morphological feature descriptors for a single ECG.

    Feature types (mutually exclusive flags):
    * ``time_freq=True`` — 16 time/frequency features per shard
    * ``mfcc_only=True`` — 39 MFCC coefficients per shard
    * neither flag — mixed: 16 time/freq + 13 MFCCs per shard (29 total)
    """
    filename = record.filename
    path = os.path.join(dest_dir, filename)

    if samp_rate == 500:
        shard_length = SHARD_SIZE_500
        cnn_compression_factor = CNN_COMPRESSION_FACTOR_500
    elif samp_rate == 100:
        shard_length = SHARD_SIZE_100
        cnn_compression_factor = CNN_COMPRESSION_FACTOR_100
    else:  # 50 Hz
        shard_length = SHARD_SIZE_50
        cnn_compression_factor = CNN_COMPRESSION_FACTOR_50

    if np.load(path).shape[0] != 93:  # skip if features already exist
        data = np.load(os.path.join(in_dir, filename))
        data = data[:, :2500]

        if np.isnan(data).all():
            return

        data = signal.decimate(data, int(500 / samp_rate))
        final_length = data.shape[0] * data.shape[1]

        shortened = []
        if samp_rate == 500:
            for i, row in enumerate(data):
                shortened.append(row[2:-2] if i % 2 == 0 else row[2:-3])
        elif samp_rate == 100:
            for row in data:
                shortened.append(row[2:-2])
        else:  # 50 Hz
            for row in data:
                shortened.append(row[1:-1])

        data = np.concatenate(shortened)
        mask = np.isnan(data)
        data = np.where(mask, data[~mask].mean(), data)

        shards = data.reshape(-1, shard_length)

        features = []
        for shard in shards:
            if time_freq:
                features.append(get_signal_features(shard))
            else:
                mfccs = compute_mfcc_features_and_derivatives(
                    torch.from_numpy(shard).to(device), samp_rate
                )
                mfccs = mfccs.cpu().numpy().tolist()[0]
                if mfcc_only:
                    features.append(mfccs)
                else:
                    features.append(get_signal_features(shard) + mfccs[:13])

        features = np.array(features, dtype=np.float32)
        np.save(os.path.join(dest_dir, filename[:-4]), features)
    else:
        logger.info(f"Skipping {filename} — features already exist.")


def dump_latent_features(
    path_to_dataset_csv: str,
    in_dir: str,
    dest_dir: str,
    start_perc: float,
    end_perc: float,
    hubert: HuBERTECG,
    output_layer: int,
    iteration: int,
    batch_size: int,
    save_csv: bool,
):
    """
    Extract and save latent representations from a trained HuBERT-ECG encoder.

    Representations are extracted from ``output_layer``-th transformer layer
    and saved as ``(n_tokens, D)`` arrays in ``dest_dir``.
    """
    data_set = ECGDataset(
        path_to_dataset_csv=path_to_dataset_csv,
        ecg_dir_path=in_dir,
        downsampling_factor=5,
        pretrain=False,
        encode=True,
    )
    data_set.ecg_dataframe = data_set.ecg_dataframe.iloc[
        int(start_perc * len(data_set)): int(end_perc * len(data_set)) + 1
    ]

    if save_csv:
        csv_out = f"latent_{int((end_perc - start_perc) * 100)}_perc_encoder_{output_layer + 1}_it{iteration}.csv"
        data_set.ecg_dataframe.to_csv(csv_out, index=False)
        logger.info(f"Saved feature-reference CSV to {csv_out}")

    dataloader = DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=5,
        collate_fn=data_set.collate,
        drop_last=False,
    )

    hubert.eval()
    for ecgs, ecg_filenames in tqdm(dataloader, total=len(dataloader)):
        ecgs = ecgs.to(next(hubert.parameters()).device)
        with torch.no_grad():
            out = hubert(
                ecgs,
                attention_mask=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        features = out["hidden_states"][output_layer].cpu().numpy()  # (B, n_tokens, D)
        ecg_paths = [os.path.join(dest_dir, fn[:-4]) for fn in ecg_filenames]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(np.save, ecg_paths, features)
        logger.info(f"Saved batch of latents: shape {features.shape}")


def main(args: argparse.Namespace):
    """Entry point for the feature-dumping script."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if args.train_iteration == 1:
        logger.info("Dumping morphological features...")
        df = pd.read_csv(args.dataframe_path)
        df = df.iloc[int(args.start_perc * len(df)): int(args.end_perc * len(df)) + 1]
        df.apply(
            dump_ecg_features,
            axis=1,
            args=(args.in_dir, args.dest_dir, args.mfcc_only, args.time_freq, device, args.samp_rate),
        )
    else:
        logger.info("Loading HuBERT-ECG to extract latent features...")
        checkpoint = torch.load(args.hubert_path, map_location="cpu")
        config = _upgrade_config(
            checkpoint["model_config"],
            checkpoint.get("pretraining_vocab_sizes"),
        )
        hubert = HuBERTECG(config)
        hubert.load_state_dict(checkpoint["model_state_dict"], strict=False)
        hubert = hubert.to(device)
        logger.info(f"Dumping latents from encoder layer {args.output_layer + 1}...")
        # BUG FIX: was `save_csv=save_csv_for_dumped_features` (undefined variable)
        dump_latent_features(
            args.dataframe_path,
            args.in_dir,
            args.dest_dir,
            args.start_perc,
            args.end_perc,
            hubert,
            args.output_layer,
            args.train_iteration,
            batch_size=args.batch_size,
            save_csv=args.save_csv_for_dumped_features,
        )

    logger.info("Features dumped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump ECG feature descriptors (morphological or latent)."
    )
    parser.add_argument(
        "train_iteration", type=int, choices=[1, 2, 3],
        help="Iteration: 1 = morphological features, 2+ = latent representations.",
    )
    parser.add_argument("dataframe_path", type=str, help="Path to dataset CSV.")
    parser.add_argument("in_dir", type=str, help="Directory containing ECG .npy files.")
    parser.add_argument("dest_dir", type=str, help="Directory to write feature files.")
    parser.add_argument("start_perc", type=float, default=0.0,
                        help="Start fraction of dataset to process.")
    parser.add_argument("end_perc", type=float, default=1.0,
                        help="End fraction of dataset to process.")
    parser.add_argument("--mfcc_only", action="store_true",
                        help="Dump 39-coefficient MFCCs only (mutually exclusive with --time_freq).")
    parser.add_argument("--time_freq", action="store_true",
                        help="Dump 16 time/frequency features only (mutually exclusive with --mfcc_only).")
    parser.add_argument("--hubert_path", type=str,
                        help="Path to HuBERT-ECG checkpoint (required if train_iteration > 1).")
    parser.add_argument("--samp_rate", type=int,
                        help="Sampling rate in Hz (required when computing MFCCs).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for latent extraction (train_iteration > 1 only).")
    parser.add_argument("--output_layer", type=int,
                        help="Encoder layer index for latent extraction (required if train_iteration > 1).")
    parser.add_argument("--save_csv_for_dumped_features", action="store_true",
                        help="Save a CSV referencing the dumped features.")

    args = parser.parse_args()

    if args.start_perc < 0.0 or args.end_perc > 1.0:
        raise ValueError("Percentages must be in [0, 1].")
    if args.train_iteration > 1 and args.hubert_path is None:
        raise ValueError("--hubert_path required when train_iteration > 1.")
    if args.train_iteration > 1 and args.output_layer is None:
        raise ValueError("--output_layer required when train_iteration > 1.")
    if args.train_iteration > 1 and not os.path.isfile(args.hubert_path):
        raise ValueError(f"hubert_path not found: {args.hubert_path}")
    if args.mfcc_only and args.time_freq:
        raise ValueError("--mfcc_only and --time_freq are mutually exclusive.")
    if args.train_iteration == 1 and not args.time_freq and args.samp_rate is None:
        raise ValueError("--samp_rate required when computing MFCC features.")

    main(args)
