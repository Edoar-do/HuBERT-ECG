from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import joblib
import numpy as np
import wandb
from loguru import logger
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ECGDataset


def cluster(args: argparse.Namespace):
    """Fit one or more MiniBatch K-means models on ECG feature descriptors."""
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "hubert-ecg"),
        group=f"clustering_iteration_{args.train_iteration}",
    )

    np.random.seed(42)
    random.seed(42)

    data_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv,
        ecg_dir_path=args.in_dir,
        pretrain=False,
        encode=True,
    )
    dataloader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
    )

    normalizer = preprocessing.Normalizer()
    n_clusters = args.n_clusters_start
    already_loaded = False

    while n_clusters <= args.n_clusters_end:
        logger.info(f"Running K-means with {n_clusters} clusters...")

        if args.model_path is not None and not already_loaded:
            logger.info("Resuming partially fitted model...")
            model = joblib.load(args.model_path)
            loaded_k = model.cluster_centers_.shape[0]
            assert loaded_k == n_clusters, (
                f"Loaded model has {loaded_k} clusters, expected {n_clusters}."
            )
            already_loaded = True
        else:
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                compute_labels=True,
                batch_size=args.batch_size * 93,
                n_init=20,
                max_no_improvement=100,
                reassignment_ratio=0.0,
            )

        for _, filenames in tqdm(dataloader, total=len(dataloader)):
            features = np.concatenate(
                [np.load(os.path.join(args.in_dir, fn)) for fn in filenames], axis=0
            )
            features = normalizer.transform(features)
            model.partial_fit(features)

        sse = model.inertia_
        wandb.log({"k": n_clusters, "SSE": sse})

        suffix = "morphology" if args.train_iteration == 1 else f"encoder_{args.layer}_{args.train_iteration}"
        model_name = f"k_means_{n_clusters}_{suffix}_{sse:.3e}.pkl"
        joblib.dump(model, model_name)
        logger.info(f"Saved {model_name}  (SSE={sse:.3e})")

        n_clusters += args.step


def evaluate_clustering(args: argparse.Namespace):
    """Evaluate a fitted K-means model using Davies-Bouldin and Calinski-Harabász scores."""
    logger.info(f"Evaluating {os.path.basename(args.model_path)}...")
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "hubert-ecg"),
        group=f"clustering_iteration_{args.train_iteration}_evaluation",
    )

    np.random.seed(42)
    random.seed(42)

    data_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv,
        ecg_dir_path=args.in_dir,
        pretrain=False,
        encode=True,
    )
    dataloader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    model = joblib.load(args.model_path)
    db_scores, ch_scores = [], []

    for _, filenames in tqdm(dataloader, total=len(dataloader)):
        features = np.concatenate(
            [np.load(os.path.join(args.in_dir, fn)) for fn in filenames], axis=0
        )
        features = preprocessing.Normalizer().fit_transform(features)
        assignments = model.predict(features)
        db_scores.append(davies_bouldin_score(features, assignments))
        ch_scores.append(calinski_harabasz_score(features, assignments))

    avg_db = float(np.mean(db_scores))
    avg_ch = float(np.mean(ch_scores))
    logger.info(f"Average Davies-Bouldin:    {avg_db:.4f}")
    logger.info(f"Average Calinski-Harabász: {avg_ch:.4f}")
    wandb.log({"avg_davies_bouldin": avg_db, "avg_calinski_harabasz": avg_ch})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster ECG feature descriptors with Mini-Batch K-means."
    )
    parser.add_argument("path_to_dataset_csv", type=str,
                        help="Dataset CSV referencing ECG feature files.")
    parser.add_argument("in_dir", type=str,
                        help="Directory containing feature .npy files.")
    parser.add_argument("train_iteration", type=int,
                        help="Pre-training iteration (used for model naming).")
    parser.add_argument("batch_size", type=int, help="Batch size.")
    parser.add_argument("--cluster", action="store_true",
                        help="Fit new models (mutually exclusive with --model_path evaluation).")
    parser.add_argument("--n_clusters_start", type=int,
                        help="Smallest K to fit (required with --cluster).")
    parser.add_argument("--n_clusters_end", type=int,
                        help="Largest K to fit (required with --cluster).")
    parser.add_argument("--step", type=int,
                        help="Step between K values (required with --cluster).")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved K-means model (evaluate or resume).")
    parser.add_argument("--layer", type=int, default=None,
                        help="Encoder layer used (required when train_iteration >= 2 and --cluster).")

    args = parser.parse_args()

    if args.cluster:
        assert args.n_clusters_start is not None, "--n_clusters_start required with --cluster"
        assert args.n_clusters_end is not None, "--n_clusters_end required with --cluster"
        assert args.step is not None, "--step required with --cluster"
        if args.train_iteration >= 2:
            assert args.layer is not None, "--layer required when train_iteration >= 2"
        cluster(args)
    else:
        assert args.model_path is not None, "--model_path required for evaluation"
        evaluate_clustering(args)
