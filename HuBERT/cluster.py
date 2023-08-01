# This script is intended to be run from command line and trains a K-means model to cluster instances according to their features
# The model is then saved
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score as sil
import os
from loguru import logger
import argparse

def yield_features(features_path):
    for entry in os.scandir(features_path):
        if entry.is_file():
            yield np.load(entry.path) #(93, 19) if train_iteration = 1 ; (93, d_model) if train_iteration > 1

def get_features(generator):
    features = np.concatenate([f for f in generator], axis=0)
    return features

def cluster(args):

    wandb.init(entity="cardi-ai", project="ECG-pretraining", config=wandb.config, group="self-supervised")

    #fixing seed
    np.random.seed(42)

    logger.info("Fetching features from ECGs...")

    generator = yield_features(args.features_path)

    features = get_features(generator)

    logger.info("Features fetched.")

    logger.info("Train kMeans...")
    model = MiniBatchKMeans(
        n_clusters = wandb.config['n_clusters'],
        random_state = 42,
        compute_labels = False,
        batch_size = 10000,
        n_init = 20,
        max_no_improvement = 100,
        reassignment_ratio = 0.0
        ).fit(features)
    logger.info("Training done.")

    sil_score = sil(features, model.labels_, metric='euclidean', random_state=42) #1 best, -1 worst
    sse = model.inertia_

    logger.info(f"Sil_score: {sil_score} - sse: {sse}")

    wandb.log({"sil_score" : sil_score, "sse" : sse})

    if args.train_iteration == 1:
        model_name = "k_means_" + "morphology"
    elif args.train_iteration == 2:
        model_name = "k_means_" + "encoder_6th_layer"
    else:
        model_name = "k_means_" + "encoder_9th_layer"

    model_name += "_" + str(sil_score) + "_" + str(sse)    

    joblib.dump(model, os.path.join("/data/ECG_AF/ECG_pretraining/HuBERT/kmeans", model_name))
    logger.info(f"{model_name} model saved.")   


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cluster ECG features or representations")

    # ECG_HUBERT_FEATURES_PATH = "/data/ECG_AF/hubert_features"
    # ENCODER_6_FEATURES_PATH = "/data/ECG_AF/ECG_pretraining/HuBERT/encoder_6_features"
    # ENCODER_9_FEATURES_PATH = "/data/ECG_AF/ECG_pretraining/HuBERT/encoder_9_features"

    parser.add_argument(
        "features_path",
        help="path to the features to cluster",
        type=str
    )

    # parser.add_argument(
    #     "n_clusters",
    #     help="number of clusters",
    #     type=int,
    # )

    #1, 2 or 3
    parser.add_argument(
        "train_iteration",
        help="first, second or third HuBERT train iteration",
        type=int,
    )
    args = parser.parse_args()
    cluster(args)
