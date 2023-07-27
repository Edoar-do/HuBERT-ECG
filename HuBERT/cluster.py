# This script is intended to be run from command line and trains a K-means model to cluster instances according to their features
# The model is then saved
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score as sil
import os
from loguru import logger
import argparse

def cluster(args):
    logger.info("Fetching ECGs' features...")
    features_path = args.features_path
    files = os.listdir(features_path)
    features = []
    # fourier features are saved as np.array (n_features, )
    # encoder's representations are saved as np.array(cnn_out_shape * d_model = n_features, )
    for file in files:
        feat = np.expand_dims(np.load(os.path.join(features_path, file)), 0) #np array (1, n_features)
        features.append(feat)
    features = np.concatenate(features, axis=0) #np.array (n_instances, n_features)
    logger.info("Features fetched.")

    features = preprocessing.normalize(features)

    #model creation and fit
    logger.info("Training a clustering model...")
    model = MiniBatchKMeans(
        n_clusters = args.n_clusters,
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

    logger.info(f"Silhouette score: {sil_score} - SSE: {sse}")

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

    parser.add_argument(
        "n_clusters",
        help="number of clusters",
        type=int,
    )

    #1, 2 or 3
    parser.add_argument(
        "train_iteration",
        help="first, second or third HuBERT train iteration",
        type=int,
    )
    args = parser.parse_args()
    cluster(args)

