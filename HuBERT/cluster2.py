from dataset import ECGDataset
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
import os
from sklearn import preprocessing
import joblib
from tqdm import tqdm
import argparse
import numpy as np
import random
from loguru import logger
import wandb
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def cluster(args):
    
    group = f"clustering_iteration_{args.train_iteration}"
   
    wandb.init(entity="cardi-ai", project="ECG-pretraining", group=group)

    np.random.seed(42)
    random.seed(42)
    
    data_set = ECGDataset(
        path_to_dataset_csv = args.path_to_dataset_csv,
        ecg_dir_path = "/data/ECG_AF/train_self_supervised/",
        pretrain = False,
        encode = True
    )
    
    # just for testing clustering before 2nd iteration
    # todo: to remove once done with this project
    data_set.ecg_dataframe = data_set.ecg_dataframe.iloc[int(0.0 * len(data_set)) : int(0.1 * len(data_set))+1]
  
    dataloader = DataLoader(
        data_set,
        batch_size = args.batch_size,
        num_workers=5,
        shuffle = True,
        pin_memory = True,
        drop_last=True
    )
    
    n_clusters = args.n_clusters_start
    
    while n_clusters <= args.n_clusters_end:
        
        logger.info(f"Training kmeans with {n_clusters} clusters...")
        
        # model creation with same batch size as in the dataloader
        model = MiniBatchKMeans(
            n_clusters = n_clusters,
            random_state = 42,
            compute_labels = True,
            batch_size = args.batch_size * 93,
            n_init = 20,
            max_no_improvement = 100,
            reassignment_ratio = 0.0
        )
        
        # retrive features from dataloader
        
        ### FITTING LOOP ###
        
        for _, filenames in tqdm(dataloader, total = len(dataloader)):

            features = [np.load(os.path.join(args.in_dir, filename)) for filename in filenames] # todo: replace below with this faster approach

            # build the batch from filenames returned by the dataloader
            
            # features = []
            # for filename in filenames:
            #     feat = np.load(os.path.join(args.in_dir, filename))
            #     '''
            #     if feat.shape[1] == 30:
            #         # take first 16 elems, skip one, take last 13 elems --> mixed
            #         feat = np.concatenate((feat[:, :16], feat[:, 17:]), axis = 1)
            #     assert feat.shape[0] == 93 and feat.shape[1] in [16, 29, 39, 768], f"Wrong shape: {feat.shape} in {filename}"
            #     '''
            #     feat = feat[:, :16] # when time_freq
            #     assert feat.shape[0] == 93 and feat.shape[1] == 16, f"detected shape {feat.shape}"
            #     # check there is not any nan
            #     if np.isnan(feat).any():
            #         print(f"NaN in {filename}")
            #         continue
            #     features.append(feat)
                
            features = np.concatenate(features, axis = 0) #(BS * 93, n_features) where n_features = 30, 39, 768 or ...
            
            # normalize features in the batch
            features = preprocessing.Normalizer().fit_transform(features)
            
            # train kmeans
            model.partial_fit(features) 
            
        ### END OF FITTING LOOP ###
            
        sse = model.inertia_
        wandb.log({"k" : n_clusters, "SSE" : sse})
        
        layer = args.path_to_dataset_csv.split("_")[3] # latent_10_perc_{layer}_layer.csv

        if args.train_iteration == 1:
            model_name = "k_means_" + str(n_clusters) +  "_morphology"
        elif args.train_iteration == 2:
            model_name = "k_means_" + str(n_clusters) + f"_encoder_" + str(layer)
        else:
            model_name = "k_means_" + str(n_clusters) + "_encoder_9th_layer"
            
        sse = "{:e}".format(int(sse))
        
        logger.info(f"SSE: {sse}")
        
        model_name +=  "_" + sse + ".pkl"
        
        joblib.dump(model, os.path.join("/data/ECG_AF/ECG_pretraining/HuBERT/kmeans", model_name))
        logger.info(f"{model_name} model saved.\n")
        
        n_clusters = n_clusters + args.step
 
def evaluate_clustering(args):
    
    logger.info(f"Ready to evaluate clustering model {args.model_path.split('/')[-1]}")
    
    group = f"clustering_iteration_{args.train_iteration}_evaluation"
    
    wandb.init(entity="cardi-ai", project="ECG-pretraining", group=group)
    
    np.random.seed(42)
    random.seed(42)
    
    data_set = ECGDataset(
        path_to_dataset_csv = args.path_to_dataset_csv,
        ecg_dir_path = args.in_dir,
        pretrain = False,
        encode = True
    )
    
  
    dataloader = DataLoader(
        data_set,
        batch_size = args.batch_size,
        num_workers=5,
        shuffle = True,
        pin_memory = True,
        drop_last=True
    )
    
    model = joblib.load(args.model_path)
    
    db_scores = []
    ch_scores = []
    
    for _, filenames in tqdm(dataloader, total = len(dataloader)):

            features = [np.load(os.path.join(args.in_dir, filename)) for filename in filenames] 
            
            # build the batch from filenames returned by the dataloader
            
            # features = []
            # for filename in filenames:
            #     feat = np.load(os.path.join(args.in_dir, filename))
                
            #     if feat.shape[1] == 30:
            #         # take first 16 elems, skip one, take last 13 elems
            #         feat = np.concatenate((feat[:, :16], feat[:, 17:]), axis = 1)
            #     assert feat.shape[0] == 93 and feat.shape[1] in [16, 29, 39, 768], f"Wrong shape: {feat.shape} in {filename}"
                
            #     # feat = feat[:, :16] # when time_freq
            #     # assert feat.shape[0] == 93 and feat.shape[1] == 16, f"detected shape {feat.shape}"
            #     # check there is not any nan
            #     if np.isnan(feat).any():
            #         print(f"NaN in {filename}")
            #         continue
            #     features.append(feat)
                
            features = np.concatenate(features, axis = 0) #(BS * 93, n_features) where n_features = 30, 39, 768 or ...
            
            # normalize features in the batch
            features = preprocessing.Normalizer().fit_transform(features)
            
            assignments = model.predict(features) # (BS * 93, ) containing values in [0, n_clusters - 1]
            
            db_scores.append(davies_bouldin_score(features, assignments))
            ch_scores.append(calinski_harabasz_score(features, assignments))
            
    logger.info(f"Average Davies-Bouldin score: {np.mean(db_scores)}")
    logger.info(f"Average Calinski-Harabasz score: {np.mean(ch_scores)}")
    wandb.log({"Average Davies-Bouldin score" : np.mean(db_scores)})
    wandb.log({"Average Calinski-Harabsz score" : np.mean(ch_scores)})
           
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Cluster ECG features or representations")
    
    parser.add_argument(
        "path_to_dataset_csv",
        help="path to the dataset in csv format to use",
        type=str
    )
    
    parser.add_argument(
        "in_dir",
        help="path to the directory containing the features to cluster",
        type=str
    )
    
    parser.add_argument(
        "--cluster", 
        help="Whether to cluster or evaluate a model",
        action="store_true"
    )
    
    parser.add_argument(
        "--n_clusters_start",
        help="initial number of clusters",
        type=int,
    )
    
    parser.add_argument(
        "--n_clusters_end",
        help="final number of clusters",
        type=int,
    )
    
    parser.add_argument(
        "--step",
        help="step between two consecutive number of clusters",
        type=int,
    )
    
    parser.add_argument(
        "train_iteration",
        help="iteration of the training",
        type=int,
    )
    
    parser.add_argument(
        "batch_size",
        help="batch size",
        type=int,
    )
    
    parser.add_argument(
        "--model_path",
        help="path to the model to evaluate",
        type=str,
        default=None
    )
    
    args = parser.parse_args()
    
    #check args
    if args.cluster:
        assert args.n_clusters_start is not None, "n_clusters_start must be specified"
        assert args.n_clusters_end is not None, "n_clusters_end must be specified"
        assert args.step is not None, "step must be specified"
        cluster(args)
    else:
        assert args.model_path is not None, "model_path must be specified"
        evaluate_clustering(args)
                
            
        
