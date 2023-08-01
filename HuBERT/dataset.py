import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from utils import SHARD_SIZE, get_signal_features, compute_attention_mask

KMEANS_MODEL_PATH_1_ITERATION = "/data/ECG_AF/ECG_pretraining/HuBERT/k_means_morphology.pkl"
KMEANS_MODEL_PATH_2_ITERATION = "/data/ECG_AF/ECG_pretraining/HuBERT/k_means_encoder_6th_layer.pkl"
KMEANS_MODEL_PATH_3_ITERATION = "/data/ECG_AF/ECG_pretraining/HuBERT/k_means_encoder_9th_layer.pkl"

ECG_HUBERT_FEATURES_PATH = "/data/ECG_AF/hubert_features"
ENCODER_6_FEATURES_PATH = "/data/ECG_AF/ECG_pretraining/HuBERT/encoder_6_features"
ENCODER_9_FEATURES_PATH = "/data/ECG_AF/ECG_pretraining/HuBERT/encoder_9_features"

class ECGDatasetSelfSupervised(Dataset):
    def __init__(
        self,
        path_to_dataset_csv : str,
        ecg_dir_path : str,
        train_iteration : int,
        pretrain : bool = True):

        self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={'filename': str})
        self.train_iteration = train_iteration
        self.pretrain = pretrain

        if pretrain:
            if train_iteration == 1:
                self.kMeans_model = joblib.load(KMEANS_MODEL_PATH_1_ITERATION)
            elif train_iteration == 2:
                self.kMeans_model = joblib.load(KMEANS_MODEL_PATH_2_ITERATION)
            else:
                self.kMeans_model = joblib.load(KMEANS_MODEL_PATH_3_ITERATION)
        
        # if reduced:            
        #     self.ecg_dataframe = self.ecg_dataframe.iloc[:int(0.25*self.ecg_dataframe.__len__())]                                            
                                             
        self.ecg_dir_path = ecg_dir_path # something like "./***_self_supevised", *** in {train, val, test}

    
    def __len__(self):
        return len(self.ecg_dataframe)


    def __getitem__(self, idx):
        record = self.ecg_dataframe.iloc[idx]
        ecg_filename = record['filename'] # the mere filename
        ecg_path = os.path.join(self.ecg_dir_path, ecg_filename)
        
        ecg_data = np.load(ecg_path) # (12, 5000)

        ecg_data = np.concatenate(ecg_data[:, :2500]) # (12*2500, )

        attention_mask = compute_attention_mask(ecg_data)

        ecg_data = np.expand_dims(ecg_data, 0) # (1, 12*2500)

        if self.pretrain:
            # !Requires the features to be dumped beforehand!         
            if self.train_iteration == 1:
                features = np.load(os.path.join(ECG_HUBERT_FEATURES_PATH, ecg_filename)) #(93, 19)
            if self.train_iteration == 2:
                features = np.load(os.path.join(ENCODER_6_FEATURES_PATH, ecg_filename)) # (93, d_model)
            else:
                features = np.load(os.path.join(ENCODER_9_FEATURES_PATH, ecg_filename)) # (93, d_model)

            labels = self.kMeans_model.predict(features) # (93, ) --> becomes (bs, 93) when batched by dataloader. Values in [0, V-1] where V is the number of clusters

            return torch.from_numpy(ecg_data).float(), torch.from_numpy(attention_mask).long(), torch.from_numpy(labels).long()
            # returned shapes: (1, 12*2500), (12*2500, ), (93,)
        else:
            #### FINE-TUNING ####
            pass

