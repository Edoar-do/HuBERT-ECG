import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset

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
        ecg_data = np.expand_dims(np.concatenate(ecg_data[:, :2500]), 0) #consider only 2500 samples out of 5000, now (1, 12*2500)

        if self.pretrain:
            #### PRE-TRAINING #### 
            # kmeans-model.predict(saved_features) --> labels (in form of indices)           
            if self.train_iteration == 1:
                features = np.load(os.path.join(ECG_HUBERT_FEATURES_PATH, ecg_filename)) #(n_features, )
            if self.train_iteration == 2:
                features = np.load(os.path.join(ENCODER_6_FEATURES_PATH, ecg_filename)) #(n_features, )
            else:
                features = np.load(os.path.join(ENCODER_9_FEATURES_PATH, ecg_filename)) #(n_features, )
            
            features = np.expand_dims(features, 0) #(1, n_features)

            labels = self.kMeans_model.predict(features) # (1, )

            return torch.from_numpy(ecg_data).unsqueeze(0), torch.from_numpy(labels).long()
            # returned shapes: (1, 1, T), (1)
        else:
            #### FINE-TUNING ####
            pass

