import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset

class ECGDatasetSelfSupervised(Dataset):
    def __init__(self, path_to_dataset_csv, ecg_dir_path):
        ''' Params:
            - path_to_dataset_csv: the full path to the csv file containing references to the dataset's instances
            - ecg_dir_path: the path to the directory containing the instances to be retrieved (e.g. "./train_self_supervised")
        '''
        self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={'filename': str})
        self.ecg_dir_path = ecg_dir_path # something like "./***_self_supevised", *** in {train, val, test}

    
    def __len__(self):
        ''' Returns the length of the dataset '''
        return len(self.ecg_dataframe)
    
    def __getitem__(self, idx):
        '''
        Params:
            - idx: integer number that indicates the location of a given instance in the dataframe
        Returns:
            - torch.Tensor (12,5000) containing the pre-processed 12L-ECG
            - age, sex (both nan in the this implementation)
        '''
        record = self.ecg_dataframe.iloc[idx]
        ecg_filename = record['filename'] # the mere filename
        age = np.nan
        sex = np.nan

        ecg_path = os.path.join(self.ecg_dir_path, ecg_filename)

        ecg_data = np.load(ecg_path) #load a pre-processed 12 x 5000 ndarray

        return torch.from_numpy(ecg_data), age, sex