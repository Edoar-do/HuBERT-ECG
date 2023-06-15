import numpy as np
import pandas as pd
import torch
import os
import wfdb
import re
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.signal import decimate, resample
from biosppy.signals.tools import filter_signal
from Configs import get_configs


# ECGDatasetSelfSupervised.py
class ECGDatasetSelfSupervised(Dataset):
    def __init__(self, path_to_dataset_csv, ecg_dir_path):
        self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={'filename': str})
        self.ecg_dir_path = ecg_dir_path
        self.physio_regex = r'^[A-Z]+\d+'
        configs = get_configs("./code/configs.json")
        self.target_fs = configs["target_fs"]
        self.window = configs["window"]
        self.n_leads = configs["n_leads"]
        self.filter_bandwidth = configs["filter_bandwidth"]
        self.patch_size = configs["patch_size"]
        self.mask_token = configs["mask_token"]
        self.mask_perc = configs["mask_perc"]
    
    def __len__(self):
        return len(self.ecg_dataframe)
    
    def __getitem__(self, idx):
        record = self.ecg_dataframe.iloc[idx]
        ecg_filename = record['filename'] #something like "./***_self_supevised", *** in {train, val, test}
        age = np.nan
        sex = np.nan

        ecg_data = np.load(ecg_filename) #load a pre-processed 12 x 5000 ndarray

        return torch.from_numpy(ecg_data), age, sex