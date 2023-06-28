import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from loguru import logger


class ECGDatasetSelfSupervised(Dataset):
    def __init__(self, path_to_dataset_csv : str, ecg_dir_path : str, reduced : bool = False):
        ''' Params:
            - `path_to_dataset_csv`: the full path to the csv file containing references to the dataset's instances
            - `ecg_dir_path`: the path to the directory containing the instances to be retrieved (e.g. "./train_self_supervised")
            - `reduced`: boolean flag that tells whether considering only 10% of dataset (for faster training exploration). Default: False
        '''
        self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={'filename': str})
        self.reduced = reduced
        
        #take only 10% of the dataset for fast training
        #return to the original length after the exploration phase
        if reduced:            
            self.ecg_dataframe = self.ecg_dataframe.iloc[:int(0.1*self.ecg_dataframe.__len__())]                                            
                                             
        self.ecg_dir_path = ecg_dir_path # something like "./***_self_supevised", *** in {train, val, test}

    
    def __len__(self):
        ''' Returns the length of the dataset '''
        return len(self.ecg_dataframe)
    
    def __getitem__(self, idx):
        '''
        Params:
            - `idx`: integer number that indicates the location of a given instance in the dataframe
        Returns:
            - torch.Tensor (12, window_size) containing the pre-processed 12L-ECG 
            (window_size is usually 5000 but could be 2500 if reduced is True)
            - age, sex (both nan in the this implementation)
        '''
        record = self.ecg_dataframe.iloc[idx]
        ecg_filename = record['filename'] # the mere filename
        age = np.nan
        sex = np.nan

        ecg_path = os.path.join(self.ecg_dir_path, ecg_filename)
        
        ecg_data = np.load(ecg_path)
        ecg_data = ecg_data[:, :5000] 
        
        # try:
        #     ecg_data = np.load(ecg_path)
        # except ValueError as e:
        #     logger.error(ecg_path)
        #     return np.nan, age, sex
		
        
        #takes samples alternated in time --> half the samples and less memory usage and faster training
        #return to the original length after the exploration phase
        if self.reduced:
            return torch.from_numpy(ecg_data[:, ::2]), age, sex
        else:
            return torch.from_numpy(ecg_data), age, sex
