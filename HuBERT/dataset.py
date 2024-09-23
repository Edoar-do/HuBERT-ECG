import numpy as np
from loguru import logger
import pandas as pd
from typing import Tuple, Any
import torch
import os
import joblib
from torch.utils.data import Dataset
import neurokit2 as nk
from scipy import signal

SAMPLES_IN_5_SECONDS_AT_500HZ = 2500
SAMPLES_IN_10_SECONDS_AT_500HZ = 5000

class ECGDataset(Dataset):
    def __init__(
        self,
        path_to_dataset_csv : str,
        ecg_dir_path : str,
        downsampling_factor : int = None,
        features_path : str = None,
        kmeans_path : str = None,
        label_start_index : int = 3,
        pretrain : bool = True,
        encode : bool = False,
        beat_based_attention_mask : bool = False,
        random_crop : bool = False,
        return_full_length : bool = False,
        ):
        '''
        Args:
        - `path_to_dataset_csv` = path to dataset in csv format to use. 
        This csv should contain as many binary columns as labels to predict in the multilabel classification task.
        In case of multiclass classification problem, only one column is expected and the values are the integers representing the classes in the range [0, n_classes-1].
        - `ecg_dir_path` = path to the dir where raw ecgs are
        - `downsampling_factor` = integer value indicating the downsampling factor to apply to the ecg signal. Default None
        - `features_path` = path to the dir where dumped features are (extracted from shards or from mid-layers of transformer). Used only when pretrain is true
        - `kmeans_path` = path to a text file that contains the paths to kmeans model used for assigning ensamble labels to the features. Used when pretrain is true
        - `label_start_index` = index of the column in the csv dataset at which labels are. Use when pretrain and encode are false
        - `pretrain` = boolean indicating whether pretraining is in progress or not
        - `encode` = additional boolean value used only to speed up features dumping
        - `beat_based_attention_mask` = optional boolean indicating whether beat-based attention mask should be calculated. Default False
        - `random_crop` = optional boolean indicating whether to randomly crop the ecg signal. Default False. To use only during finetuning and testing to avoid misalignments between signals and features.
        - `return_full_length` = optional boolean indicating whether to return the full length of the ecg signal. Default False. 
        
        The `__getitem__` method returns:
        - ecg_data = (12*2500/downsampling_factor,) float tensor
        - attention_mask = (12*2500/downsampling_factor,) long tensor
        - labels = (ensamble_length, n_tokens,) when `pretrain` is True, (n_classes,) when `pretrain` is False and `encode` is False
        - ecg_filename = string indicating the filename of the ecg item when `encode` is True and `pretrain` is False
        
        The "batch_size" dimension is prepended to the returned tensors when the dataloader is used.
        '''
        
        logger.info(f"Loading dataset from {path_to_dataset_csv}...")

        self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={'filename': str})
        self.ecg_dir_path = ecg_dir_path
        self.downsampling_factor = downsampling_factor
        self.pretrain = pretrain
        self.encode = encode
        self.beat_based_attention_mask = beat_based_attention_mask
        self.random_crop = random_crop
        self.return_full_length = return_full_length

        if pretrain:
            with open(kmeans_path, 'r') as f:
                kmeans_paths = f.readlines()
                
            # filter out commented lines
            kmeans_paths = [path for path in kmeans_paths if not path.startswith("#")]
            
            # logging just for testing purposes
            logger.info(f"Loading {len(kmeans_paths)} kmeans models...")
            for path in kmeans_paths:
                logger.info(f"Loading {path.strip()}...")
                
            self.ensamble_length = len(kmeans_paths)
            self.ensamble_kmeans = [joblib.load(path.strip()) for path in kmeans_paths]
            self.features_path = features_path
        elif not encode:
            self.diagnoses_cols = self.ecg_dataframe.columns.values.tolist()[label_start_index:]
            assert len(self.diagnoses_cols) > 0, "No labels found in the dataset"
            self.weights = self.compute_weights()
            
        np.random.seed(42)
        
    def compute_weights(self):
        logger.info("Computing weights...")        
        if len(self.diagnoses_cols) > 1:
            weights = []
            for label in self.diagnoses_cols:
                count = self.ecg_dataframe[label].sum()
                weight = (self.ecg_dataframe.__len__() - count) / (count + 1e-9)
                weights.append(weight)
            logger.info("Done with the weights.")
        else:
            num_labels = self.ecg_dataframe[self.diagnoses_cols[0]].max() + 1
            weights = num_labels / self.ecg_dataframe[self.diagnoses_cols].value_counts()
            weights = weights.values.tolist()
        return torch.FloatTensor(weights)

    
    def __len__(self):
        return len(self.ecg_dataframe)


    def __getitem__(self, idx):
        
        record = self.ecg_dataframe.iloc[idx]
        ecg_filename = record['filename'] # the mere filename

        if  "/" in ecg_filename: # labelled datasets have filenames as full paths
            ecg_path = ecg_filename
        else:
            ecg_path =  os.path.join(self.ecg_dir_path, ecg_filename)

        ecg_data = np.load(ecg_path) # (12, 5000+)
        
        # check if ecg_data has been corrupted
        if np.isnan(ecg_data).any():
            logger.warning(f"Corrupted data found in {ecg_path}")
            with open("logs.txt", 'a') as f:
                f.write(f"Corrupted data found in {ecg_path}\n")
            return None
            
        # cut to 5 seconds
        if self.random_crop and not self.pretrain and not self.encode: # only for finetuning and testing
            start = np.random.randint(0, ecg_data.shape[1] - SAMPLES_IN_5_SECONDS_AT_500HZ + 1)
            ecg_data = ecg_data[:, start:start+SAMPLES_IN_5_SECONDS_AT_500HZ]
        elif self.return_full_length:
            # in case there are ecgs with varying length in a dataset, extract a random 10 sec window
            # such duration is aligned with durations seen in literature
            start = np.random.randint(0, ecg_data.shape[1] - SAMPLES_IN_10_SECONDS_AT_500HZ + 1)
            ecg_data = ecg_data[:, start:start+SAMPLES_IN_10_SECONDS_AT_500HZ]  
        else:
            # default behavior, used in pretraining and encoding, is to extract the first 5 seconds
            ecg_data = ecg_data[:, :SAMPLES_IN_5_SECONDS_AT_500HZ] # (12, SAMPLES_IN_5_SECONDS_AT_500HZ)
       
        
        mask = np.isnan(ecg_data)
        ecg_data = np.where(mask, ecg_data[~mask].mean(), ecg_data)
        
        # flatten the leads 
        ecg_data = ecg_data.reshape(-1) # (12*SAMPLES_IN_5_SECONDS_AT_500HZ,)
        
        # downsampling 
        if self.downsampling_factor is not None:
            ecg_data = signal.decimate(ecg_data, self.downsampling_factor)
            
        # compute attention mask
        if not self.encode:
            if self.beat_based_attention_mask:
                attention_mask = self.compute_beat_based_attention_mask(ecg_data)
            else:
                attention_mask = self.compute_attention_mask_for_padding(ecg_data)
        
            
        if self.pretrain:
            
            try:
                feat_path = os.path.join(self.features_path, ecg_filename)
                features = np.load(feat_path, allow_pickle=True) #(n_tokens, *) or (64, n_tokens, *), * = 39 if train_iteration==1, else d_model
            except:
                logger.warning(f"features {feat_path} not found")
                with open("logs.txt", 'a') as f:
                    f.write(f"features {feat_path} not  found\n")
                
            ###
            #features = features[:, :16] # time freq            
            #assert features.shape[0] == 93 and features.shape[1] == 16, f"features shape {features.shape} not as expected"
            ###

            try:                
                # [ensamble_length, n_tokens], where values on row i-th are in [0, V_i - 1] and V_i is the number of clusters for the i-th kmeans model
                labels = [kmeans.predict(features).tolist() for kmeans in self.ensamble_kmeans] 
            except ValueError as e:
                logger.warning(f"Exception {e}")
                with open("logs.txt", 'a') as f:
                    f.write(f"features {ecg_filename} cannot be fed into kmeans model {kmeans.cluster_centers_.shape}. features shape {features.shape}\n")
            # labels are (ensamble_length, n_tokens, ) --> becomes (bs, ensamble_length, n_tokens) when batched by dataloader.
            
            output = (
                torch.from_numpy(ecg_data.copy()).float(),
                torch.from_numpy(attention_mask.copy()).long(),
                torch.Tensor(labels).long()    
            )

            return output
        
        elif self.encode:
            
            return torch.from_numpy(ecg_data.copy()).float(), ecg_filename
        
        else: # finetuning
            labels = record[self.diagnoses_cols].values.astype(float if len(self.diagnoses_cols) > 1 else int)
            output = (
                torch.from_numpy(ecg_data.copy()).float(),
                torch.from_numpy(attention_mask.copy()).long(),
                torch.from_numpy(labels.copy()).float() if len(self.diagnoses_cols) > 1 else torch.from_numpy(labels.copy()).long()
            )
            
            return output
   
    def collate(self, batch : Tuple[Any]):
        unpacked = tuple(zip(*batch))
        if self.encode and not self.pretrain:
            ecg_data = torch.stack(unpacked[0], dim=0)
            ecg_filenames = unpacked[1]
            return ecg_data, ecg_filenames
        else:
            return tuple(map(torch.stack, unpacked))
        
    # def compute_attention_mask_for_padding(self, array):
    #     array = array.reshape(12, -1)     # 12 x SAMPLES_IN_5_SECONDS_AT_500HZ   
    #     for index in range(array.shape[1]):
    #         if np.any(array[:, index]):
    #             break
    #     start = index
    #     for index in range(array.shape[1]-1, -1, -1):
    #         if np.any(array[:, index]):
    #             break
    #     end = index
    #     attention_mask = np.zeros(array.shape[1])
    #     attention_mask[start:end+1] = 1
    #     attention_mask = np.repeat([attention_mask], 12, axis=0)
    #     attention_mask = np.concatenate(attention_mask, axis=0)
    #     return attention_mask
          
    def compute_attention_mask_for_padding(self, ecg_data):
        ''' Computes attention mask focusing only on the non-padding values of the sequence'''
        
        example_lead = ecg_data.reshape(12, -1)[0]
        
        # zero-padding is on the right side of the sequence
        # scanning the sequence from right to left, the first non-zero value is the last value of the sequence
        attention_mask = np.ones(len(example_lead), dtype=int)
        for i in range(len(example_lead)-1, -1, -1):
            if example_lead[i] == 0:
                attention_mask[i] = 0
            else:
                break
        attention_mask = np.repeat([attention_mask], 12, axis=0) # 
        attention_mask = np.concatenate(attention_mask, axis=0)
        return attention_mask
    
    def compute_beat_based_attention_mask(self, ecg_data):
        ''' 
        Computes attention mask focusing only on P wave, QRS complex and T wave
        '''
        
        ecg_data = ecg_data.reshape(12, SAMPLES_IN_5_SECONDS_AT_500HZ)
        _, rpeaks = nk.ecg_peaks(ecg_data[1], sampling_rate=500) #compute R peaks from II
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_data[1], rpeaks, sampling_rate=500, method="dwt", show=False, show_type='all')
        signal_dwt['ECG_R_Peaks'] = 0
        signal_dwt['ECG_R_Peaks'].iloc[rpeaks['ECG_R_Peaks']] = 1
        
        p_wave = signal_dwt['ECG_P_Onsets'] | signal_dwt['ECG_P_Offsets'] # binary serie with 1 where P waves start and stop
        qrs_complex = signal_dwt['ECG_Q_Peaks'] | signal_dwt['ECG_S_Peaks'] # binary serie with 1 where QRS complexes start and stop
        t_wave = signal_dwt['ECG_T_Onsets'] | signal_dwt['ECG_T_Offsets'] # binary serie with 1s where T waves start and stop
        
        p_starts_stops = p_wave[p_wave != 0].index.tolist()
        if len(p_starts_stops) % 2 != 0:
            p_starts_stops.append(min(p_starts_stops[-1]+1, 2499))
        p_starts_stops = np.array(p_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each P wave detected
        
        t_starts_stops = t_wave[t_wave != 0].index.tolist()
        if len(t_starts_stops) % 2 != 0:
            t_starts_stops.append(min(t_starts_stops[-1]+1, 2499))
        t_starts_stops = np.array(t_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each T wave detected
        
        
        qrs_starts_stops = qrs_complex[qrs_complex != 0].index.tolist()
        if len(qrs_starts_stops) % 2 != 0:
            qrs_starts_stops.append(min(qrs_starts_stops[-1]+1, 2499))
        qrs_starts_stops = np.array(qrs_starts_stops).reshape(-1, 2) # list of couples <start, stop> for each QRS complex detected
        
        # building the attention mask in order to attend only samples in the p waves
        for start, stop in p_starts_stops:
            p_wave.iloc[start : stop] = 1
        
        # building the attention mask in order to attend only samples in the t waves    
        for start, stop in t_starts_stops:
            t_wave.iloc[start : stop] = 1
        
        # building the attention mask in order to attend only samples in the qrs complexes    
        for start, stop in qrs_starts_stops:
            qrs_complex.iloc[start : stop] = 1
        
        # global attention mask merging all interest regions    
        attention_mask = (p_wave | t_wave | qrs_complex).tolist() 
        attention_mask = np.repeat([attention_mask], 12, axis=0) # since the leads are temporally aligned, interest regions should be located within the same intervals
        attention_mask = np.concatenate(attention_mask, axis=0) 
        
        return attention_mask # 
    
    #def collate(self, batch):
        #ecg_data, attention_masks, labels = tuple(zip(*batch))
        #ecg_data = torch.stack(ecg_data, dim=0)
        #attention_masks = torch.stack(attention_masks, dim=0)
        #labels = torch.stack(labels, dim=0)
        #return ecg_data, attention_masks, labels       
