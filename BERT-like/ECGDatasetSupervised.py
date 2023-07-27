
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

#TODO: da rivedere e riscrivere alla luce dei cambiamenti e affinché sia compatibile con ***_supervised
#todo: *** i {train, val, test}
#todo: mi aspetto un ecg:data tensor (12, 5000), age e sex float e labels tensor (n_labels)

class ECGDatasetSupervised(Dataset):
        def __init__(self, path_to_dataset_csv, ecg_dir_path):
            self.ecg_dataframe = pd.read_csv(path_to_dataset_csv, dtype={'filename' : str})
            self.ecg_dir_path = ecg_dir_path
            self.physio_regex = r'^[A-Z]+\d+'
            self.diagnoses_columns = self.ecg_dataframe.columns.values.tolist()[3:]
            configs = get_configs("./code/configs.json")
            self.target_fs = configs["target_fs"]
            self.window = configs["window"]
            self.n_leads = configs["n_leads"]
            self.filter_bandwidth = configs["filter_bandwidth"]
            self.patch_size = configs["patch_size"]
            self.mask_token = configs["mask_token"]
            self.mask_perc = configs["mask_perc"]
            with open("./TNMG_xxx.txt", 'r') as f:
                self.tnmg_lines = f.readlines()
        
        def __len__(self):
            return len(self.ecg_dataframe)
        
        def __normalize(self, seq, smooth=1e-8):
            #TODO: normalization could be better done if dataset-based (still on the fly, just applying xmin, xmax) than istance-based like beneath
            #TODO: have to ask Mauro about it
            #TODO: in case, this function needs to be called when distinguishing if filename is physio/TNMG/hefei + requires xmin, xmax additionals params

            ''' Normalize each sequence between -1 and 1 '''
            # todo: ritornare anche min e max per ogni istanza così da fare il mapping back per visualizzazione
            return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1
        
        def __deriveLeads(self, I, II):
            ''' Derive leads III, aVR, aVR, aVF from leads I and II '''
            III = II-I 
            aVR = -(I+II)/2 
            aVL = (I-II)/2 
            aVF = (II-I)/2
            return III, aVR, aVL, aVF
        
        def __apply_filter(self, signal, filter_bandwidth, fs=500):
            ''' Bandpass filtering to remove noise, artifacts etc '''
            # Calculate filter order
            order = int(0.3 * fs)
            # Filter signal
            signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                        order=order, frequency=filter_bandwidth, 
                                        sampling_rate=fs)
            return signal
        
        def patchifier(self, ecg_matrix, patch_size : tuple):
            '''Return a sequence of non-overlapping 1D or 2D patches extracted from the 2D ECG matrix'''

            if ecg_matrix.shape[1] % patch_size[1] != 0 or ecg_matrix.shape[0] % patch_size[0] != 0 or len(patch_size) != 2:
                return
            patches = []
            for i in range(0, ecg_matrix.shape[0], patch_size[0]):
                for j in range(0, ecg_matrix.shape[1], patch_size[1]):
                    patch = torch.from_numpy(ecg_matrix[i:i+patch_size[0], j:j+patch_size[1]])          
                    patches.append(patch)
            return torch.stack(patches, dim=0) #output tensor has shape (n_patches == n_channels, patch_height, patch_width)

        def __getitem__(self, index):
            record = self.ecg_dataframe.iloc[index]
            ecg_label = record[self.diagnoses_columns].to_numpy(dtype=np.float32)
            
            #NOTE: ecg_dir_path should be ECG_AF, in which one can find all subdirs containing the ecg files. 
            # This class is general and has no notion of the purpose of the data (training, validation or testing)
            
            #get filename from dataframe
            ecg_filename = record['filename']

            #get demographics from dataframe
            age = float(record['age']) 
            sex = float(record['sex'])

            if '/' not in ecg_filename and re.match(self.physio_regex, ecg_filename): #from Physio          
                #reading file
                file_path = "./PHYSIONET/files/challenge-2021/1.0.3/training/" + ecg_filename
                ecg_data = loadmat(file_path.replace(".hea", ".mat"))

                with open(file_path, 'r') as f:
                    first_line = f.readline()
                fs = int(first_line.split()[2])

                #reading tracings
                ecg_data = np.asarray(ecg_data['val'], dtype=np.float64)

            elif ecg_filename.endswith(".txt"): #from Hefei
                #reading file
                with open(os.path.join(self.ecg_dir_path, "HEFEI", ecg_filename), 'r') as f:
                    lines = f.readlines()
                
                fs = 500

                #reading tracings
                ecg_data = [list(map(float, line.strip().split())) for line in lines[1:]]
                ecg_data = np.array(ecg_data).T
                III, aVR, aVL, aVF = self.__deriveLeads(ecg_data[0], ecg_data[1])
                ecg_data = np.vstack((ecg_data[:2], III, aVR, aVL, aVF, ecg_data[2:]))

            else: #from TNMG

                file_path = ""

                for xxx in self.tnmg_lines:
                    xxx = xxx.strip()
                    if int(ecg_filename) >= int(xxx + "0000") and int(ecg_filename) <= int(xxx + "9999"):                        
                        file_path = os.path.join(self.ecg_dir_path, "TNMG", "S" + xxx + "0000", "TNMG" + ecg_filename + "_N1") #overwriting file_path if supervised
                        break

                #reading file
                record = wfdb.rdrecord(file_path)

                fs = int(record.fs)

                #reading tracings
                ecg_data = record.p_signal.T
                III, aVR, aVL, aVF = self.__deriveLeads(ecg_data[0], ecg_data[1])
                ecg_data = np.vstack((ecg_data[:2], III, aVR, aVL, aVF, ecg_data[2:]))


            #adapting to 500 Hz
            if fs > self.target_fs:
                ecg_data = decimate(ecg_data, int(fs / self.target_fs))
            elif fs < self.target_fs:
                ecg_data = resample(ecg_data, int(ecg_data.shape[-1] * (self.target_fs / fs)), axis=1)

            #bandpass filtering
            ecg_data = self.__apply_filter(ecg_data, self.filter_bandwidth)

            #normalize to [-1, 1]
            ecg_data = self.__normalize(ecg_data)

            # zero-padding
            if ecg_data.shape[-1] < self.window:
                padding = ((0, 0), (0, self.window-ecg_data.shape[-1])) # for right zero-padding
                # padding = ((0, 0), ((self.window-ecg_data.shape[-1])//2, (self.window-ecg_data.shape[-1]+1)//2))
                ecg_data = np.pad(ecg_data, padding, mode='constant', constant_values=0)

            #discretization using patches
            patches = self.patchifier(ecg_data[:,:self.window], self.patch_size)
        

            return patches, age, sex, torch.Tensor(ecg_label)