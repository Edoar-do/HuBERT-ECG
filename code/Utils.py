import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.distributed import init_process_group, destroy_process_group
import torch
import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate, resample
from biosppy.signals.tools import filter_signal
import re
import wfdb
import glob

def ddp_setup(rank: int, world_size: int):
    ''' Args:
        rank: unique id of each process
        world_size: total number of processes
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def create_hierarchical_dataset(dataset_path, groups):
    '''Creates a hierarchical dataset from a dataset in csv format.
    One or more columns are grouped together to form a new column whose values are the logical OR (via max) of the grouped columns.
    Groups is a list of lists, where each list contains the names of the columns to be grouped together.'''

    dataframe = pd.read_csv(dataset_path, dtype={'filename': str})
    hierarchical_dataframe = dataframe.copy()
    for i, group in enumerate(groups):
        print(group)
        hierarchical_dataframe["group"+str(i)] = hierarchical_dataframe[group].max(axis=1)
    for group in groups:
        hierarchical_dataframe = hierarchical_dataframe.drop(columns=group)
    dir_path = os.path.dirname(dataset_path)
    hierarchical_dataframe.to_csv(os.path.join(dir_path, "hierarchical_dataset.csv"), index=False)

def train_test_splitter(dataset_path, test_size, val_size):
    '''Splits a dataset in csv format into train, 
    val and test dataset in csv formats'''
    
    full_df = pd.read_csv(dataset_path, dtype={'filename': str})
    label_columns = full_df.columns.values.tolist()[3:]
    train_df, test_df = train_test_split(full_df, test_size=test_size,
                                         random_state = 42,
                                         stratify=full_df[label_columns])

    train_df, val_df = train_test_split(train_df, test_size=val_size,
                                        random_state = 42,
                                        stratify=train_df[label_columns])
    
    dir_path = os.path.dirname(dataset_path)
    train_df.to_csv(os.path.join(dir_path, "train_supervised.csv"), index=False)
    val_df.to_csv(os.path.join(dir_path, "val_supervised.csv"), index=False)
    test_df.to_csv(os.path.join(dir_path, "test_supervised.csv"), index=False)

#iterate over dataframe
def normalize(seq, smooth = 1e-8):
    ''' Normalize a sequence between -1 and 1 '''
    return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1

def deriveLeads(I, II):
    ''' Derive leads III, aVR, aVR, aVF from leads I and II '''
    III = II-I 
    aVR = -(I+II)/2 
    aVL = (I-II)/2 
    aVF = (II-I)/2
    return III, aVR, aVL, aVF

def apply_filter(signal, filter_bandwidth, fs=500):
    ''' Bandpass filtering to remove noise, artifacts etc '''
    # Calculate filter order
    order = int(0.3 * fs)
    # Filter signal
    signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                order=order, frequency=filter_bandwidth, 
                                sampling_rate=fs)
    return signal

def offline_preprocessing(path_to_dataset, path_to_dest_dir, start = 0):
    print("Reading dataset...")
    dataframe = pd.read_csv(path_to_dataset, dtype={'filename': str}, usecols=[0])
    print("Dataset read")
    physio_regex = r'^[A-Z]+\d+'
    if start > 0:
        print(f"Continue preprocessing from index {start} ...")
    else:
        print("Starting preprocessing...")
    for _, series in dataframe.iteritems():
        filename = series['filename']

        #check if the final npy file is already present in dest dir. If s, continue
        new_path = os.path.join(path_to_dest_dir, filename.split('/')[-1])
        if os.path.isfile(new_path + ".npy"):
            #print("skip")
            if i % 100 == 0:
                print(f"Skipped {i} files")
            continue
        if '/' not in filename and re.match(physio_regex, filename): #from Physio          
            #reading file
            file_path = "./PHYSIONET/files/challenge-2021/1.0.3/training/" + filename
            ecg_data = loadmat(file_path.replace(".hea", ".mat"))

            with open(file_path, 'r') as f:
                first_line = f.readline()
            fs = int(first_line.split()[2])

            #reading tracings
            ecg_data = np.asarray(ecg_data['val'], dtype=np.float64)

        elif filename.endswith(".txt"): #from Hefei
            #reading file
            file_path = os.path.join(".", "HEFEI", filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            fs = 500

            #reading tracings
            ecg_data = [list(map(float, line.strip().split())) for line in lines[1:]]
            ecg_data = np.array(ecg_data).T
            III, aVR, aVL, aVF = deriveLeads(ecg_data[0], ecg_data[1])
            ecg_data = np.vstack((ecg_data[:2], III, aVR, aVL, aVF, ecg_data[2:]))

        else: #from TNMG
            file_path = os.path.join(".", "TNMG", filename)                

            #reading file
            record = wfdb.rdrecord(file_path)

            fs = int(record.fs)

            #reading tracings
            ecg_data = record.p_signal.T
            III, aVR, aVL, aVF = deriveLeads(ecg_data[0], ecg_data[1])
            ecg_data = np.vstack((ecg_data[:2], III, aVR, aVL, aVF, ecg_data[2:]))

        #adapting to 500 Hz
        if fs > 500:
            ecg_data = decimate(ecg_data, int(fs / 500))
        elif fs < 500:
            ecg_data = resample(ecg_data, int(ecg_data.shape[-1] * (500 / fs)), axis=1)

        #bandpass filtering
        ecg_data = apply_filter(ecg_data, [0.05, 47])

        #normalize to [-1, 1]
        ecg_data = normalize(ecg_data)

        # zero-padding
        if ecg_data.shape[-1] < 5000:
            padding = ((0, 0), (0, 5000-ecg_data.shape[-1])) # for right zero-padding
            # padding = ((0, 0), ((window-ecg_data.shape[-1])//2, (window-ecg_data.shape[-1]+1)//2))
            ecg_data = np.pad(ecg_data, padding, mode='constant', constant_values=0)

        # ! manca un ecg_data = ecg_data[:, :5000]
        #save this file
        np.save(new_path, ecg_data)

        #at which point are we?
        if i % 100 == 0:
            print(f"Processed {i} files")

def callable_offline_preprocessing(row, path_to_dest_dir=None):
    filename = row['filename']
    #check if the final npy file is already present in dest dir. If s, continue
    physio_regex = r'^[A-Z]+\d+'
    new_path = os.path.join(path_to_dest_dir, filename.split('/')[-1])
    if os.path.isfile(new_path + ".npy"):
        #print("skip")
        return
    if '/' not in filename and re.match(physio_regex, filename): #from Physio          
        #reading file
        file_path = "./PHYSIONET/files/challenge-2021/1.0.3/training/" + filename
        ecg_data = loadmat(file_path.replace(".hea", ".mat"))

        with open(file_path, 'r') as f:
            first_line = f.readline()
        fs = int(first_line.split()[2])

        #reading tracings
        ecg_data = np.asarray(ecg_data['val'], dtype=np.float64)

    elif filename.endswith(".txt"): #from Hefei
        #reading file
        file_path = os.path.join(".", "HEFEI", filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        fs = 500

        #reading tracings
        ecg_data = [list(map(float, line.strip().split())) for line in lines[1:]]
        ecg_data = np.array(ecg_data).T
        III, aVR, aVL, aVF = deriveLeads(ecg_data[0], ecg_data[1])
        ecg_data = np.vstack((ecg_data[:2], III, aVR, aVL, aVF, ecg_data[2:]))

    else: #from TNMG
        file_path = os.path.join(".", "TNMG", filename)                

        #reading file
        record = wfdb.rdrecord(file_path)

        fs = int(record.fs)

        #reading tracings
        ecg_data = record.p_signal.T
        III, aVR, aVL, aVF = deriveLeads(ecg_data[0], ecg_data[1])
        ecg_data = np.vstack((ecg_data[:2], III, aVR, aVL, aVF, ecg_data[2:]))

    #adapting to 500 Hz
    if fs > 500:
        ecg_data = decimate(ecg_data, int(fs / 500))
    elif fs < 500:
        ecg_data = resample(ecg_data, int(ecg_data.shape[-1] * (500 / fs)), axis=1)

    #bandpass filtering
    ecg_data = apply_filter(ecg_data, [0.05, 47])

    #normalize to [-1, 1]
    ecg_data = normalize(ecg_data)

    # zero-padding
    if ecg_data.shape[-1] < 5000:
        padding = ((0, 0), (0, 5000-ecg_data.shape[-1])) # for right zero-padding
        # padding = ((0, 0), ((window-ecg_data.shape[-1])//2, (window-ecg_data.shape[-1]+1)//2))
        ecg_data = np.pad(ecg_data, padding, mode='constant', constant_values=0)

    # ! manca un ecg_data = ecg_data[:, :5000]
    #save this file
    np.save(new_path, ecg_data)

 
def create_supervised_processed_dataset(path_to_csv_supervised_dataset):
    
    dataframe = pd.read_csv(path_to_csv_supervised_dataset, dtype={'filename':str})
    new_dataframe = dataframe.copy()
    
    train_path = "./train_self_supervised"
    val_path = "./val_self_supervised"
    test_path = "./test_self_supervised"
    
    paths = [train_path, val_path, test_path]
    
    physio_regex = r'^[A-Z]+\d+'
    
    for idx, row in dataframe.iterrows():
        
        filename = row['filename']        

        if '/' not in filename and re.match(physio_regex, filename): #from Physio 
                       
            for path in paths:
                new_path = os.path.join(path, filename + ".npy")
                if os.path.isfile(new_path):
                    new_dataframe.loc[idx, 'filename'] = new_path
                    break            

        elif filename.endswith(".txt"): #from Hefei
            
            for path in paths:
                new_path = os.path.join(path, filename + ".npy")
                if os.path.isfile(new_path):
                    new_dataframe.loc[idx, 'filename'] = new_path
                    break            
                
        else: #from TNMG
            
            for path in paths:
                new_path = os.path.join(path, "TNMG" + filename + "_N1.npy")
                if os.path.isfile(new_path):
                    new_dataframe.loc[idx, 'filename'] = new_path
                    break
    
    new_dataframe.to_csv(path_to_csv_supervised_dataset[:-4] + "_processed.csv", index=False)            
            
            
            

    
    
    
# Hierarchical aggregatation
# TODO: For the future it might be interesting to test different aggregations, following different criteria
