import pandas as pd
import os
import numpy as np

def compute_real_fourier_coeffs_from_discrete_set(discrete_function, N = 40):
    '''
    Computes N * 2 Fourier coefficients of a discrete function
    
    Args: 
        discrete_function is the sequence of points describing the waveform
        N: number of harmonics to consider to approximate the function (N+1 are the # of complex coeffs)
    '''

    result = []
    T = len(discrete_function)
    t = np.arange(T)
    for n in range(N+1):
        an = (2./T) * (discrete_function * np.cos(2 * np.pi * n * t / T)).sum()
        bn = (2./T) * (discrete_function * np.sin(2 * np.pi * n * t / T)).sum()
        result.append((an, bn))
    return np.array(result)

def dump_ecg_features(record, in_dir, N):
    '''
    Save on disk the Fourier coefficients of the ECG signal concatenated in a single vector

    Args:
        - record from the dataframe apply() function was called on
        - in_dir is the path to th input directory where the data pointed by the record is
    '''
    filename = record.filename
    path = os.path.join("/data/ECG_AF/hubert_features", filename)
    if not os.path.isfile(path): #if features do not exist then calculate them, skip otherwise
        data = np.load(os.path.join(in_dir, filename))
        data = np.concatenate(data[:, :2500]) #consider only 2500 samples out of 5000, (2500*12, )
        coefficients = compute_real_fourier_coeffs_from_discrete_set(data, N = N)
        coefficients = coefficients[:, 0].tolist() + coefficients[:, 1].tolist() #--> total = N * 2
        np.save(os.path.join("/data/ECG_AF/hubert_features", filename[:-4]), coefficients) #saved shape (N*2,)

def dump_ecg_features_from_hubert(record, in_dir, hubert, output_layer, dest_dir):
    '''
    Save on disk at `dest_dir` the features/representations coming out from Hubert encoder's `output_layer`.
    `in_dir` is where to load the raw ecg from, using the ecg filename in the corresponding `record`, while `hubert_path` is where to load the model from
    '''

    filename = record.filename
    path = os.path.join(dest_dir, filename)
    if not os.path.isfile(path):
        data =  np.load(os.path.join(in_dir, filename))
        data = np.concatenate(data[., :2500]) #(12*2500, )
        data = np.expand_dimes(data, 0) #(1, 12*2500)
        data = torch.from_numpy(data).unsqueeze(0) #(1, 1, 2500*12)
        # hubert = torch.jit.load(hubert_path) --> pass the model, otherwise it has to be loaded every time
        features = hubert.features(data, output_layer) #np.array (cnn_out_shape, d_model)
        np.save(os.path.join(dest_dir, filename[:-4]), np.concatenate(features)) #saved shape (cnn_out_shape * d_model, )


if __name__ == "__main__":

    #to compute the features for clustering for 25% of training set

    print("Start reading the dataset...")

    dataframe = pd.read_csv("/data/ECG_AF/train_self_supervised_processed.csv")
    # dataframe = dataframe.sample(frac=1) #shuffle only when train_iteration changes, NOT within the same train_iteration
    dataframe = dataframe.iloc[:int(0.1*dataframe.__len__())]

    print("Dataset read into a dataframe.")

    print("Start dumping ecg features from part of it...")

    # train_iteration = 1
    dataframe.apply(dump_ecg_features, axis=1, args=("/data/ECG_AF/train_self_supervised", 40))
    
    #train_iteration = 2
    dataframe.apply(dump_ecg_features_from_hubert, axis=1, args=("/data/ECG_AF/train_self_supervised", hubert, 6, "/data/ECG_AF/ECG_pretraining/HuBERT/encoder_6_features", ))

    #train_iteration = 3
    dataframe.apply(dump_ecg_features_from_hubert, axis=1, args=("/data/ECG_AF/train_self_supervised", hubert, 9, "/data/ECG_AF/ECG_pretraining/HuBERT/encoder_9_features", ))


    

    print("Features dumped. ")
