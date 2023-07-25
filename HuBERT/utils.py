import pandas as pd
import os
import numpy as np

def compute_real_fourier_coeffs_from_discrete_set(discrete_function, N):
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

def dump_ecg_feature(record, in_dir):
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
        data = np.concatenate(data[:, :2500]) #consider only 2500 samples out of 5000
        coefficients = compute_real_fourier_coeffs_from_discrete_set(data, N = 40)
        coefficients = coefficients[:, 0].tolist() + coefficients[:, 1].tolist() #--> total = N * 2
        np.save(os.path.join("/data/ECG_AF/hubert_features", filename[:-4]), coefficients)


if __name__ == "__main__":

    #to compute the features for clustering for 25% of training set

    print("Start reading the dataset...")

    dataframe = pd.read_csv("/data/ECG_AF/train_self_supervised_processed.csv")
    dataframe = dataframe.iloc[:int(0.25*dataframe.__len__())]

    print("Dataset read into a dataframe.")

    print("Start dumping ecg features from part of it...")

    dataframe.apply(dump_ecg_feature, axis=1, args=("/data/ECG_AF/train_self_supervised",))

    print("Features dumped. ")
