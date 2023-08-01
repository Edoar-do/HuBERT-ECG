import pandas as pd
import os
import numpy as np
import scipy.stats as stats
from scipy.fft import fft
from loguru import logger
import random
import argparse
import torch

SHARD_SIZE = 322

def compute_attention_mask(ecg_data):
    '''Compute attention mask for padding sequences. 0 for padding, 1 for real data. Operates on flattened ecg_data.'''

    padding_positions = np.where(np.diff(np.concatenate(([0], ecg_data == 0, [0]))))[0] #these are start and end positions of padding sequences
    padding_positions = padding_positions.reshape(-1, 2) #reshape to pairs
    attention_mask = np.ones(len(ecg_data), dtype=int)
    for start, end in padding_positions:
        attention_mask[start:end] = 0
    return attention_mask

def get_signal_features(signal):
    '''Extracts 19 features from a signal considering both time and frequency domain'''

    ## TIME DOMAIN ##
    Min = (np.min(signal))
    Max = (np.max(signal))
    Mean = (np.mean(signal))
    Rms = (np.sqrt(np.mean(signal**2)))
    Var = (np.var(signal))
    Std = (np.std(signal))
    Power = (np.mean(signal**2))
    Peak = (np.max(np.abs(signal)))
    P2p = (np.ptp(signal))
    CrestFactor = (np.max(np.abs(signal))/np.sqrt(np.mean(signal**2)))
    Skew = (stats.skew(signal))
    Kurtosis = (stats.kurtosis(signal))
    FormFactor = (np.sqrt(np.mean(signal**2))/np.mean(signal))
    PulseIndicator = (np.max(np.abs(signal))/np.mean(signal))
    ## FREQ DOMAIN ##
    ft = fft(signal)
    S = np.abs(ft**2)/len(signal)
    Max_f = (np.max(S))
    Sum_f = (np.sum(S))
    Mean_f = (np.mean(S))
    Var_f = (np.var(S))    
    Peak_f = (np.max(np.abs(S)))
    Skew_f = (stats.skew(signal))
    Kurtosis_f = (stats.kurtosis(signal))

    features = [Min,Max,Mean,Rms,Var,Std,Power,Peak,P2p,CrestFactor,Skew,Kurtosis,Max_f,Sum_f,Mean_f,Var_f,Peak_f,Skew_f,Kurtosis_f]
    return features

def dump_ecg_features(record, in_dir, dest_dir):
    '''
    Save on disk the features of the ECG signal concatenated in a single vector.

    Args:
        - record from the dataframe apply() function was called on
        - in_dir is the path to th input directory where the data pointed by the record is
        - dest_dir is the path to the output directory where the features will be saved
    '''
    filename = record.filename
    path = os.path.join("/data/ECG_AF/hubert_features", filename)
    if not os.path.isfile(path): #if features do not exist then calculate them, skip otherwise
        data = np.load(os.path.join(in_dir, filename))
        data = data[:, :2500] #(12, 2500)

        shortened_data = []
        for i in range(len(data)):
            if i % 2 == 0:
                shortened_data.append(data[i, 2:-2])
            else:
                shortened_data.append(data[i, 2:-3])
        data = np.concatenate(shortened_data) #(29946,) perfect to fit in shards with 322 samples each

        features = []
        for i in range(0, len(data), SHARD_SIZE):
            shard = data[i:i+SHARD_SIZE]
            features.append(get_signal_features(shard)) #(19,)

        features = np.array(features, dtype=np.float32) #(93, 19)
        np.save(os.path.join(dest_dir, filename[:-4]), features) #saved shape (93, 19)
    # else:
    #     logger.info(f"Skipping {filename} because features already exist")

def dump_ecg_features_from_hubert(record, in_dir, hubert, output_layer, dest_dir):
    '''
    Save on disk at `dest_dir` the features/representations coming out from Hubert encoder's `output_layer`.
    `in_dir` is where to load the raw ecg from, using the ecg filename in the corresponding `record`, while `hubert_path` is where to load the model from
    '''

    filename = record.filename
    path = os.path.join(dest_dir, filename)
    if not os.path.isfile(path):
        data =  np.load(os.path.join(in_dir, filename))
        data = np.concatenate(data[:, :2500]) #(12*2500, )
        data = np.expand_dimes(data, 0) #(1, 12*2500)
        data = torch.from_numpy(data).float() #(1, 12*2500)
        attention_mask = compute_attention_mask(data) #(12*2500, )
        features = hubert(data, torch.from_numpy(attention_mask).long().unsqueeze(0), None, False, True, True)['hidden_states'][output_layer] #(1, 93, d_model)
        features = features.squeeze(0).cpu().numpy() #(93, d_model)
        features = features.astype(np.float32) # to reduce the memory occupation when it will be loaded for clustering
        np.save(os.path.join(dest_dir, filename[:-4]), features) #saved shape (93, d_model)

def main(args):

    #This function is intended to be called offline and is useful to both
    # - dump features (morphological and hidden) to use to train the kmeans models at various train_iterations
    # - dump features (morphological and hidden) to use in the get_item() function of the dataset to compute the labels at various train_iterations

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #fixing seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    logger.info("Loading dataframe...")

    dataframe = pd.read_csv("/data/ECG_AF/train_self_supervised_processed.csv")
    dataframe = dataframe.iloc[:int(args.perc*dataframe.__len__())+1]

    if args.train_iteration == 1:
        logger.info("Dumping morphological features...")
        dataframe.apply(dump_ecg_features, axis=1, args=("/data/ECG_AF/train_self_supervised", "/data/ECG_AF/hubert_features", ))
    elif args.train_iteration == 2:
        logger.info("Dumping Hubert hidden features from 6th layer...")
        hubert = torch.jit.load(args.hubert_path)
        hubert.to(device)
        hubert.eval()
        dataframe.apply(dump_ecg_features_from_hubert, axis=1, args=("/data/ECG_AF/train_self_supervised", hubert, 5, "/data/ECG_AF/ECG_pretraining/HuBERT/encoder_6_features", ))
    else:
        logger.info("Dumping Hubert hidden features from 9th layer...")
        hubert = torch.jit.load(args.hubert_path)
        hubert.to(device)
        hubert.eval()
        dataframe.apply(dump_ecg_features_from_hubert, axis=1, args=("/data/ECG_AF/train_self_supervised", hubert, 8, "/data/ECG_AF/ECG_pretraining/HuBERT/encoder_9_features", ))
    
    logger.info("Features dumped.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "train_iteration",
        help="The training iteration to consider to dump the features."
        "If 1, then dump morphological featrues. If 2+, then dump Hubert hidden features.",
        type=int
    )

    parser.add_argument(
        "perc",
        help="The percentage of the dataset to consider to dump the features.",
        type=float
    )

    #optional parameter to hubert model path
    parser.add_argument(
        "--hubert_path",
        help="The path to the Hubert model to use to extract hidden features.",
        type=str,
    )

    args = parser.parse_args()

    #check if train_iteration is valid
    if args.train_iteration < 1 or args.train_iteration > 3:
        raise ValueError(f"train_iteration must be 1, 2 or 3. Inserted {args.train_iteration}.")

    #check if perc is valid
    if args.perc < 0. or args.perc > 1.:
        raise ValueError(f"perc must be between 0 and 1. Inserted {args.perc}.")

    #check if hubert_path is valid
    if args.train_iteration > 1 and args.hubert_path is None:
        raise ValueError("hubert_path must be specified if train_iteration is 2 or 3.")
    
    if args.train_iteration > 1 and not os.path.isfile(args.hubert_path):
        raise ValueError("hubert_path must be a valid path to a Hubert model.")

    if args.train_iteration == 1 and args.hubert_path is not None:
        logger.warning("hubert_path is not needed if train_iteration is 1. Ignoring it...")


    main(args)
