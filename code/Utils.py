import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.distributed import init_process_group, destroy_process_group
import torch

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

# Hierarchical aggregatation
# TODO: For the future it might be interesting to test different aggregations, following different criteria