from cv2 import norm
import numpy as np
import pandas as pd
from torchmetrics import Metric
from loguru import logger
import torch
from torchmetrics.utilities import dim_zero_cat

# MAPPING is a dictionary to map conditions that may appear under slightly different names in the weights file or the dataset.
MAPPING = {
            'STE' : 'STE_',
            'STD' : 'STD_',
            'NSR' : 'NORM',
            'IAVB' : '1AVB',
            'IIAVB' : '2AVB',
            'RBBB' : 'CRBBB|RBBB',
            'LBBB' : 'CLBBB|LBBB',
            'CRBBB' : 'CRBBB|RBBB',
            'CLBBB' : 'CLBBB|LBBB',
            'SVPB' : 'PAC|SVPB',
            'PAC' : 'PAC|SVPB',
            'SVPB|PAC' : 'PAC|SVPB',
            'LAE' : 'LAH',
            "LAO" : 'LAH',
            "RAO" : 'RAH',
            "RAE" : 'RAH',
            "LAO/LAE" : 'LAH',
            "RAO/RAE" : 'RAH',
            'LQT' : 'LNGQT',
            'SB' : 'SBRAD',
            'ST' : 'STACH',
            'AFIB' : 'AF',
            'AFLT' : 'AFL',
            'TAB_' : 'TAB',
            'SA' : 'SARRH',
            "QWAVE" : "QAB",
            "PVC" : "VPC|VPB",
            "VPB" : "VPC|VPB",
            "VPC" : "VPC|VPB",
            "PVC|VPB" : "VPC|VPB",
            "IIAVBII" : "2AVB2",
            "CCR" : "-ROT",
            "CR" : "+ROT",
            "_AVB" : "AVB",  
            "SVTAC" : "SVT",
            "LANFB" : "LAFB", 
            "INVT" : "TINV",
            "ISCIN" : "IIS",
            "ISCAN" : "ANMIS",
            'DEATH' : "SAMITROP-DEATH",
            "MOI" : "2AVB1",
            "CHB" : "3AVB",
            "CMI" : "CMIS"                
}

def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.

    num_recordings, num_classes = labels.shape
    A = np.zeros((num_classes, num_classes))    

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A

class CinC2020(Metric):
            '''
            Pytorch implementation of the CinC2020 Physionet Challenge metric.
            Only weighted conditions are considered
            '''
    def __init__(self, conditions, weights_path : str = "/path/to/weights.csv", verbose : bool = False):
        super().__init__()
        self.verbose = verbose
        self.add_state("gt", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        
        self.weights = self.build_weights(weights_path)
        
        self.conditions = self.build_conditions(conditions)
        
        self.scored_conditions = self.weights.columns.tolist()
        
        
    def build_conditions(self, conditions):
        conditions = [c.upper() for c in conditions]
        # rename conditions in self.conditions based on MAPPING        
        for i, condition in enumerate(conditions):
            if condition in MAPPING.keys():
                conditions[i] = MAPPING[condition] # rename conditions in self.conditions based on MAPPING        
        return conditions
    
    def build_weights(self, weights_path):
        weights = pd.read_csv(weights_path, index_col=0)
        # upper the columns and indexes
        weights.columns = weights.columns.str.upper()
        weights.index = weights.index.str.upper()
        weights.rename(columns=MAPPING, index=MAPPING, inplace=True)        
        return weights
                 
    def update(self, preds : torch.Tensor, gt : torch.Tensor) -> None:
        self.gt.append(gt)
        self.preds.append(preds)
        
    def compute(self):
        '''
        - gt is the ground truth with shape batch_size x len(conditions)
        - preds is the predictions with shape batch_size x len(conditions)
        - conditions is the list of conditions names
        '''
        
        device = torch.device('cpu')
        
        gt = dim_zero_cat(self.gt).to(device)
        preds = dim_zero_cat(self.preds).to(device)
        
        assert len(gt) == len(preds), "gt and preds must have the same shape. Found {} and {}".format(gt.shape, preds.shape)
        assert gt.size(1) == len(self.conditions), "Number of conditions in gt and preds must be the same. Found {} and {}".format(gt.shape[1], len(self.conditions))       

                
        if self.verbose:
            logger.info("Dataset Conditions")
            print(self.conditions)
            print()
            logger.info("Scored conditions")
            print(self.scored_conditions)   
            print()     
        
        normal_class = 'NORM'
        
        # create dataframes fro gt and preds. Columns are the conditions
        gt = pd.DataFrame(gt.numpy(), columns=self.conditions)
        preds = pd.DataFrame(preds.numpy(), columns=self.conditions)
        
        # filter out non-scored conditions
        self.conditions = [c for c in self.conditions if c in self.scored_conditions]
        gt = gt[self.conditions]
        preds = preds[self.conditions]
        
        if self.verbose:
            logger.info("Filtered conditions")
            print(self.conditions)
            print()
            logger.info("Weights shape, columns and indexes")
            print(self.weights.shape)
            print(self.weights.columns)
            print(self.weights.index)
        
        # back to tensor
        gt = gt.to_numpy(dtype=float)
        preds = preds.to_numpy(dtype=float)        
        
        assert gt.shape[1] == len(self.conditions), "Number of conditions in gt and preds cols must be the same. Found {} and {}".format(gt.shape[1], len(self.conditions))
        assert gt.shape == preds.shape, "gt and preds must have the same shape. Found {} and {}".format(gt.shape, preds.shape)
                  
        task_weights = self.weights.loc[self.conditions, self.conditions]
        
        if self.verbose:
            logger.info("Task weights")
            print(task_weights)
            
        normal_index = task_weights.columns.tolist().index(normal_class) if normal_class in self.conditions else None
        
        # check that column and index order is preserved
        assert task_weights.index.tolist() == self.conditions, "task_weights index must be the same as conditions. Found {} and {}".format(task_weights.index.tolist(), self.conditions)
        assert task_weights.columns.tolist() == self.conditions, "task_weights columns must be the same as conditions. Found {} and {}".format(task_weights.columns.tolist(), self.conditions)
        
        task_weights = task_weights.to_numpy(dtype=float)
        
        assert task_weights.shape == (len(self.conditions), len(self.conditions)), "task_weights must have shape len(conditions) x len(conditions). Found {}".format(task_weights.shape)
        
        task_weights = torch.from_numpy(task_weights).to(device)
        
        modified_confusion_matrix = compute_modified_confusion_matrix(gt, preds)
        observed_score = np.nansum(task_weights * modified_confusion_matrix)
        
        modified_confusion_matrix = compute_modified_confusion_matrix(gt, gt)
        correct_score = np.nansum(task_weights * modified_confusion_matrix)
        
        if normal_index is not None:
            inactive_outputs = np.zeros(gt.shape, dtype=bool)
            inactive_outputs[:, normal_index] = 1
            modified_confusion_matrix = compute_modified_confusion_matrix(gt, inactive_outputs)
            inactive_score = np.nansum(task_weights * modified_confusion_matrix)
        else:
            inactive_score = 0.0
            
        if self.verbose:
            logger.info("Observed score: {}".format(observed_score))
            logger.info("Correct score: {}".format(correct_score))
            logger.info("Inactive score: {}".format(inactive_score))
        
        return float(observed_score - inactive_score) / float(correct_score - inactive_score) if correct_score != inactive_score else 0.0
