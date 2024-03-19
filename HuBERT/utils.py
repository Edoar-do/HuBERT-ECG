import pandas as pd
import os
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def load_weights(weights_path, equivalent_classes):
    snomed = pd.read_csv("/data/ECG_AF/snomed.csv", dtype=str)
    weights = pd.read_csv(weights_path, dtype=str)
    weights = weights.set_index('Unnamed: 0')
    
    # replace equivalent classes
    weights = weights.rename(index=equivalent_classes)
    weights = weights.rename(columns=equivalent_classes)
    
    rows = weights.index.values.tolist()
    
    classes = [x for j, x in enumerate(rows) if x not in rows[:j]] # remove duplicates and keep order
    indices = [rows.index(c) for c in classes] # get indices of classes
    weights = weights.to_numpy[np.ix_(indices, indices)] # get weights of classes

    return classes, weights

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

def compute_challenge_metric(weights : np.array, labels : np.array, outputs : np.array, classes : list, normal_class : str):
    
    num_recordings, num_classes = labels.shape
    normal_index = classes.index(normal_class)

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score
    

def cinc2020_metric(labels : pd.DataFrame, outputs : np.array):
    
    assert labels.shape == outputs.shape, "labels and outputs must have the same shape. Found {} and {}".format(labels.shape, outputs.shape)
    
    weights_path = "/data/ECG_AF/weights.csv"
    normal_class = '426783006'
    equivalent_classes = {'59118001': '713427006', '63593006': '284470004', '17338001': '427172004'}

    classes, weights = load_weights(weights_path, equivalent_classes)
    
    # assign columns' names from labels dataframe to outputs columns
    outputs = pd.DataFrame(outputs, columns=labels.columns)    
    
    # replace equivalent classes in labels and merge columns with the same name using max
    labels = labels.rename(columns=equivalent_classes)
    labels = labels.groupby(axis=1, level=0).max()
    labels = labels.to_numpy()
    
    outputs = outputs.rename(columns=equivalent_classes)
    outputs = outputs.groupby(axis=1, level=0).max()
    outputs = outputs.to_numpy()        
        
    return compute_challenge_metric(weights, labels, outputs, classes, normal_class)

def multilabel_split(path_to_csv_file, test_size=0.25, val_size=0.6):
    
    df = pd.read_csv(path_to_csv_file)
    info = df.columns.values[:3]
    labels = df.columns.values[3:]
    
    X = df[info]
    y = df[labels]
    
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    
    for train_index, test_index in msss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    train = pd.concat([X_train, y_train], axis=1)
    
    X = X_test
    y = y_test
    
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1-val_size, random_state=42)
    
    for val_index, test_index in msss.split(X, y):
        X_val, X_test = X.iloc[val_index], X.iloc[test_index]
        y_val, y_test = y.iloc[val_index], y.iloc[test_index]
        
    val = pd.concat([X_val, y_val], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    
    train.to_csv(path_to_csv_file[:-4] + "_train.csv", index=False)
    val.to_csv(path_to_csv_file[:-4] + "_val.csv", index=False)
    test.to_csv(path_to_csv_file[:-4] + "_test.csv", index=False)

if __name__ == '__main__':

    # ptb splits
    # multilabel_split("/data/ECG_AF/ptb_all.csv")  # 71 labels
    # multilabel_split("/data/ECG_AF/ptb_diag.csv") # 44 labels
    # multilabel_split("/data/ECG_AF/ptb_form.csv") # 19 labels  
    # multilabel_split("/data/ECG_AF/ptb_rhythm.csv") # 12 labels
    # multilabel_split("/data/ECG_AF/ptb_diag_subclass.csv") # 23 labels
    # multilabel_split("/data/ECG_AF/ptb_diag_superclass.csv") # 5 labels     











	




