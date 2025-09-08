import pandas as pd
import os
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import torch
from torchmetrics.classification import MultilabelF1Score as F1_score
from torchmetrics.classification import MultilabelRecall as Recall
from torchmetrics.classification import MultilabelPrecision as Precision
from torchmetrics.classification import MultilabelSpecificity as Specificity
from torchmetrics.classification import MultilabelAUROC as AUROC
from torchmetrics.classification import MulticlassRecall, MulticlassSpecificity
from torcheval.metrics import MultilabelAUPRC as AUPRC
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy, MulticlassAUPRC
import argparse
from scipy.signal import resample
from biosppy.signals.tools import filter_signal

def get_CI_intervals_by_bootstrapping(path_to_csv_test_set, label_start_index=3, N=1000, task='multi_label', average="none", alpha=0.95, path_to_performance=None):
    """
    Computes 95% confidence intervals for classification metrics using bias-corrected bootstrapping.
    
    Args:
        path_to_csv_test_set (str): Path to the CSV file containing the test set.
        label_start_index (int): Column index where labels start in the CSV file.
        N (int): Number of bootstrap iterations.
        task (str): Task type ('multi_label' or 'multi_class') used to find the right metrics.

    Returns:
        dict: Dictionary containing confidence intervals for each metric.

    Ref: https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf 
    empirical bootstrap rather than bootstrap percentiles
    """

    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load dataset
    df = pd.read_csv(path_to_csv_test_set)
    num_labels = len(df.columns[label_start_index:])

    # Load precomputed probability outputs
    probs_dir = os.path.join(f"probs/{os.path.basename(path_to_csv_test_set)[:-4]}")
    probs = [os.path.join(probs_dir, prob) for prob in os.listdir(probs_dir) if prob.endswith(".npy")]
    probs.sort(key=lambda x: os.path.getmtime(x))
    probs = [np.load(prob) for prob in probs]
    probs = np.vstack(probs)

    print(f"Fetched soft predictions and stacked into a {probs.shape} tensor") 
    print()

    # Sanity checks
    assert probs.shape == (len(df), num_labels), f"Mismatch in shape: expected ({len(df)}, {num_labels}), got {probs.shape}"

    # Define metrics based on the task type
    task2metric = {
        'multi_label' : {
            'test_f1_score' : F1_score(num_labels=num_labels, average=average), 
            'test_recall' : Recall(num_labels=num_labels, average=average),
            'test_precision' : Precision(num_labels=num_labels, average=average),
            'test_specificity' : Specificity(num_labels=num_labels, average=average),
            'test_auroc' : AUROC(num_labels=num_labels, average=average), 
            'test_auprc' : AUPRC(num_labels=num_labels, average=average)
                         },
        
        'multi_class' : {
            'test_accuracy' : MulticlassAccuracy(num_classes=num_labels),
            'test_auroc' : MulticlassAUROC(num_classes=num_labels),
            'test_recall' : MulticlassRecall(num_classes=num_labels),
            'test_specificity' : MulticlassSpecificity(num_classes=num_labels),
            'test_auprc' : MulticlassAUPRC(num_classes=num_labels)
                        },
        
        'regression' : {}
                         
    }

    metrics = task2metric[task]

    # Move metrics to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for metric in metrics.values():
        metric.to(device)


    y_true = torch.tensor(df.iloc[:, label_start_index:].values, dtype=torch.long).to(device)
    y_pred = torch.tensor(probs, dtype=torch.float32).to(device)

    print("Converted to torch tensors", y_true.size(), y_pred.size())

    assert y_true.size() == y_pred.size(), f"Size mismatch: y_true {y_true.size()}, y_pred {y_pred.size()}"

    # point estimate
    print("POINT ESTIMATES")

    if path_to_performance is None:
        point_estimate = {}
        for metric_name, metric in metrics.items():
            metric.reset()
            metric.update(y_pred, y_true)
            val = metric.compute().cpu().numpy()
            print(f"{metric_name} : {val}")
            point_estimate[metric_name] = val
    else:
        point_estimate = pd.read_csv(path_to_performance, index_col=0) # metrics name as index, labels as columns

        if average != "none":
            point_estimate = point_estimate.mean(axis=1).to_dict()
        else:
            point_estimate = point_estimate.to_dict(orient='index')


    print()

    # bootstrapping
    bootstrapped_differences = {metric_name : [] for metric_name in metrics.keys()}
    for i in range(N):
        ids = resample(range(len(y_true)), n_samples=len(y_true), stratify=df.iloc[:, label_start_index:].values)
        for metric_name, metric in metrics.items():
            metric.reset()
            metric.update(y_pred[ids], y_true[ids])
            val = metric.compute().cpu().numpy() # numpy array of shape (num_classes,) if average is "none", else a scalar numpy array
            bootstrapped_differences[metric_name].append(val - point_estimate[metric_name]) # list of N numpy arrays of shape (num_classes,) if average is "none", else a list of N scalar numpy arrays

    CI_intervals = {}
    lower_bound = ((1.0 - alpha) / 2.0) * 100
    upper_bound = (alpha + ((1.0 - alpha) / 2.0)) * 100

    for metric_name in metrics.keys():
        diffs = np.array(bootstrapped_differences[metric_name]) # transforms list of differences into numpy array of differences

        # Compute percentiles correctly based on whether we have class-wise metrics or single values
        lower_percentile = np.percentile(diffs, lower_bound, axis=0 if average == "none" else None)
        upper_percentile = np.percentile(diffs, upper_bound, axis=0 if average == "none" else None)

        # Add the point estimate back
        CI_intervals[metric_name] = (
            point_estimate[metric_name] + lower_percentile,
            point_estimate[metric_name] + upper_percentile
        )

    # Print results
    print("\nConfidence Intervals:")
    for metric_name, (lower, upper) in CI_intervals.items():
        if average == "none":
            print(f"{metric_name}:")
            for i, (l, u) in enumerate(zip(lower, upper)):
                print(f"  Class {i}: {l:.4f} - {u:.4f}")
        else:
            print(f"{metric_name}: {lower:.4f} - {upper:.4f}")
            

def multilabel_split(path_to_csv_file, test_size, val_size, fold='', random_state=None, label_start_index=3, save=False):
    '''
    Split a multilabel dataset into train, validation and test sets preserving the distribution of labels.
    '''
    
    df = pd.read_csv(path_to_csv_file)
    info = df.columns.values[:label_start_index]
    labels = df.columns.values[label_start_index:]
    
    X = df[info]
    y = df[labels]
    
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    for train_index, test_index in msss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
          
    test = pd.concat([X_test, y_test], axis=1)
    
    X = X_train
    y = y_train
    
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    
    for train_index, val_index in msss.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
    train = pd.concat([X_train, y_train], axis=1)
    val = pd.concat([X_val, y_val], axis=1)
    
    if save:    
        train.to_csv(path_to_csv_file[:-4] + "_" + fold + "_train.csv", index=False)
        val.to_csv(path_to_csv_file[:-4] + "_" + fold + "_val.csv", index=False)
        test.to_csv(path_to_csv_file[:-4] + "_" + fold + "_test.csv", index=False)
    return train, val, test

def prepare_data_ptb_xl(min_cnt=1):
    
    data_path = "/data/ECG_AF/"
    
    # reading df
    ptb_xl_csv = os.path.join(data_path, "ptbxl_database.csv")
    df_ptb_xl=pd.read_csv(ptb_xl_csv, index_col="ecg_id")
    
    #print(df_ptb_xl.columns)
    df_ptb_xl.scp_codes=df_ptb_xl.scp_codes.apply(lambda x: eval(x.replace("nan","np.nan")))

    # preparing labels
    ptb_xl_label_df = pd.read_csv(os.path.join(data_path, "scp_statements.csv"))
    ptb_xl_label_df=ptb_xl_label_df.set_index(ptb_xl_label_df.columns[0])

    ptb_xl_label_diag= ptb_xl_label_df[ptb_xl_label_df.diagnostic > 0]
    ptb_xl_label_form= ptb_xl_label_df[ptb_xl_label_df.form > 0]
    ptb_xl_label_rhythm= ptb_xl_label_df[ptb_xl_label_df.rhythm > 0]

    diag_class_mapping={}
    diag_subclass_mapping={}
    for id,row in ptb_xl_label_diag.iterrows():
        if(isinstance(row["diagnostic_class"],str)):
            diag_class_mapping[id]=row["diagnostic_class"]
        if(isinstance(row["diagnostic_subclass"],str)):
            diag_subclass_mapping[id]=row["diagnostic_subclass"]
            
    # NOTE: EVERY LABEL THAT OCCURS IN ENTRIES IS CONSIDERED AS ONE, NO MATTER ITS LIKELIHOOD --> NON-SENSE. WOULD BE WISER CONSIDERING 1 IF LIKELIHOOD > 0.5

    df_ptb_xl["label_all"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys()])
    df_ptb_xl["label_diag"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_diag.index])
    df_ptb_xl["label_form"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_form.index])
    df_ptb_xl["label_rhythm"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_rhythm.index])

    df_ptb_xl["label_diag_subclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_subclass_mapping[y] for y in x if y in diag_subclass_mapping])
    df_ptb_xl["label_diag_superclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_class_mapping[y] for y in x if y in diag_class_mapping])

    df_ptb_xl["dataset"]="ptb_xl"
    #filter (can be reapplied at any time)
    df_ptb_xl, lbl_itos_ptb_xl = filter_ptb_xl(df_ptb_xl,min_cnt=min_cnt)
    
    # drop useless cols
    df_ptb_xl = df_ptb_xl.drop(columns=['patient_id', 'height', 'weight', 'nurse', 'site' , 'device',
                    'recording_date' ,'report' ,'scp_codes' ,'heart_axis' ,'infarction_stadium1',
                    'infarction_stadium2' ,'validated_by' ,'second_opinion',
                    'initial_autogenerated_report', 'validated_by_human', 'baseline_drift',
                    'static_noise', 'burst_noise', 'electrodes_problems', 'extra_beats',
                    'pacemaker', 'filename_lr'])
    
    # PTB ALL
    ptb_all = df_ptb_xl[['filename_hr', 'age', 'sex', 'strat_fold']]
    ptb_all[lbl_itos_ptb_xl['label_all'].tolist()] = 0
    for index, row in df_ptb_xl.iterrows():
        for label in row['label_all_filtered']:
            ptb_all.at[index, label] = 1
    ptb_all.rename(columns={'filename_hr': 'filename'}, inplace=True)
    ptb_all['filename'] = ptb_all['filename'].apply(lambda x : "HR" + os.path.basename(x).replace("_hr", ".hea.npy"))
    
    # PTB FORM
    ptb_form = df_ptb_xl[['filename_hr', 'age', 'sex', 'strat_fold']]
    ptb_form[lbl_itos_ptb_xl['label_form'].tolist()] = 0
    for index, row in df_ptb_xl.iterrows():
        for label in row['label_form_filtered']:
            ptb_form.at[index, label] = 1
    ptb_form.rename(columns={'filename_hr': 'filename'}, inplace=True)
    ptb_form['filename'] = ptb_form['filename'].apply(lambda x : "HR" + os.path.basename(x).replace("_hr", ".hea.npy"))
    
    # PTB RHYTHM
    ptb_rhythm = df_ptb_xl[['filename_hr', 'age', 'sex', 'strat_fold']]
    ptb_rhythm[lbl_itos_ptb_xl['label_rhythm'].tolist()] = 0
    for index, row in df_ptb_xl.iterrows():
        for label in row['label_rhythm_filtered']:
            ptb_rhythm.at[index, label] = 1
    ptb_rhythm.rename(columns={'filename_hr': 'filename'}, inplace=True)
    ptb_rhythm['filename'] = ptb_rhythm['filename'].apply(lambda x : "HR" + os.path.basename(x).replace("_hr", ".hea.npy"))
    
    # PTB DIAG
    ptb_diag = df_ptb_xl[['filename_hr', 'age', 'sex', 'strat_fold']]
    ptb_diag[lbl_itos_ptb_xl['label_diag'].tolist()] = 0
    for index, row in df_ptb_xl.iterrows():
        for label in row['label_diag_filtered']:
            ptb_diag.at[index, label] = 1
    ptb_diag.rename(columns={'filename_hr': 'filename'}, inplace=True)
    ptb_diag['filename'] = ptb_diag['filename'].apply(lambda x : "HR" + os.path.basename(x).replace("_hr", ".hea.npy"))
    
    # PTB DIAG SUBCLASS
    ptb_diag_subclass = df_ptb_xl[['filename_hr', 'age', 'sex', 'strat_fold']]
    ptb_diag_subclass[lbl_itos_ptb_xl['label_diag_subclass'].tolist()] = 0
    for index, row in df_ptb_xl.iterrows():
        for label in row['label_diag_subclass_filtered']:
            ptb_diag_subclass.at[index, label] = 1
    ptb_diag_subclass.rename(columns={'filename_hr': 'filename'}, inplace=True)
    ptb_diag_subclass['filename'] = ptb_diag_subclass['filename'].apply(lambda x : "HR" + os.path.basename(x).replace("_hr", ".hea.npy"))
    
    # PTB DIAG SUPERCLASS
    ptb_diag_superclass = df_ptb_xl[['filename_hr', 'age', 'sex', 'strat_fold']]
    ptb_diag_superclass[lbl_itos_ptb_xl['label_diag_superclass'].tolist()] = 0
    for index, row in df_ptb_xl.iterrows():
        for label in row['label_diag_superclass_filtered']:
            ptb_diag_superclass.at[index, label] = 1
    ptb_diag_superclass.rename(columns={'filename_hr': 'filename'}, inplace=True)
    ptb_diag_superclass['filename'] = ptb_diag_superclass['filename'].apply(lambda x : "HR" + os.path.basename(x).replace("_hr", ".hea.npy"))
    
    # save ptbs in csv files
    # ptb_all.to_csv(os.path.join(data_path, "ptb_all.csv"), index=False)
    # ptb_form.to_csv(os.path.join(data_path, "ptb_form.csv"), index=False)
    # ptb_rhythm.to_csv(os.path.join(data_path, "ptb_rhythm.csv"), index=False)
    # ptb_diag.to_csv(os.path.join(data_path, "ptb_diag.csv"), index=False)
    # ptb_diag_subclass.to_csv(os.path.join(data_path, "ptb_diag_subclass.csv"), index=False)
    # ptb_diag_superclass.to_csv(os.path.join(data_path, "ptb_diag_superclass.csv"), index=False)
    
    
    return ptb_all, df_ptb_xl, lbl_itos_ptb_xl

def filter_ptb_xl(df,min_cnt=10,categories=["label_all","label_diag","label_form","label_rhythm","label_diag_subclass","label_diag_superclass"]):
    #filter labels
    def select_labels(labels, min_cnt=10):
        lbl, cnt = np.unique([item for sublist in list(labels) for item in sublist], return_counts=True)
        return list(lbl[np.where(cnt>=min_cnt)[0]])
    
    df_ptb_xl = df.copy()
    lbl_itos_ptb_xl = {}
    for selection in categories:
        label_selected = select_labels(df_ptb_xl[selection],min_cnt=min_cnt)
        df_ptb_xl[selection+"_filtered"]=df_ptb_xl[selection].apply(lambda x:[y for y in x if y in label_selected])
        lbl_itos_ptb_xl[selection] = np.array(list(set([x for sublist in df_ptb_xl[selection+"_filtered"] for x in sublist])))
        lbl_stoi = {s:i for i,s in enumerate(lbl_itos_ptb_xl[selection])}
        df_ptb_xl[selection+"_filtered_numeric"]=df_ptb_xl[selection+"_filtered"].apply(lambda x:[lbl_stoi[y] for y in x])
    return df_ptb_xl, lbl_itos_ptb_xl

def ptb_splits(ptb):
    '''Takes a PTB-XL dataframe and splits into training, validation and test sets based on strat_fold column.
    As suggested in the original paper,the tenth is for test and the nineth for evaluation'''
    train = ptb[ptb.strat_fold < 9]
    val = ptb[ptb.strat_fold == 9]
    test = ptb[ptb.strat_fold == 10]
    return train, val, test

def drop_nsr(df, perc):
    ''' Drop records in Normal Sinus Rhythm (NSR) class from Ribeiro dataset. To create Ribeiro-dev'''
    labels = df.columns.values[3:]
    mask = df[labels].eq(0).all(axis=1)
    to_drop = df[mask].sample(frac=perc).index
    df = df.drop(to_drop)
    return df

def simple_split(df, test_size, label_start_index = 3, random_state=None, n_splits=1):
    info = df.columns.values[:label_start_index]
    labels = df.columns.values[label_start_index:]
    
    X = df[info]
    y = df[labels]
    
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    trains = []
    tests = []
    
    for train_index, test_index in msss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        train = pd.concat([X_train, y_train], axis=1)      
        test = pd.concat([X_test, y_test], axis=1)
        
        trains.append(train)
        tests.append(test)
        
    if n_splits == 1:
        return trains[0], tests[0]
    else:
        return trains, tests

def label_distribution(df, label_start=3, return_counts=False):
    labels = df.columns.values[label_start:]
    dist = df[labels].sum(axis=0)
    if return_counts:
        return dist
    else:
        return dist / len(df)

def apply_filter(signal, filter_bandwidth, fs=500):
    ''' Bandpass filtering to remove noise, artifacts etc '''
    # Calculate filter order
    order = int(0.3 * fs)
    # Filter signal
    signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                order=order, frequency=filter_bandwidth, 
                                sampling_rate=fs)
    return signal

def scaling(seq, smooth=1e-8):
    return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1

def ecg_preprocessing(ecg_signal, original_frequency, target_frequency=100, band_pass=[0.05, 47]):

    assert ecg_signal.shape[0] == 12, "ecg_signal should have (12, signal_length) shape for pre-processing"

    ecg_signal = resample(ecg_signal, int(ecg_signal.shape[-1] * (500/original_frequency)), axis=1) 
    # 500 hz is the highest and most common sampling rate found in literature and respects Shannon theorem, as max spectral component is said to be 150 hz

    ecg_signal = apply_filter(ecg_signal, band_pass) # this band focuses on dominant component of ecg waves

    return scaling(ecg_signal)
