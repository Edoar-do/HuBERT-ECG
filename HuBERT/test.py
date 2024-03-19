from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from dataset import ECGDataset
from tqdm import tqdm
from loguru import logger
import argparse
import numpy as np
from torchmetrics.classification import MultilabelF1Score as F1_score
from torchmetrics.classification import MultilabelRecall as Recall
from torchmetrics.classification import MultilabelPrecision as Precision
from torchmetrics.classification import MultilabelSpecificity as Specificity
from torchmetrics.classification import MultilabelAUROC as AUROC
from torcheval.metrics import MultilabelAUPRC as AUPRC
from typing import List
from hubert_ecg import HuBERTECG as HuBERT
from hubert_ecg_classification import HuBERTForECGClassification as HuBERTClassification

def test(args, model : nn.Module, criterion : nn.functional, metrics : List[nn.Module]):
    
    #fixing seed
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    testset = ECGDataset(
        path_to_dataset_csv = args.path_to_dataset_csv,
        ecg_dir_path = args.ecg_dir_path,
        pretrain = False,
        leads_as_channels=hubert.hubert_ecg.feature_extractor.leads_as_channels,
        downsampling_factor=args.downsampling_factor
    )
    
    dataloader = DataLoader(
        testset,
        collate_fn=testset.collate,
        num_workers=5,
        batch_size = args.batch_size,
        shuffle=False,
        sampler=None,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info("Test set and dataloader have been created")
    
    model = model.to(device)
    for metric in metrics:
        metric.to(device)
    
    logger.info("Start testing...")
    
    model.eval()
    
    losses = []
    
    for ecg, attention_mask, labels in tqdm(dataloader, total=len(dataloader)):
        
        ecg = ecg.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            out = model(ecg, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True)
            if isinstance(model, HuBERT):
                logits = model.logits(out['last_hidden_state']).traspose(1, 2)
            elif isinstance(model, HuBERTClassification):
                logits = out[0]
                
        loss = criterion(logits, labels).item()
        losses.append(loss)
        
        logger.info(f"Criterion {criterion} score: {loss}")
        
        labels = labels.long() # because of torchmetrics
        for i, metric in enumerate(metrics):
            metric.update(logits, labels) # compute metric per batch
    
            
    logger.info("Loss = {}".format(np.mean(losses)))
    
    for metric in metrics:
        score = metric.compute() # compute metric over all test set
        logger.info(f"{metric} = {score}, macro: {score.mean()}")
        
            
    logger.info("End of testing.")
    
if __name__ == "__main__":
    
    ### PARSING ARGUMENTS ###
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("path_to_dataset_csv", type=str, help="Path to dataset csv file")
    
    parser.add_argument("ecg_dir_path", type=str, help="Path to directory with ecg files")
    
    parser.add_argument("pretrained_model_path", type=str, help="Path to pretrained model")
    
    parser.add_argument("finetuned_model_path", type=str, help="Path to finetuned model")
    
    parser.add_argument("batch_size", type=int, help="Batch size (Should be the same used during finetuning)")
    
    parser.add_argument("--downsampling_factor", type=int, help="Downsampling factor")
    
    args = parser.parse_args()
    
    
    ### LOADING MODEL TO TEST ###
    cpu_device = torch.device('cpu')
    checkpoint_pretrained = torch.load(args.pretrained_model_path, map_location = cpu_device)
    checkpoint_finetune = torch.load(args.finetuned_model_path, map_location = cpu_device)

    config = checkpoint_pretrained['model_config']
    config.vocab_size = checkpoint_pretrained['pretraining_vocab_size']
    pretrained_hubert = HuBERT(config, leads_as_channels=False) # checkpoint['leads_as_channels']
    pretrained_hubert.load_state_dict(checkpoint_pretrained['model_state_dict'])
    
    logger.info("Pretrained model loaded.")
    
    keys = list(checkpoint_finetune['model_state_dict'].keys())    
    num_labels = checkpoint_finetune['model_state_dict'][keys[-1]].size(-1) #todo: checkpoint_finetune['finetuning_vocab_size'], needs adding as key during finetuning save()
    
    if checkpoint_finetune['model_state_dict'][keys[-2]].size(-1) == pretrained_hubert.config.classifier_proj_size: # checkpoint_finetune['linear']
        classifier_hidden_size = None
    else:
        classifier_hidden_size = checkpoint_finetune['model_state_dict'][keys[-2]].size(-1)
    hubert = HuBERTClassification(pretrained_hubert, num_labels=num_labels, classifier_hidden_size=classifier_hidden_size)
    hubert.load_state_dict(checkpoint_finetune['model_state_dict'])
    
    logger.info("Finetuned model loaded.")
    
    ### START TESTING ###
    
    test(args, hubert, nn.functional.binary_cross_entropy_with_logits,
         [F1_score(num_labels=num_labels, average=None), Recall(num_labels=num_labels, average=None),
          Precision(num_labels=num_labels, average=None), Specificity(num_labels=num_labels, average=None),
          AUROC(num_labels=num_labels, average=None), AUPRC(num_labels=num_labels, average=None)])  # add cinc2020 metric
