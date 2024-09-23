from re import S
from matplotlib.style import use
import torch
from torch.nn import functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from loguru import logger
import argparse
from tqdm import tqdm
from hubert_ecg import HuBERTECG as HuBERT
from transformers import HubertConfig, get_linear_schedule_with_warmup
import torch.optim as optim
import wandb
import numpy as np
import copy
from hubert_ecg_classification import HuBERTForECGClassification as HuBERTClassification
import os
from math import ceil
from dataset import ECGDataset
from torchmetrics.classification import MultilabelF1Score as F1_score
from torchmetrics.classification import MultilabelRecall as Recall
from torchmetrics.classification import MultilabelPrecision as Precision
from torchmetrics.classification import MultilabelSpecificity as Specificity
from torchmetrics.classification import MultilabelAUROC as AUROC
from torcheval.metrics import MultilabelAUPRC as AUPRC
from utils import cinc2020_metric
import pandas as pd
import random

EPS = 1e-9
MINIMAL_IMPROVEMENT = 1e-3
SUPERVISED_MODEL_CKPT_PATH = "/path/to/folder/for/finetuned/models"
DROPOUT_DYNAMIC_REG_FACTOR = 0.05
    

def dynamic_regularizer(optimizer, model, penalty):
    if penalty:
        # penalizing model with regularization but not too much
        optimizer.param_groups[0]['weight_decay'] *= 5
        for name, module in model.named_modules():
            if 'dropout' in name:
                module.p += 0.05
    else:
        # unburdening model from regularization
        # minimum attainable weight decay is 0.01, dropout is 0.1
        optimizer.param_groups[0]['weight_decay'] = max(0.01, optimizer.param_groups[0]['weight_decay'] / 5)
        for name, module in model.named_modules():
            if 'dropout' in name:
                module.p = max(0.1, module.p - DROPOUT_DYNAMIC_REG_FACTOR)

def finetune(args):
    
    device = torch.device('cuda')
    
    ### NOTE: comment for sweeps, uncomment for normal run ###
    wandb.init(entity="your-entity", project="your_project", group="your-group")
    
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    
    lr = args.lr
    betas = (0.9, 0.98)
    weight_decay = 0.01 * args.weight_decay_mult
    accumulation_steps = args.accumulation_steps
    
    if args.resume_finetune:
        
        logger.info(f"Loading pretraining checkpoint {args.load_path.split('/')[-1]} to resume finetuning")
        
        checkpoint = torch.load(args.load_path, map_location = 'cpu')
        config = checkpoint['model_config']
        vocab_sizes = checkpoint['pretraining_vocab_sizes']
        leads_as_channels = False 
        pretrained_hubert = HuBERT(config, leads_as_channels=leads_as_channels, ensamble_length=len(vocab_sizes), vocab_sizes = vocab_sizes)
        pretrained_hubert.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        hubert = HuBERTClassification(pretrained_hubert, num_labels=args.vocab_size, classifier_hidden_size=args.classifier_hidden_size, use_label_embedding=args.use_label_embedding)
        hubert.to(device)
        
        logger.info(f"Loading checkpoint {args.load_path_finetune.split('/')[-1]} to resume finetuning")
        
        checkpoint = torch.load(args.load_path_finetune, map_location = 'cpu')
        
        hubert.load_state_dict(checkpoint['model_state_dict'])
        
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint['best_val_loss']
        patience_count = checkpoint['patience_count']
        best_val_target_score = checkpoint[f'target_val_{args.target_metric}']
        
        if args.freezing_steps is not None and global_step < args.freezing_steps:
            hubert.set_transformer_blocks_trainable(n_blocks=0)
            hubert.set_feature_extractor_trainable(False)
        else:
            hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze)
            hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder)
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, hubert.parameters()),
            lr=lr,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(args.ramp_up_perc*(args.training_steps-global_step)),
            num_training_steps=args.training_steps,
        )
        
        logger.info("Checkpoint loaded. Model ready to resume finetuning.")
        
    elif args.random_init:
        logger.info("Creating a model for fully supervised training")
        
        ### size model hyperparams ###
        if args.largeness == "base":
            hidden_size = 768
            num_hidden_layers = 12
            num_attention_heads = 12
            intermediate_size = 3072    
            classifier_proj_size = 256
            layerdrop = 0.1
        elif args.largeness == "large":
            hidden_size = 1024
            num_hidden_layers = 24
            num_attention_heads = 16
            intermediate_size = 4096
            classifier_proj_size = 768
            layerdrop = 0.0
        else: # x-large
            hidden_size = 1280
            num_hidden_layers = 48
            num_attention_heads = 16
            intermediate_size = 5120
            classifier_proj_size = 1024
            layerdrop = 0.0
        
        if args.downsampling_factor is None:
            conv_kernel = (10, 3, 3, 3, 3, 2, 2)
            conv_stride = (5, 2, 2, 2, 2, 2, 2)
            conv_dim = (512, 512, 512, 512, 512, 512, 512)   
        elif args.downsampling_factor == 5: 
            conv_kernel = (10, 3, 3, 2, 2)
            conv_stride = (4, 2, 2, 2, 2)
            conv_dim = (512, 512, 512, 512, 512)
        elif args.downsampling_factor == 10:
            conv_kernel = (10, 3, 3, 2)
            conv_stride = (4, 2, 2, 2)
            conv_dim = (512, 512, 512, 512)
        else:
            raise ValueError(f"Downsampling factor {args.downsampling_factor} not supported")
                
        config = HubertConfig(
            hidden_size = hidden_size,
            num_hidden_layers = num_hidden_layers,
            num_attention_heads = num_attention_heads,
            intermediate_size = intermediate_size,
            mask_time_prob = 0.0, 
            classifier_proj_size = classifier_proj_size,
            layerdrop = layerdrop,
            conv_kernel = conv_kernel,
            conv_stride = conv_stride,
            conv_dim = conv_dim,
            mask_time_length = 1,
            hidden_dropout=0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult,
            activation_dropout=0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult,
            attention_dropout=0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult,
            feat_proj_dropout=0 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult,
            final_dropout=0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult,    
        ) # + other default params
        
        hubert = HuBERT(config, ensamble_length = len(args.vocab_sizes), vocab_sizes = args.vocab_sizes)
        
        hubert = HuBERTClassification(hubert, num_labels=args.vocab_size, classifier_hidden_size=args.classifier_hidden_size, use_label_embedding=args.use_label_embedding)
        hubert.to(device)  
        
        
        global_step = 0
        best_val_loss = float("inf")
        best_val_target_score = 0.0
        patience_count = 0  
              
        optimizer = optim.AdamW(
            hubert.parameters(),
            lr=lr,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(0.08*args.training_steps),
            num_training_steps=args.training_steps
        )
        
        logger.info("Model created. Ready for fully supervised training.")
        
    else:
        logger.info(f"Loading pretraining checkpoint {args.load_path.split('/')[-1]} to start finetuning")
                
        checkpoint = torch.load(args.load_path, map_location = 'cpu')
        config = checkpoint['model_config']
        #config.vocab_size = checkpoint['model_state_dict'][list(checkpoint['model_state_dict'].keys())[-1]].size(0) # vocab size used for pretraining
        #config.vocab_size = checkpoint["pretraining_vocab_size"]
        leads_as_channels = False # checkpoint['leads_as_channels']
        vocab_sizes = checkpoint['pretraining_vocab_sizes'] # vocab_sizeS
        pretrained_hubert = HuBERT(config, leads_as_channels=leads_as_channels, ensamble_length=1 if type(vocab_sizes) == int else len(vocab_sizes), vocab_sizes=[vocab_sizes] if type(vocab_sizes) != list else vocab_sizes)
        pretrained_hubert.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # restore original p-dropout or set multipliers
        for name, module in pretrained_hubert.named_modules():
            if 'dropout' in name:
                module.p = 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult
        
        hubert = HuBERTClassification(
            pretrained_hubert, 
            num_labels=args.vocab_size,
            classifier_hidden_size=args.classifier_hidden_size,
            use_label_embedding=args.use_label_embedding
            )
        
        hubert.to(device)            
        
        global_step = 0
        best_val_loss = float('inf')
        patience_count = 0
        best_val_target_score = 0.0        
        
        # ASSUMPTION: the classification head is always trainable
        # transformer_blocks_to_unfreeze is 0 by default meaning that if no input comes from the user, the encoder remains frozen
        # if the user wants to unfreeze some blocks, he must provide the number of blocks to unfreeze
        # whether freezing the convolutional feature extractor is a precise choice of the user. By default it is frozen
        # if a user wants to unfreeze n_blocks of the transformer encoder and/or the convolutional feature extractor,
        # he must provide the number of blocks to unfreeze and set the flag to unfreeze the conv feature extractor as well as freezing_steps argument
        # if the freezing_steps argument is provided, then the model is completely frozen to allow to train only the head for that number of steps
        # then n_blocks are unfronzen and the conv feature extractor is unfrozen as well in the training loop
        # freezing_steps is none by default
        # note: if freezing_steps equals training_steps, then the model is completely frozen for the whole training
        
        if args.freezing_steps is not None:
            hubert.set_transformer_blocks_trainable(n_blocks=0) # freeze all transformer blocks
            hubert.set_feature_extractor_trainable(False) # freeze conv feature extractor
        else:        
            hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze) # makes trainable only the last n_blocks of the transformer encoder
            hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder) # frozen by default
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, hubert.parameters()),
            lr=lr,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(args.ramp_up_perc * args.training_steps),
            num_training_steps=args.training_steps,
        )
        
        logger.info("Checkpoint loaded. Model ready for finetuning.")
    
    scaler = amp.GradScaler()
    
    train_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_train,
        ecg_dir_path="/data/ECG_AF/train_self_supervised",
        label_start_index=args.label_start_index,
        downsampling_factor=args.downsampling_factor,
        pretrain=False
    )

    train_pos_weights = train_set.weights.to(device) if args.use_loss_weights else None

    val_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_val,
        ecg_dir_path="/data/ECG_AF/val_self_supervised",
        label_start_index=args.label_start_index,
        downsampling_factor=args.downsampling_factor,
        pretrain=False
        )
    
    val_pos_weights = val_set.weights.to(device) if args.use_loss_weights else None
    
    train_dl = DataLoader(
        train_set,
        collate_fn=train_set.collate,
        num_workers=6,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
        )

    val_dl = DataLoader(
        val_set,
        collate_fn=val_set.collate,
        num_workers=6,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
        )

    epochs = args.training_steps // (len(train_dl) // accumulation_steps) + 1 if args.training_steps is not None else args.epochs
    args.val_interval = len(train_dl) if args.val_interval is None else args.val_interval

    start_epoch = global_step // len(train_dl)
    
    metrics = {
        "f1-score" : F1_score(num_labels=args.vocab_size, average=None),
        "recall" : Recall(num_labels=args.vocab_size, average=None),
        "specificity" : Specificity(num_labels=args.vocab_size, average=None),
        "precision" : Precision(num_labels=args.vocab_size, average=None),
        "auroc" : AUROC(num_labels=args.vocab_size, average=None),
        "auprc" : AUPRC(num_labels=args.vocab_size, average=None)
        }
    
    for name, metric in metrics.items():
        metric.to(device)
    
    for epoch in range(start_epoch, epochs):
    
        hubert.train()
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        train_losses = []
        
        for ecg, attention_mask, labels in tqdm(train_dl, total=len(train_dl)):
            
            global_step += 1
            
            ecg = ecg.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            with amp.autocast():
                logits, _ = hubert(ecg, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False, return_dict=True)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=train_pos_weights)
                
                loss /= accumulation_steps # normalize loss
                
            scaler.scale(loss).backward() # accumulate normalized loss
            train_losses.append(loss.item())                
                
            if global_step % accumulation_steps == 0: 
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()
                
            if args.freezing_steps is not None and global_step >= args.freezing_steps:
                
                # unfreeze what you wanted to unfreeze, as specificied by user input
                hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze)
                hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder)
                
                # recreate optimizer to update params when unfreezing them
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, hubert.parameters()),
                    lr=lr,
                    betas=betas,
                    eps=EPS,
                    weight_decay=weight_decay,
                )
                # recreate lr_scheduler to update params when unfreezing them
                lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=ceil(args.ramp_up_perc*(args.training_steps-global_step)),
                    num_training_steps=args.training_steps,
                )
            
            ### validation ###        
            if global_step % args.val_interval == 0:
                
                hubert.eval()
                
                # reset metrics and losses                
                val_losses = []                
                for name, metric in metrics.items():
                    metric.reset()
                val_cinc2020_scores = [] if args.target_metric == "cinc2020" else None
                
                logger.info("Start validation at step {}".format(global_step))
                
                ### validation loop ###
                for ecg, attention_mask, labels in tqdm(val_dl, total=len(val_dl)):
                    
                    ecg = ecg.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    
                    with torch.no_grad():
                        logits, _ = hubert(ecg, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True)
                        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=val_pos_weights)
                    
                    val_losses.append(loss.item())
                    
                    labels = labels.long() # for metrics
                    
                    # compute metrics on single batch
                    for name, metric in metrics.items():
                        metric.update(logits, labels)

                    if args.target_metric == "cinc2020":
                        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                        labels = labels.cpu().numpy()
                        labels = pd.DataFrame(labels, columns=val_set.diagnoses_cols)
                        val_cinc2020_scores.append(cinc2020_metric(labels, preds))
                        
                    
                ### end of validation loop ###
                
                val_loss = np.mean(val_losses)
                train_loss = np.mean(train_losses)
                train_losses.clear() # to keep train loss aligned with val loss
                
                val_cinc2020_score = np.mean(val_cinc2020_scores) if args.target_metric == "cinc2020" else None # already macro
                                
                # log non averaged metrics [1, vocab_size]
                logger.info("Validation loss = {}".format(val_loss))
                if args.target_metric == "cinc2020":
                    logger.info("Validation CINC2020 score = {}".format(val_cinc2020_score))
                    wandb.log({
                        "Validation_CINC2020_score": val_cinc2020_score
                        })
                    
                # compute metrics on whole validation set and log them
                # such metrics are vectors num_labels long containing the metric for each label
                macros = []
                for name, metric in metrics.items():
                    score = metric.compute()
                    macro = score.mean()
                    macros.append(macro)
                    logger.info(f"Validation {name} = {score}, macro: {macro}")
                    if name == args.target_metric:
                        target_score = macro

                dict_to_log = {f"Validation_{name}" : macro for name, macro in zip(metrics.keys(), macros)}
                dict_to_log["Training_loss"] = train_loss
                dict_to_log["Validation_loss"] = val_loss
                wandb.log(dict_to_log)
                
                if args.target_metric == "cinc2020":
                    target_score = val_cinc2020_score
                    
                hubert.train()
                
                ### save if there's improvement in loss or target metric ###
                
                if val_loss <= best_val_loss - MINIMAL_IMPROVEMENT: 
                    
                    best_val_loss = val_loss
                    patience_count = 0
                    checkpoint = {
                        'global_step': global_step,
                        'best_val_loss': best_val_loss,
                        'model_config': hubert.config,
                        'model_state_dict': copy.deepcopy(hubert.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'lr_scheduler_state_dict': copy.deepcopy(lr_scheduler.state_dict()),
                        'patience_count': patience_count,
                        'linear' : True if args.classifier_hidden_size is None and not args.use_label_embedding else False,
                        'leads_as_channels' : leads_as_channels,
                        'finetuning_vocab_size' : args.vocab_size,
                        'use_label_embedding' : args.use_label_embedding,
                        f'target_val_{args.target_metric}': target_score
                    }
                    
                    checkpoint_name = f"hubert_{args.train_iteration}_iteration_{global_step}_finetuned_{wandb.run.id}.pt"
                    
                    torch.save(checkpoint, os.path.join(SUPERVISED_MODEL_CKPT_PATH, args.sweep_dir, checkpoint_name))
                    
                    logger.info(f"New best val loss = {best_val_loss}. Checkpoint saved at step {global_step}")
                    
                    dynamic_regularizer(optimizer=optimizer, model=hubert, penalty=False) if args.dynamic_reg else None
                                
                elif target_score >= best_val_target_score + MINIMAL_IMPROVEMENT:
                    
                    best_val_target_score = target_score
                    
                    # do not reset patience count
                    
                    checkpoint = {
                        'global_step': global_step,
                        'best_val_loss': best_val_loss,
                        'model_config': hubert.config,
                        'model_state_dict': copy.deepcopy(hubert.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'lr_scheduler_state_dict': copy.deepcopy(lr_scheduler.state_dict()),
                        'patience_count': patience_count,
                        'linear' : True if args.classifier_hidden_size is None and not args.use_label_embedding else False,
                        'leads_as_channels' : leads_as_channels,
                        'finetuning_vocab_size' : args.vocab_size,
                        'use_label_embedding' : args.use_label_embedding,
                        f'target_val_{args.target_metric}': target_score
                    }
                    
                    checkpoint_name = f"hubert_{args.train_iteration}_iteration_{global_step}_finetuned_{wandb.run.id}.pt"
                    
                    torch.save(checkpoint, os.path.join(SUPERVISED_MODEL_CKPT_PATH, args.sweep_dir, checkpoint_name))
                    
                    logger.info(f"Val loss not improved but {args.target_metric} did (= {target_score}). Checkpoint saved at step {global_step}")
                    
                    dynamic_regularizer(optimizer=optimizer, model=hubert, penalty=False) if args.dynamic_reg else None
                    
                else: # loss not improved and target metric not improved
                    patience_count += 1
                    
                    if args.dynamic_reg and patience_count % (args.patience // args.intervals_for_penalty) == 0 and patience_count != args.patience:
                        dynamic_regularizer(optimizer=optimizer, model=hubert, penalty=True) # penalize model with regularization but not too much
                                
                    if patience_count == args.patience:
                        logger.warning(f"Early stopping at step {global_step}.")
                        wandb.log({
                            "patience_count": patience_count
                            })
                        return
                
    logger.info("End of finetuning.")
    logger.info(f"Best val loss = {best_val_loss}")
    logger.info(f"Best val target score = {best_val_target_score}, ({args.target_metric})")
                        
                        
if __name__ == "__main__":
    
    ### NOTE: comment for sweeps, uncomment for normal run ###

    # parser = argparse.ArgumentParser(description="Train Hubert-ECG")
    
    # #train_iteration
    # parser.add_argument(
    #     "train_iteration",
    #     help="Hubert training iteration in {1, 2, 3}", 
    #     type=int,
    #     choices=[1, 2, 3]
    #     )
    
    # #path_to_dataset_csv_train
    # parser.add_argument(
    #     "path_to_dataset_csv_train",
    #     help="Path to the csv file containing the training dataset",
    #     type=str
    # )
    
    # #path_to_dataset_csv_val
    # parser.add_argument(
    #     "path_to_dataset_csv_val",
    #     help="Path to the csv file containing the validation dataset",
    #     type=str
    # )
    
    # #sweep_dir
    # parser.add_argument(
    #     "--sweep_dir",
    #     help="Sweep directory. Default `.`",
    #     type=str,
    #     default="."
    # )
    
    # #ramp_up_perc
    # parser.add_argument(
    #     "--ramp_up_perc",
    #     help="[OPT.] Percentage of training steps for the ramp up phase. Default 0.08",
    #     type=float,
    #     default=0.08
    # )
    
    # #training_steps
    # parser.add_argument(
    #     "--training_steps",
    #     help="Number of training steps to perform",
    #     type=int
    # )
    
    # #epochs
    # parser.add_argument(
    #     "--epochs",
    #     help="[OPT.] Number of epochs to perform",
    #     type=int
    # )
    
    # #vocab_size
    # parser.add_argument(
    #     "vocab_size",
    #     help="Vocabulary size, i.e. num of labels/clusters",
    #     type=int
    # )
    
    # #val_interval
    # parser.add_argument(
    #     "--val_interval",
    #     help="Number of training steps to wait before validating the in-training model. Defatul len(dataloader), i.e. at each epoch",
    #     type=int,
    #     default=None
    # )
    
    # #patience
    # parser.add_argument(
    #     "patience",
    #     help="Patience for early stopping",
    #     type=int
    # )
    
    # #batch_size
    # parser.add_argument(
    #     "batch_size",
    #     help="Batch_size",
    #     type=int
    # )
    
    # #downsampling_factor
    # parser.add_argument(
    #     "--downsampling_factor",
    #     help="[OPT] Downsampling factor to apply to the ECG signal",
    #     type=int
    # )
    
    # #target_metric
    # parser.add_argument(
    #     "target_metric",
    #     type=str,
    #     help="Target metric (macro) to optimize during finetuning",
    #     choices=["f1_score", "recall", "precision", "specificity", "auroc", "auprc", "cinc2020"]
    # )
    
    # #accumulation_steps
    # parser.add_argument(
    #     "--accumulation_steps", 
    #     help="[OPT] Number of batch gradients to accumulate before updating model params. Default 1",
    #     type=int,
    #     default=1
    # )
    
    # #label_start_index
    # parser.add_argument(
    #     "--label_start_index",
    #     help="[OPT] Index of the first label in the dataset csv file. Default 3",
    #     type=int,
    #     default=3
    # )
    
    # #freezing_steps
    # parser.add_argument(
    #     "--freezing_steps",
    #     help="[OPT] Number of finetuning steps to keep frozen the base model weights",
    #     type=int,
    #     default=None
    # )  
    
    # #resume_finetune
    # parser.add_argument(
    #     "--resume_finetune",
    #     help="Whether to resume finetuning",
    #     action="store_true",
    #     default=False
    # )
    
    # #unfreeze_conv_embedder
    # parser.add_argument(
    #     "--unfreeze_conv_embedder",
    #     help="[OPT] Whether to unfreeze the convolutional feature extractor during fine-tuning. Normally frozen",
    #     action="store_true",
    #     default=False
    # )
    
    # #transformer_blocks_to_unfreeze
    # parser.add_argument(
    #     "--transformer_blocks_to_unfreeze",
    #     help="[OPT] Number of transformer blocks to unfreeze after freezing_steps. Default None",
    #     type=int,
    #     default=0
    # )
    
    # #lr
    # parser.add_argument(
    #     "--lr",
    #     help="[OPT] Learning rate. Default 1e-5",
    #     type=float,
    #     default=1e-5
    # )
    
    # #load_path
    # parser.add_argument(
    #     "load_path",
    #     help="Path to load a partially/completely pretrained model from in order to resume pretrainig, start finetuning or resume finetuning",
    #     type=str
    # )
    
    # #load_path_finetune
    # parser.add_argument(
    #     "--load_path_finetune",
    #     help="[OPT] Path to load a partially finetuned model from in order to resume finetuning",
    #     type=str
    # )
    
    # #classifier_hidden_size
    # parser.add_argument(
    #     "--classifier_hidden_size",
    #     help="[OPT.] Hidden size of the MLP head used for classification in finetuning. If None, then linear classifier. Default None",
    #     type=int,
    #     default=None
    # )
    
    # #use_label_embedding
    # parser.add_argument(
    #     "--use_label_embedding",
    #     help="[OPT.] Whether to use label embeddings in the classification head. Default False",
    #     action="store_true"
    # )

    # #intervals_for_penalty
    # parser.add_argument(
    #     "--intervals_for_penalty",
    #     help="['OPT.] Number of validation intervals with worsening performance to wait before applying penalizing regularization",
    #     type=int,
    #     default=3
    # )
    
    # #dynamic_reg
    # parser.add_argument(
    #     "--dynamic_reg",
    #     help="[OPT.] Whether to apply dynamic regularization to the model. Default False",
    #     action="store_true"
    # )
    
    # #use_loss_weights
    # parser.add_argument(
    #     "--use_loss_weights",
    #     help="[OPT.] Whether to use loss weights in the loss function. Default False",
    #     action="store_true"
    # )
    
    # #random_init
    # parser.add_argument(
    #     "--random_init",
    #     help="[OPT.] Whether to initialize the model with random weights. Default False",
    #     action="store_true"
    # ) 
    
    # #largeness
    # parser.add_argument(
    #     "--largeness",
    #     help="[OPT.] Model largeness in {base, large, x-large} in case of random initialization. Default base",
    #     type=str,
    #     choices=["base", "large", "x-large"]
    # )
    
    # # weight_decay_mult
    # parser.add_argument(
    #     "--weight_decay_mult",
    #     help="Weight decay. Default 1",
    #     type=int,
    #     default=1
    # )
    
    # # model_dropout_mult
    # parser.add_argument(
    #     "--model_dropout_mult",
    #     help="Model dropout. Default 0",
    #     type=int,
    #     default=0
    # )   
     
    
    # args = parser.parse_args()
    
    # if not torch.cuda.is_available():
    #     logger.error("CUDA not available. CPU finetuning not supported")
    #     exit(1)
        
    # if args.train_iteration > 3 or args.train_iteration < 1:
    #     raise ValueError(f"Argument train_iteration must be an integer in [1, 3] range. Inserted {args.train_iteration}")
    
    # if args.training_steps is None and args.epochs is None:
    #     raise ValueError("Argument training_steps or epochs must be provided")
        
    # if args.training_steps is not None and args.training_steps % args.val_interval != 0:
    #     raise ValueError(f"Argument training_steps must be divisible by argument val_interval. Inserted {args.training_steps} and {args.val_interval}")
    
    # if args.ramp_up_perc < 0 or args.ramp_up_perc > 1:
    #     raise ValueError("Argument ramp_up_perc must be a float in [0, 1] range")
    
    # if args.resume_finetune and (args.load_path is None or args.load_path_finetune is None):
    #     raise ValueError("Arguments load_path and load_path_finetune must be provided if argument resume_finetune is provided")
    
    # if args.freezing_steps is not None and args.freezing_steps > args.training_steps:
    #     raise ValueError("Argument freezing_steps cannot be greater than argument training steps")
    
    # if args.accumulation_steps is not None and args.training_steps % args.accumulation_steps != 0:
    #     raise ValueError("Argument training_steps must be divisible by argument accumulation_steps")
    
    # if args.label_start_index is not None and (args.label_start_index < 0 or args.label_start_index >= args.vocab_size):
    #     raise ValueError("Argument label_start_index must be a positive integer between 0 and vocab_size-1")
    
    # if args.random_init and args.largeness is None:
    #     raise ValueError("Argument largeness must be provided if argument random_init is provided")
    
    # if not args.resume_finetune and args.load_path_finetune is not None:
    #     raise ValueError("Argument load_path_finetune can't be inserted if argument resume_finetune is False")
    
    # print("Inserted arguments: ")
    # for arg in vars(args):
    #     print(f"{arg} = {getattr(args, arg)}")
        
    
    ### NOTE: this is to test sweeps ###

    wandb.init(entity="cardi-ai", project="ECG-pretraining", group=("supervised"))
    
    args = wandb.config

    finetune(args)  
                
