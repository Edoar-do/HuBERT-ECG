import torch
from torch.nn import functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from loguru import logger
import argparse
from tqdm import tqdm
from hubert_ecg import HuBERTECG as HuBERT, HuBERTECGConfig
from transformers import get_linear_schedule_with_warmup
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
from torchmetrics.classification import MultilabelAUROC
from torchmetrics.classification import MulticlassAccuracy as Accuracy
from torchmetrics.classification import MulticlassAUROC
from torcheval.metrics import MultilabelAUPRC as AUPRC
import random
from transformers import HubertConfig

EPS = 1e-9
MINIMAL_IMPROVEMENT = 1e-3
SUPERVISED_MODEL_CKPT_PATH = "/models/checkpoints/supervised/"
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
    wandb.init(entity="my-entity", project="my-project", group="supervised")

    if args.wandb_run_name is not None:
        wandb.run.name = args.wandb_run_name
    
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    
    train_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_train,
        ecg_dir_path="/data/ECG_AF/train_self_supervised",
        label_start_index=args.label_start_index,
        downsampling_factor=args.downsampling_factor,
        pretrain=False,
        random_crop=args.random_crop
    )

    train_pos_weights = train_set.weights.to(device) if args.use_loss_weights else None

    val_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_val,
        ecg_dir_path="/data/ECG_AF/val_self_supervised",
        label_start_index=args.label_start_index,
        downsampling_factor=args.downsampling_factor,
        pretrain=False,
        random_crop=args.random_crop
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
    
    lr = args.lr
    betas = (0.9, 0.98)
    weight_decay = max(0, 0.01 * args.weight_decay_mult)
    accumulation_steps = args.accumulation_steps
    
    task2criteria = {
        'multi_class': (torch.nn.CrossEntropyLoss(weight=train_pos_weights), torch.nn.CrossEntropyLoss(weight=val_pos_weights)),
        'multi_label': (torch.nn.BCEWithLogitsLoss(pos_weight=train_pos_weights), torch.nn.BCEWithLogitsLoss(pos_weight=val_pos_weights)),
        'regression' : (torch.nn.MSELoss(), torch.nn.MSELoss())
    }    
    
    criterion = task2criteria[args.task]
    criterion_train = criterion[0].to(device)
    criterion_val = criterion[1].to(device)
    
    args.training_steps = args.training_steps if args.training_steps is not None else ((args.epochs - 1) * (len(train_dl) // accumulation_steps))
    
    args.val_interval = len(train_dl) if args.val_interval is None else args.val_interval
    
    logger.info(f"{args.training_steps} training steps to perform")
    logger.info(f"{args.val_interval} steps to wait before validation")
    
    if args.resume_finetuning:
        
        logger.info(f"Loading pretraining checkpoint {args.load_path.split('/')[-1]} to resume finetuning")
        
        checkpoint = torch.load(args.load_path, map_location = 'cpu')

        config = checkpoint['model_config']

        if type(config) == HubertConfig:
            config = HuBERTECGConfig(**config.to_dict())
            
        pretrained_hubert = HuBERT(config)
        
        assert checkpoint['finetuning_vocab_size'] == args.vocab_size, "Vocab size mismatch"
        assert checkpoint['use_label_embedding'] == args.use_label_embedding, "Label embedding mismatch"
        assert checkpoint['linear'] == True if args.classifier_hidden_size is None and not args.use_label_embedding else False, "Classifier mismatch"
        
        hubert = HuBERTClassification(pretrained_hubert, num_labels=args.finetuning_vocab_size, classifier_hidden_size=args.classifier_hidden_size, use_label_embedding=args.use_label_embedding)
        hubert.to(device)
        
        hubert.load_state_dict(checkpoint['model_state_dict'], strict=False) # strict false prevents errors when trying to match mask token key
        
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
            
        # if layuer_wise_lr, then set a higher lr for deeper transformer layers + head than that of the rest of the trainable body of the model
        parameters_group = []    
        if args.layer_wise_lr and all(p.requires_grad for p in hubert.hubert_ecg.encoder.layers.parameters()):
            parameters_group.append({"params": hubert.hubert_ecg.feature_projection.parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[:args.transformer_blocks_to_unfreeze-4].parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[args.transformer_blocks_to_unfreeze-4:].parameters(), "lr": lr})
            parameters_group.append({"params": hubert.classifier.parameters(), "lr": 1e-5})
        else:
            parameters_group.append({"params" : filter(lambda p : p.requires_grad, hubert.parameters()), "lr": lr})
        
        optimizer = optim.AdamW(
            parameters_group,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(args.ramp_up_perc*(args.training_steps-global_step)),
            num_training_steps=args.training_steps,
        )

        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
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
        elif args.largeness == "large":
            hidden_size = 960
            num_hidden_layers = 16
            num_attention_heads = 12
            intermediate_size = 3840
            classifier_proj_size = 512
        elif args.largeness == 'small': # small
            hidden_size = 512
            num_hidden_layers = 8
            num_attention_heads = 8
            intermediate_size = 2048
            classifier_proj_size = 256
        else:
            raise ValueError(f"Model largeness {args.largeness} not supported")
        
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
                
        config = HuBERTECGConfig(
            hidden_size = hidden_size,
            num_hidden_layers = num_hidden_layers,
            num_attention_heads = num_attention_heads,
            intermediate_size = intermediate_size,
            mask_time_prob = 0.0, 
            classifier_proj_size = classifier_proj_size,
            layerdrop = args.finetuning_layerdrop,
            conv_kernel = conv_kernel,
            conv_stride = conv_stride,
            conv_dim = conv_dim,
            mask_time_length = 1,
            hidden_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            activation_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            attention_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            feat_proj_dropout=max(0, 0 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            final_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),    
        ) # + other default params
        
        hubert = HuBERT(config)
        
        hubert = HuBERTClassification(hubert,
                                      num_labels=args.vocab_size,
                                      classifier_hidden_size=args.classifier_hidden_size,
                                      use_label_embedding=args.use_label_embedding
                                      )
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
        if type(config) == HubertConfig:
            config = HuBERTECGConfig(**config.to_dict())
        config.layerdrop = args.finetuning_layerdrop

        pretrained_hubert = HuBERT(config)
        hubert.hubert_ecg.load_state_dict(checkpoint['model_state_dict']) # load backbone weights
        
        # restore original p-dropout or set multipliers
        for name, module in pretrained_hubert.named_modules():
            if 'dropout' in name:
                module.p = 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult
        
        hubert = HuBERTClassification(pretrained_hubert, num_labels=args.vocab_size, classifier_hidden_size=args.classifier_hidden_size,  use_label_embedding=args.use_label_embedding)
        hubert.to(device)            
        
        
        global_step = 0
        best_val_loss = float('inf')
        patience_count = 0
        best_val_target_score = 0.0
        
        
        # ASSUMPTION: the classification head is always trainable
        # transformer_blocks_to_unfreeze is 0 by default, meaning that if no input comes from the user, the transformer encoder remains frozen
        # if the user wants to unfreeze some blocks, he must provide the number of blocks to unfreeze.
        
        # Freezing the convolutional feature extractor is a precise choice of the user. By default it is frozen, but it can unfrozen by using the flag --unfreeze_conv_embedder
        
        # If the freezing_steps argument is provided, then the model is completely frozen to allow to train only the head for that number of steps.
        # After that, n blocks of the transformer's encoder are unfronzen and the conv feature extractor is unfrozen as well in the training loop if the flag --unfreeze_conv_embedder is used.
        # Freezing_steps is none by default
        
        # Nte: if freezing_steps equals training_steps, then the model is completely frozen for the whole training
        
        if args.freezing_steps is not None:
            hubert.set_transformer_blocks_trainable(n_blocks=0) # freeze all transformer blocks
            hubert.set_feature_extractor_trainable(False) # freeze conv feature extractor
        else:        
            hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze) # makes trainable only the last n_blocks of the transformer encoder
            hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder) # frozen by default
        
        # if layer_wise_lr, then set a higher lr for deeper transformer layers + head than that of the rest of the trainable body of the model
        # first 8 layer with lower lr and last 4 with higher lr based on Primer Bertology
        parameters_group = []    
        if args.layer_wise_lr and all(p.requires_grad for p in hubert.hubert_ecg.encoder.layers.parameters()):
            logger.info("Setting layer-wise learning rate")
            parameters_group.append({"params": hubert.hubert_ecg.feature_projection.parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[:args.transformer_blocks_to_unfreeze-4].parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[args.transformer_blocks_to_unfreeze-4:].parameters(), "lr": lr})
            parameters_group.append({"params": hubert.classifier.parameters(), "lr": 1e-5})
        else:
            parameters_group.append({"params" : filter(lambda p : p.requires_grad, hubert.parameters()), "lr": lr})
        
        optimizer = optim.AdamW(
            parameters_group,
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

    epochs = args.training_steps // (len(train_dl) // accumulation_steps) + 1 if args.training_steps is not None else args.epochs
    
    start_epoch = global_step // len(train_dl)
    
    task2metric = {
        'multi_label': {
                        "f1-score" : F1_score(num_labels=args.vocab_size, average=None),
                        "recall" : Recall(num_labels=args.vocab_size, average=None),
                        "specificity" : Specificity(num_labels=args.vocab_size, average=None),
                        "precision" : Precision(num_labels=args.vocab_size, average=None),
                        "auroc" : MultilabelAUROC(num_labels=args.vocab_size, average=None),
                        "auprc" : AUPRC(num_labels=args.vocab_size, average=None),
        },
        'multi_class': {
                        'accuracy' : Accuracy(num_classes=args.vocab_size),
                        'auroc' : MulticlassAUROC(num_classes=args.vocab_size)
            },
        'regression' : {}
    }
    
    metrics = task2metric[args.task]
    
    assert args.target_metric in metrics.keys(), f"Target metric {args.target_metric} not available for task {args.task}"
    
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
            labels = labels.squeeze().to(device)
            
            with amp.autocast():
                logits, _ = hubert(ecg, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False, return_dict=False)
                loss = criterion_train(logits, labels)
                
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
                
                logger.info("Start validation at step {}".format(global_step))
                
                ### validation loop ###
                for ecg, _, labels in tqdm(val_dl, total=len(val_dl)):
                    
                    ecg = ecg.to(device)
                    labels = labels.squeeze().to(device)
                    
                    with torch.no_grad():
                        logits, _ = hubert(ecg, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False)
                        loss = criterion_val(logits, labels)
                    
                    val_losses.append(loss.item())
                    
                    labels = labels.long() # for metrics
                    
                    # compute metrics on single batch
                    for name, metric in metrics.items():
                        metric.update(logits, labels)
                    
                ### end of validation loop ###
                
                val_loss = np.mean(val_losses)
                train_loss = np.mean(train_losses)
                train_losses.clear() # to keep train loss aligned with val loss
                
                                
                # log non averaged metrics [1, vocab_size]
                logger.info("Validation loss = {}".format(val_loss))
                    
                # compute metrics on whole validation set and log them
                # such metrics are vectors num_labels long containing the metric for each label
                for name, metric in metrics.items():
                    score = metric.compute()
                    macro = score.mean()
                    logger.info(f"Validation {name} = {score}, macro: {macro}")
                    wandb.log({f"Validation_{name}": macro})
                    if name == args.target_metric:
                        target_score = macro
                
                # log losses
                wandb.log({
                    "Training_loss": train_loss,
                    "Validation_loss": val_loss,
                    })
                    
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
                        'finetuning_vocab_size' : args.vocab_size,
                        'use_label_embedding' : args.use_label_embedding,
                        f'target_val_{args.target_metric}': target_score
                    }
                    
                    checkpoint_name = f"hubert_{args.train_iteration}_iteration_{global_step}_finetuned_{wandb.run.id}.pt"
                    
                    torch.save(checkpoint, os.path.join(SUPERVISED_MODEL_CKPT_PATH, args.sweep_dir, checkpoint_name))
                    
                    logger.info(f"New best val loss = {best_val_loss}. Checkpoint saved at step {global_step}")
                    
                    dynamic_regularizer(optimizer=optimizer, model=hubert, penalty=False) if args.dynamic_reg else None # reward for loss
                                
                elif target_score >= best_val_target_score + MINIMAL_IMPROVEMENT:
                    
                    best_val_target_score = target_score
                    
                    # do not increment patience count but do not reset it either
                    
                    checkpoint = {
                        'global_step': global_step,
                        'best_val_loss': best_val_loss,
                        'model_config': hubert.config,
                        'model_state_dict': copy.deepcopy(hubert.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'lr_scheduler_state_dict': copy.deepcopy(lr_scheduler.state_dict()),
                        'patience_count': patience_count,
                        'linear' : True if args.classifier_hidden_size is None and not args.use_label_embedding else False,
                        'finetuning_vocab_size' : args.vocab_size,
                        'use_label_embedding' : args.use_label_embedding,
                        f'target_val_{args.target_metric}': target_score
                    }
                    
                    checkpoint_name = f"hubert_{args.train_iteration}_iteration_{global_step}_finetuned_{wandb.run.id}.pt"
                    
                    torch.save(checkpoint, os.path.join(SUPERVISED_MODEL_CKPT_PATH, args.sweep_dir, checkpoint_name))
                    
                    logger.info(f"Val loss not improved but {args.target_metric} did (= {target_score}). Checkpoint saved at step {global_step}")
                    
                    dynamic_regularizer(optimizer=optimizer, model=hubert, penalty=False) if args.dynamic_reg else None # reward for target metric
                    
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

    parser = argparse.ArgumentParser(description="Train Hubert-ECG")
    
    #train_iteration
    parser.add_argument(
        "train_iteration",
        help="Hubert training iteration in {1, 2, 3}", 
        type=int,
        choices=[1, 2, 3]
        )
    
    #path_to_dataset_csv_train
    parser.add_argument(
        "path_to_dataset_csv_train",
        help="Path to the csv file containing the training dataset",
        type=str
    )
    
    #path_to_dataset_csv_val
    parser.add_argument(
        "path_to_dataset_csv_val",
        help="Path to the csv file containing the validation dataset",
        type=str
    )
    
    #sweep_dir
    parser.add_argument(
        "--sweep_dir",
        help="Sweep directory. Default `.`",
        type=str,
        default="."
    )
    
    #ramp_up_perc
    parser.add_argument(
        "--ramp_up_perc",
        help="[OPT.] Percentage of training steps for the ramp up phase. Default 0.08",
        type=float,
        default=0.08
    )
    
    #training_steps
    parser.add_argument(
        "--training_steps",
        help="[OPT.] Number of training steps to perform. Use this or epochs",
        type=int
    )
    
    #epochs
    parser.add_argument(
        "--epochs",
        help="[OPT.] Number of epochs to perform. Use this or training_steps",
        type=int
    )
    
    #vocab_size
    parser.add_argument(
        "vocab_size",
        help="Vocabulary size, i.e. num of labels/clusters",
        type=int
    )
    
    #val_interval
    parser.add_argument(
        "--val_interval",
        help="[OPT.] Training steps to wait before validation. Must be specified if training_steps is used. Default None",
        type=int,
        default=None
    )
    
    #patience
    parser.add_argument(
        "patience",
        help="Patience for early stopping",
        type=int
    )
    
    #batch_size
    parser.add_argument(
        "batch_size",
        help="Batch_size",
        type=int
    )
    
    #downsampling_factor
    parser.add_argument(
        "--downsampling_factor",
        help="[OPT] Downsampling factor to apply to the ECG signal",
        type=int
    )

    #random_crop
    parser.add_argument(
        "--random_crop", 
        help="Whether to perform random crop of 5 sec as data augmentation",
        action="store_true",
        default=False
    )
    
    #target_metric
    parser.add_argument(
        "target_metric",
        type=str,
        help="Target metric (macro) to optimize during finetuning",
        choices=["f1_score", "recall", "precision", "specificity", "auroc", "auprc", "accuracy"]
    )
    
    #accumulation_steps
    parser.add_argument(
        "--accumulation_steps", 
        help="[OPT] Number of batch gradients to accumulate before updating model params. Default 1",
        type=int,
        default=1
    )
    
    #label_start_index
    parser.add_argument(
        "--label_start_index",
        help="[OPT] Index of the first label in the dataset csv file. Default 3",
        type=int,
        default=3
    )
    
    #freezing_steps
    parser.add_argument(
        "--freezing_steps",
        help="[OPT] Number of finetuning steps to keep frozen the base model weights",
        type=int,
        default=None
    )  
    
    #resume_finetuning
    parser.add_argument(
        "--resume_finetuning",
        help="Whether to resume finetuning",
        action="store_true",
        default=False
    )
    
    #unfreeze_conv_embedder
    parser.add_argument(
        "--unfreeze_conv_embedder",
        help="[OPT] Whether to unfreeze the convolutional feature extractor during fine-tuning. Normally frozen",
        action="store_true",
        default=False
    )
    
    #transformer_blocks_to_unfreeze
    parser.add_argument(
        "--transformer_blocks_to_unfreeze",
        help="[OPT] Number of transformer blocks to unfreeze after freezing_steps. Default 0",
        type=int,
        default=0
    )
    
    #lr
    parser.add_argument(
        "--lr",
        help="[OPT] Learning rate. Default 1e-5",
        type=float,
        default=1e-5
    )
    
    #layer_wise_lr
    parser.add_argument(
        "--layer_wise_lr",
        help="[OPT] Whether to use layer-wise learning rate. Use --lr for last 4 encoding layers and head, 1e-8 for the rest",
        action="store_true",
        default=False
    )
    
    #load_path
    parser.add_argument(
        "--load_path",
        help="Path to a model checkpoint that is to load to start/resume fine-tuning",
        type=str
    )
    
    #classifier_hidden_size
    parser.add_argument(
        "--classifier_hidden_size",
        help="[OPT.] Hidden size of the MLP head used for classification in finetuning. If None, then linear classifier. Default None",
        type=int,
        default=None
    )
    
    #use_label_embedding
    parser.add_argument(
        "--use_label_embedding",
        help="[OPT.] Whether to use label embeddings in the classification head. Default False",
        action="store_true"
    )

    #intervals_for_penalty
    parser.add_argument(
        "--intervals_for_penalty",
        help="['OPT.] Number of validation intervals with worsening performance to wait before applying penalizing regularization",
        type=int,
        default=3
    )
    
    #dynamic_reg
    parser.add_argument(
        "--dynamic_reg",
        help="[OPT.] Whether to apply dynamic regularization to the model. Default False",
        action="store_true"
    )
    
    #use_loss_weights
    parser.add_argument(
        "--use_loss_weights",
        help="[OPT.] Whether to use loss weights in the loss function. Default False",
        action="store_true"
    )
    
    #random_init
    parser.add_argument(
        "--random_init",
        help="[OPT.] Whether to initialize the model with random weights. Default False",
        action="store_true"
    ) 
    
    #largeness
    parser.add_argument(
        "--largeness",
        help="[OPT.] Model largeness in {base, large, x-large} in case of random initialization. Default base",
        type=str,
        choices=["small", "base", "large"]
    )
    
    # weight_decay_mult
    parser.add_argument(
        "--weight_decay_mult",
        help="Weight decay mult. Default 1 (i.e. WD=0.01)",
        type=int,
        default=1
    )
    
    # model_dropout_mult
    parser.add_argument(
        "--model_dropout_mult",
        help="Model dropout. Default 0 (i.e. dropout=0.1)",
        type=int,
        default=0
    )
    
    #finetuning_layerdrop
    parser.add_argument(
        "--finetuning_layerdrop",
        help="Layerdrop for the finetuning phase. Default 0.1 as in pre-training",
        type=float,
        default=0.1
    )

    #wandb_run_name
    parser.add_argument(
        "--wandb_run_name",
        help="The name to give to this run",
        type=str,
        default=None
    )
    
    parser.add_argument(
        '--task', 
        help="Task to perform in {multi_class, multi_label, regression}",
        choices=["multi_class", "multi_label", "regression"],
        type=str,
        default="multi_label"
    )
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available. CPU finetuning not supported")
        exit(1)
        
    if args.train_iteration > 3 or args.train_iteration < 1:
        raise ValueError(f"Argument train_iteration must be an integer in [1, 3] range. Inserted {args.train_iteration}")
    
    if args.training_steps is None and args.epochs is None:
        raise ValueError("One argument between training_steps and epochs must be provided")
        
    if args.training_steps is not None and args.val_interval is None:
        raise ValueError("Argument val_interval must be provided if argument training_steps is provided")

    if args.training_steps is not None and args.training_steps % args.val_interval != 0:
        raise ValueError(f"Argument training_steps must be divisible by argument val_interval. Inserted {args.training_steps} and {args.val_interval}")
    
    if args.ramp_up_perc < 0 or args.ramp_up_perc > 1:
        raise ValueError("Argument ramp_up_perc must be a float in [0, 1] range") 
    
    if args.random_init and args.resume_finetuning:
        raise ValueError("Arguments random_init and resume_finetuning cannot be provided together")
    
    if (args.resume_finetuning or args.random_init == False) and args.load_path is None:
        raise ValueError("Argument load_path must be provided when start/resume finetuning")
       
    if args.freezing_steps is not None and args.freezing_steps > args.training_steps:
        raise ValueError("Argument freezing_steps cannot be greater than argument training steps")
    
    if args.accumulation_steps is not None and args.training_steps is not None and args.training_steps % args.accumulation_steps != 0:
        raise ValueError("Argument training_steps must be divisible by argument accumulation_steps")
    
    if args.random_init and args.largeness is None:
        raise ValueError("Argument largeness must be provided if argument random_init is provided")
    
    if args.random_init and args.load_path is not None:
        logger.warning("Argument random_init is provided. Argument load_path will be ignored")
    
    if args.dynamic_reg and args.patience < args.intervals_for_penalty:
        raise ValueError(f"Argument patience must be greater or equal to argument intervals_for_penalty when using dynamic_reg. Setting patience = {args.intervals_for_penalty}")

    
    print("Inserted arguments: ")
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
        
    
    ### NOTE: this is to test sweeps ###

    # wandb.init(entity="cardi-ai", project="ECG-pretraining", group=("supervised"))
    
    # args = wandb.config

    finetune(args) 
