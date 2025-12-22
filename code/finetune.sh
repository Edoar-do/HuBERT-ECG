#!/bin/bash

# Commands to run to perform the fine-tuning of HuBERT-ECG SMALL, BASE and LARGE models on the PTB-XL All dataset as done in the paper
#!/bin/bash

### FINE-TUNING SMALL MODEL SIZE ###

# RIBEIRO

# We cannot disclose the Ribeiro dataset. Ask the original authors for access.
python finetune.py 3 path/to/ribeiro_train.csv path/to/ribeiro_val.csv 6 5 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=5000 --finetuning_layerdrop=0.0 --wandb_run_name=SMALL_ribeiro

# NINGBO

python finetune.py 3 path/to/ningbo_train_0.csv path/to/ningbo_val_0.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_ningbo0
python finetune.py 3 path/to/ningbo_train_1.csv path/to/ningbo_val_1.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_ningbo1
python finetune.py 3 path/to/ningbo_train_2.csv path/to/ningbo_val_2.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_ningbo2
python finetune.py 3 path/to/ningbo_train_3.csv path/to/ningbo_val_3.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_ningbo3

# PTB

python finetune.py 3 path/to/ptb_train_0.csv path/to/ptb_train_0.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptb_0
python finetune.py 3 path/to/ptb_train_1.csv path/to/ptb_train_1.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptb_1
python finetune.py 3 path/to/ptb_train_2.csv path/to/ptb_train_2.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptb_2
python finetune.py 3 path/to/ptb_train_3.csv path/to/ptb_train_3.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptb_3

# CPSC

python finetune.py 3 path/to/cpsc_train_0.csv path/to/cpsc_val_0.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_cpsc0
python finetune.py 3 path/to/cpsc_train_1.csv path/to/cpsc_val_1.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_cpsc1
python finetune.py 3 path/to/cpsc_train_2.csv path/to/cpsc_val_2.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_cpsc2
python finetune.py 3 path/to/cpsc_train_3.csv path/to/cpsc_val_3.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_cpsc3

#SPH

python finetune.py 3 path/to/sph_train.csv path/to/sph_val.csv 44 8 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_sph

#HEFEI

python finetune.py 3 path/to/hefei_train0.csv path/to/hefei_val0.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_hefei0
python finetune.py 3 path/to/hefei_train1.csv path/to/hefei_val1.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_hefei1
python finetune.py 3 path/to/hefei_train2.csv path/to/hefei_val2.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_hefei2
python finetune.py 3 path/to/hefei_train3.csv path/to/hefei_val3.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_hefei3

# PTB_ALL

python finetune.py 3 path/to/ptb_all_train.csv path/to/ptb_all_val.csv 71 8 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptb_all

# PTB_FORM

python finetune.py 3 path/to/ptb_form_train.csv path/to/ptb_form_val.csv 19 8 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptb_form

# PTB_RHYTHM

python finetune.py 3 path/to/ptb_rhythm_train.csv path/to/ptb_rhythm_val.csv 12 8 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptb_rhythm

# PTB_DIAG

python finetune.py 3 path/to/ptb_diag_train.csv path/to/ptb_diag_val.csv 44 8 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptb_diag

# PTB_DIAG_SUPERCLASS

python finetune.py 3 path/to/ptb_diag_subclass_train.csv path/to/ptb_diag_subclass_val.csv 23 8 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptb_diag_subclass

# PTB_DIAG_SUBCLASS

python finetune.py 3 path/to/ptb_diag_superclass_train.csv path/to/ptb_diag_superclass_val.csv 5 8 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=ptb_diag_superclass

# SAMITROP 

python finetune.py 3 path/to/samitrop_train0.csv path/to/samitrop_train0.csv 2 2 64 auroc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=SMALL_samitrop0 --task=multi_class
python finetune.py 3 path/to/samitrop_train1.csv path/to/samitrop_train1.csv 2 2 64 auroc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=SMALL_samitrop1 --task=multi_class
python finetune.py 3 path/to/samitrop_train2.csv path/to/samitrop_train2.csv 2 2 64 aurpc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=SMALL_samitrop2 --task=multi_class
python finetune.py 3 path/to/samitrop_train3.csv path/to/samitrop_train3.csv 2 2 64 auroc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=SMALL_samitrop3 --task=multi_class

# CPSC-EXTRA

python finetune.py 3 path/to/cpsc_extra_train_0.csv path/to/cpsc_extra_train_0.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_cpsc_extra0
python finetune.py 3 path/to/cpsc_extra_train_1.csv path/to/cpsc_extra_train_1.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_cpsc_extra1
python finetune.py 3 path/to/cpsc_extra_train_2.csv path/to/cpsc_extra_train_2.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_cpsc_extra2
python finetune.py 3 path/to/cpsc_extra_train_3.csv path/to/cpsc_extra_train_3.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_cpsc_extra3

# GEORGIA

python finetune.py 3 path/to/georgia_train_0.csv path/to/georgia_val_0.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_georgia0
python finetune.py 3 path/to/georgia_train_1.csv path/to/georgia_val_1.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_georgia1
python finetune.py 3 path/to/georgia_train_2.csv path/to/georgia_val_2.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_georgia2
python finetune.py 3 path/to/georgia_train_3.csv path/to/georgia_val_3.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=SMALL_georgia3

# CHAPMAN 

python finetune.py 3 path/to/chapman_train0.csv path/to/chapman_val0.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=SMALL_chapman0
python finetune.py 3 path/to/chapman_train1.csv path/to/chapman_val1.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=SMALL_chapman1
python finetune.py 3 path/to/chapman_train2.csv path/to/chapman_val2.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=SMALL_chapman2
python finetune.py 3 path/to/chapman_train3.csv path/to/chapman_val3.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=SMALL_chapman3

# CARDIOLEARNING 

# We cannot disclose the CardioLearning dataset because it contains Ribeiro. Once accessed Ribeiro, we have detailed how to prepare CardioLearning in the paper.

python finetune.py 3 path/to/cardiolearning_train_0.csv path/to/cardiolearning_val_0.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=SMALL_cardiolearning0 --transformer_blocks_to_unfreeze=8 --task=multi_label
python finetune.py 3 path/to/cardiolearning_train_1.csv path/to/cardiolearning_val_1.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=SMALL_cardiolearning1 --transformer_blocks_to_unfreeze=8 --task=multi_label
python finetune.py 3 path/to/cardiolearning_train_2.csv path/to/cardiolearning_val_2.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=SMALL_cardiolearning2 --transformer_blocks_to_unfreeze=8 --task=multi_label
python finetune.py 3 path/to/cardiolearning_train_3.csv path/to/cardiolearning_val_3.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=SMALL_cardiolearning3 --transformer_blocks_to_unfreeze=8 --task=multi_label

# CLINICAL DATA

python finetune.py 3 path/to/clinical_dataset_train_0.csv path/to/clinical_dataset_train_0.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_clinical_data0 
python finetune.py 3 path/to/clinical_dataset_train_1.csv path/to/clinical_dataset_train_1.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_clinical_data1
python finetune.py 3 path/to/clinical_dataset_train_2.csv path/to/clinical_dataset_train_2.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_clinical_data2 
python finetune.py 3 path/to/clinical_dataset_train_3.csv path/to/clinical_dataset_train_3.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_clinical_data3 

### FINE-TUNING BASE MODEL SIZE ###

# RIBEIRO

# We cannot disclose the Ribeiro dataset. Ask the original authors for access.
python finetune.py 3 path/to/ribeiro_train.csv path/to/ribeiro_val.csv 6 5 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=5000 --finetuning_layerdrop=0.0 --wandb_run_name=BASE_ribeiro

# NINGBO

python finetune.py 3 path/to/ningbo_train_0.csv path/to/ningbo_val_0.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ningbo0
python finetune.py 3 path/to/ningbo_train_1.csv path/to/ningbo_val_1.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ningbo1
python finetune.py 3 path/to/ningbo_train_2.csv path/to/ningbo_val_2.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ningbo2
python finetune.py 3 path/to/ningbo_train_3.csv path/to/ningbo_val_3.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ningbo3

# PTB

python finetune.py 3 path/to/ptb_base_train_0.csv path/to/ptb_base_train_0.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_ptb_base0
python finetune.py 3 path/to/ptb_base_train_1.csv path/to/ptb_base_train_1.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_ptb_base1
python finetune.py 3 path/to/ptb_base_train_2.csv path/to/ptb_base_train_2.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_ptb_base2
python finetune.py 3 path/to/ptb_base_train_3.csv path/to/ptb_base_train_3.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_ptb_base3

# CPSC

python finetune.py 3 path/to/cpsc_train_0.csv path/to/cpsc_val_0.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.2 --random_crop --wandb_run_name=BASE_cpsc0
python finetune.py 3 path/to/cpsc_train_1.csv path/to/cpsc_val_1.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.2 --random_crop --wandb_run_name=BASE_cpsc1
python finetune.py 3 path/to/cpsc_train_2.csv path/to/cpsc_val_2.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.2 --random_crop --wandb_run_name=BASE_cpsc2
python finetune.py 3 path/to/cpsc_train_3.csv path/to/cpsc_val_3.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.2 --random_crop --wandb_run_name=BASE_cpsc3

#SPH

python finetune.py 3 path/to/sph_train.csv path/to/sph_val.csv 44 8 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_sph

#HEFEI

python finetune.py 3 path/to/hefei_train0.csv path/to/hefei_val0.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_hefei0
python finetune.py 3 path/to/hefei_train1.csv path/to/hefei_val1.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_hefei1
python finetune.py 3 path/to/hefei_train2.csv path/to/hefei_val2.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_hefei2
python finetune.py 3 path/to/hefei_train3.csv path/to/hefei_val3.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_hefei3

# PTB_ALL

python finetune.py 3 path/to/ptb_all_train.csv path/to/ptb_all_val.csv 71 8 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ptb_all

# PTB_FORM

python finetune.py 3 path/to/ptb_form_train.csv path/to/ptb_form_val.csv 19 8 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ptb_form

# PTB_RHYTHM

python finetune.py 3 path/to/ptb_rhythm_train.csv path/to/ptb_rhythm_val.csv 12 8 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ptb_rhythm
# PTB_DIAG

python finetune.py 3 path/to/ptb_diag_train.csv path/to/ptb_diag_val.csv 44 8 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ptb_diag

# PTB_DIAG_SUPERCLASS

python finetune.py 3 path/to/ptb_diag_subclass_train.csv path/to/ptb_diag_subclass_val.csv 23 8 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ptb_diag_subclass

# PTB_DIAG_SUBCLASS

python finetune.py 3 path/to/ptb_diag_superclass_train.csv path/to/ptb_diag_superclass_val.csv 5 8 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=ptb_diag_superclass

# SAMITROP 

python finetune.py 3 path/to/samitrop_train0.csv path/to/samitrop_train0.csv 2 2 64 auroc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=BASE_samitrop0 --task=multi_class
python finetune.py 3 path/to/samitrop_train1.csv path/to/samitrop_train1.csv 2 2 64 auroc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=BASE_samitrop1 --task=multi_class
python finetune.py 3 path/to/samitrop_train2.csv path/to/samitrop_train2.csv 2 2 64 aurpc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=BASE_samitrop2 --task=multi_class
python finetune.py 3 path/to/samitrop_train3.csv path/to/samitrop_train3.csv 2 2 64 auroc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=BASE_samitrop3 --task=multi_class

# CPSC-EXTRA

python finetune.py 3 path/to/cpsc_extra_train_0.csv path/to/cpsc_extra_train_0.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_cpsc_extra0
python finetune.py 3 path/to/cpsc_extra_train_1.csv path/to/cpsc_extra_train_1.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_cpsc_extra1
python finetune.py 3 path/to/cpsc_extra_train_2.csv path/to/cpsc_extra_train_2.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_cpsc_extra2
python finetune.py 3 path/to/cpsc_extra_train_3.csv path/to/cpsc_extra_train_3.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_cpsc_extra3

# GEORGIA

python finetune.py 3 path/to/georgia_train_0.csv path/to/georgia_val_0.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.2 --random_crop --wandb_run_name=BASE_georgia0
python finetune.py 3 path/to/georgia_train_1.csv path/to/georgia_val_1.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.2 --random_crop --wandb_run_name=BASE_georgia1
python finetune.py 3 path/to/georgia_train_2.csv path/to/georgia_val_2.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.2 --random_crop --wandb_run_name=BASE_georgia2
python finetune.py 3 path/to/georgia_train_3.csv path/to/georgia_val_3.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.2 --random_crop --wandb_run_name=BASE_georgia3

# CHAPMAN 

python finetune.py 3 path/to/chapman_train0.csv path/to/chapman_val0.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=BASE_chapman0
python finetune.py 3 path/to/chapman_train1.csv path/to/chapman_val1.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=BASE_chapman1
python finetune.py 3 path/to/chapman_train2.csv path/to/chapman_val2.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=BASE_chapman2
python finetune.py 3 path/to/chapman_train3.csv path/to/chapman_val3.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=BASE_chapman3

# CARDIOLEARNING 

# We cannot disclose the CardioLearning dataset because it contains Ribeiro. Once accessed Ribeiro, we have detailed how to prepare CardioLearning in the paper.

python finetune.py 3 path/to/cardiolearning_train_0.csv path/to/cardiolearning_val_0.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=BASE_cardiolearning0 --transformer_blocks_to_unfreeze=12 --task=multi_label
python finetune.py 3 path/to/cardiolearning_train_1.csv path/to/cardiolearning_val_1.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=BASE_cardiolearning1 --transformer_blocks_to_unfreeze=12 --task=multi_label
python finetune.py 3 path/to/cardiolearning_train_2.csv path/to/cardiolearning_val_2.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=BASE_cardiolearning2 --transformer_blocks_to_unfreeze=12 --task=multi_label
python finetune.py 3 path/to/cardiolearning_train_3.csv path/to/cardiolearning_val_3.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=BASE_cardiolearning3 --transformer_blocks_to_unfreeze=12 --task=multi_label

# CLINICAL DATA

python finetune.py 3 path/to/clinical_dataset_train_0.csv path/to/clinical_dataset_train_0.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_clinical_data0 
python finetune.py 3 path/to/clinical_dataset_train_1.csv path/to/clinical_dataset_train_1.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_clinical_data1
python finetune.py 3 path/to/clinical_dataset_train_2.csv path/to/clinical_dataset_train_2.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_clinical_data2 
python finetune.py 3 path/to/clinical_dataset_train_3.csv path/to/clinical_dataset_train_3.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=BASE_clinical_data3 

### FINE-TUNING LARGE MODEL SIZE ###

# RIBEIRO

# We cannot disclose the Ribeiro dataset. Ask the original authors for access.
python finetune.py 3 path/to/ribeiro_train.csv path/to/ribeiro_val.csv 6 5 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=5000 --finetuning_layerdrop=0.0 --wandb_run_name=LARGE_ribeiro

# NINGBO

python finetune.py 3 path/to/ningbo_train_0.csv path/to/ningbo_val_0.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ningbo0
python finetune.py 3 path/to/ningbo_train_1.csv path/to/ningbo_val_1.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ningbo1
python finetune.py 3 path/to/ningbo_train_2.csv path/to/ningbo_val_2.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ningbo2
python finetune.py 3 path/to/ningbo_train_3.csv path/to/ningbo_val_3.csv 76 10 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ningbo3

# PTB

python finetune.py 3 path/to/ptb_large_train_0.csv path/to/ptb_large_train_0.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ptb_large0
python finetune.py 3 path/to/ptb_large_train_1.csv path/to/ptb_large_train_1.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ptb_large1
python finetune.py 3 path/to/ptb_large_train_2.csv path/to/ptb_large_train_2.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ptb_large2
python finetune.py 3 path/to/ptb_large_train_3.csv path/to/ptb_large_train_3.csv 14 1 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ptb_large3

# CPSC

python finetune.py 3 path/to/cpsc_train_0.csv path/to/cpsc_val_0.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_cpsc0
python finetune.py 3 path/to/cpsc_train_1.csv path/to/cpsc_val_1.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_cpsc1
python finetune.py 3 path/to/cpsc_train_2.csv path/to/cpsc_val_2.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_cpsc2
python finetune.py 3 path/to/cpsc_train_3.csv path/to/cpsc_val_3.csv 9 13 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_cpsc3

#SPH

python finetune.py 3 path/to/sph_train.csv path/to/sph_val.csv 44 8 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_sph

#HEFEI

python finetune.py 3 path/to/hefei_train0.csv path/to/hefei_val0.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_hefei0
python finetune.py 3 path/to/hefei_train1.csv path/to/hefei_val1.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_hefei1
python finetune.py 3 path/to/hefei_train2.csv path/to/hefei_val2.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_hefei2
python finetune.py 3 path/to/hefei_train3.csv path/to/hefei_val3.csv 27 8 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_hefei3

# PTB_ALL

python finetune.py 3 path/to/ptb_all_train.csv path/to/ptb_all_val.csv 71 8 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=LARGE_ptb_all

# PTB_FORM

python finetune.py 3 path/to/ptb_form_train.csv path/to/ptb_form_val.csv 19 8 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=LARGE_ptb_form

# PTB_RHYTHM

python finetune.py 3 path/to/ptb_rhythm_train.csv path/to/ptb_rhythm_val.csv 12 8 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=LARGE_ptb_rhythm

# PTB_DIAG

python finetune.py 3 path/to/ptb_diag_train.csv path/to/ptb_diag_val.csv 44 8 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=LARGE_ptb_diag

# PTB_DIAG_SUPERCLASS

python finetune.py 3 path/to/ptb_diag_subclass_train.csv path/to/ptb_diag_subclass_val.csv 23 8 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=LARGE_ptb_diag_subclass

# PTB_DIAG_SUBCLASS

python finetune.py 3 path/to/ptb_diag_superclass_train.csv path/to/ptb_diag_superclass_val.csv 5 8 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=ptb_diag_superclass

# SAMITROP 

python finetune.py 3 path/to/samitrop_train0.csv path/to/samitrop_train0.csv 2 2 64 auroc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=LARGE_samitrop0 --task=multi_class
python finetune.py 3 path/to/samitrop_train1.csv path/to/samitrop_train1.csv 2 2 64 auroc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=LARGE_samitrop1 --task=multi_class
python finetune.py 3 path/to/samitrop_train2.csv path/to/samitrop_train2.csv 2 2 64 aurpc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=LARGE_samitrop2 --task=multi_class
python finetune.py 3 path/to/samitrop_train3.csv path/to/samitrop_train3.csv 2 2 64 auroc --load_path=../modesl/12-lead/12-lead/checkpoints/pretrained_hubert_ecgs/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=6 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0. --random_crop --wandb_run_name=LARGE_samitrop3 --task=multi_class

# CPSC-EXTRA

python finetune.py 3 path/to/cpsc_extra_train_0.csv path/to/cpsc_extra_train_0.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_cpsc_extra0
python finetune.py 3 path/to/cpsc_extra_train_1.csv path/to/cpsc_extra_train_1.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_cpsc_extra1
python finetune.py 3 path/to/cpsc_extra_train_2.csv path/to/cpsc_extra_train_2.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_cpsc_extra2
python finetune.py 3 path/to/cpsc_extra_train_3.csv path/to/cpsc_extra_train_3.csv 52 2 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_cpsc_extra3

# GEORGIA

python finetune.py 3 path/to/georgia_train_0.csv path/to/georgia_val_0.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_georgia0
python finetune.py 3 path/to/georgia_train_1.csv path/to/georgia_val_1.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_georgia1
python finetune.py 3 path/to/georgia_train_2.csv path/to/georgia_val_2.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_georgia2
python finetune.py 3 path/to/georgia_train_3.csv path/to/georgia_val_3.csv 62 15 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --dynamic_reg --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_georgia3

# CHAPMAN 

python finetune.py 3 path/to/chapman_train0.csv path/to/chapman_val0.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=LARGE_chapman0
python finetune.py 3 path/to/chapman_train1.csv path/to/chapman_val1.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=LARGE_chapman1
python finetune.py 3 path/to/chapman_train2.csv path/to/chapman_val2.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=LARGE_chapman2
python finetune.py 3 path/to/chapman_train3.csv path/to/chapman_val3.csv 52 20 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=100000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --dynamic_reg --random_crop --wandb_run_name=LARGE_chapman3

# CARDIOLEARNING 

# We cannot disclose the CardioLearning dataset because it contains Ribeiro. Once accessed Ribeiro, we have detailed how to prepare CardioLearning in the paper.

python finetune.py 3 path/to/cardiolearning_train_0.csv path/to/cardiolearning_val_0.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=LARGE_cardiolearning0 --transformer_blocks_to_unfreeze=16 --task=multi_label
python finetune.py 3 path/to/cardiolearning_train_1.csv path/to/cardiolearning_val_1.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=LARGE_cardiolearning1 --transformer_blocks_to_unfreeze=16 --task=multi_label
python finetune.py 3 path/to/cardiolearning_train_2.csv path/to/cardiolearning_val_2.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=LARGE_cardiolearning2 --transformer_blocks_to_unfreeze=16 --task=multi_label
python finetune.py 3 path/to/cardiolearning_train_3.csv path/to/cardiolearning_val_3.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=LARGE_cardiolearning3 --transformer_blocks_to_unfreeze=16 --task=multi_label

# CLINICAL DATA

python finetune.py 3 path/to/clinical_dataset_train_0.csv path/to/clinical_dataset_train_0.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_clinical_data0 
python finetune.py 3 path/to/clinical_dataset_train_1.csv path/to/clinical_dataset_train_1.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_clinical_data1
python finetune.py 3 path/to/clinical_dataset_train_2.csv path/to/clinical_dataset_train_2.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_clinical_data2 
python finetune.py 3 path/to/clinical_dataset_train_3.csv path/to/clinical_dataset_train_3.csv 7 10 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=1 --use_loss_weights --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=50 --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_clinical_data3 



