#!/bin/bash

# small model size

python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_small_random_38.5k_ningbo0.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo0_small_random --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_small_random_30.5k_ningbo1.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo1_small_random --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_small_random_43k_ningbo2.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo2_small_random --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_small_random_30k_ningbo3.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo3_small_random --tta_aggregation=max

python test.py /path/to/ptb_test_0.csv . 64 /path/to/hubert_small_random_4k_ptb0.pt --downsampling_factor=5 --tta --save_id=ptb0_small_random --tta_aggregation=max
python test.py /path/to/ptb_test_1.csv . 64 /path/to/hubert_small_random_3.9k_ptb1.pt --downsampling_factor=5 --tta --save_id=ptb1_small_random --tta_aggregation=max
python test.py /path/to/ptb_test_2.csv . 64 /path/to/hubert_small_random_3.25k_ptb2.pt --downsampling_factor=5 --tta --save_id=ptb2_small_random --tta_aggregation=max
python test.py /path/to/ptb_test_3.csv . 64 /path/to/hubert_small_random_2.95k_ptb3.pt --downsampling_factor=5 --tta --save_id=ptb3_small_random --tta_aggregation=max

python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_small_random_6.15k_cpsc0.pt --downsampling_factor=5 --tta --save_id=cpsc0_small_random --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_small_random_5.7k_cpsc1.pt --downsampling_factor=5 --tta --save_id=cpsc1_small_random --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_small_random_8.35k_cpsc2.pt --downsampling_factor=5 --tta --save_id=cpsc2_small_random --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_small_random_6.75k_cpsc3.pt --downsampling_factor=5 --tta --save_id=cpsc3_small_random --tta_aggregation=max --n_augs=5

python test.py /path/to/sph_test.csv . 64 /path/to/hubert_small_random_27k_sph.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=sph_small_random --tta_aggregation=max --n_augs=5

python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_small_random_20.5k_hefei4.pt --downsampling_factor=5 --tta --save_id=hefei3_small_random --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_small_random_15.5k_hefei0.pt --downsampling_factor=5 --tta --save_id=hefei0_small_random --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_small_random_22k_hefei1.pt --downsampling_factor=5 --tta --save_id=hefei1_small_random --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_small_random_15k_hefei2.pt --downsampling_factor=5 --tta --save_id=hefei2_small_random --tta_aggregation=max --n_augs=5

python test.py /path/to/ptb_all_test.csv . 64 /path/to/hubert_small_random_27k_ptbAll.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_all_small_random --tta_aggregation=max
python test.py /path/to/ptb_form_test.csv . 64 /path/to/hubert_small_random_17k_ptbForm.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_form_small_random --tta_aggregation=max --n_augs=3
python test.py /path/to/ptb_rhythm_test.csv . 64 /path/to/hubert_small_random_19.5k_ptbRhythm.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_rhythm_small_random --tta_aggregation=max
python test.py /path/to/ptb_diag_test.csv . 64 /path/to/hubert_small_random_21k_ptbDiag.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_small_random --tta_aggregation=max
python test.py /path/to/ptb_diag_subclass_test.csv . 64 /path/to/hubert_small_random_12.5k_ptbDiagSubclass.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_subclass_small_random --tta_aggregation=max
python test.py /path/to/ptb_diag_superclass_test.csv . 64 /path/to/hubert_small_random_18.5k_ptbDiagSuperclass.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_superclass_small_random --tta_aggregation=max

python test.py /path/to/ribeiro_test_set.csv /path/to/ribeiro_test/ 64 /path/to/hubert_small_random_60k_tnmg.pt --downsampling_factor=5 --save_id=ribeiro_small_random 

python test.py /path/to/samitrop_test0.csv . 64 /path/to/hubert_small_random_1.3k_samitrop0.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop0_small_random --tta_aggregation=max
python test.py /path/to/samitrop_test1.csv . 64 /path/to/hubert_small_random_1.55k_samitrop1.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop1_small_random --tta_aggregation=max
python test.py /path/to/samitrop_test2.csv . 64 /path/to/hubert_small_random_1.3k_samitrop2.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop2_small_random --tta_aggregation=max
python test.py /path/to/samitrop_test3.csv . 64 /path/to/hubert_small_random_1.4k_samitrop3.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop3_small_random --tta_aggregation=max

python test.py /path/to/cpsc_extra_test_0.csv . 64 /path/to/hubert_small_random_7.8k_cpscExtra0.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=7 --save_id=cpsc_extra0_small_random 
python test.py /path/to/cpsc_extra_test_1.csv . 64 /path/to/hubert_small_random_6.7k_cpscExtra1.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=cpsc_extra1_small_random 
python test.py /path/to/cpsc_extra_test_2.csv . 64 /path/to/hubert_small_random_8.95k_cpscExtra2.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=cpsc_extra2_small_random 
python test.py /path/to/cpsc_extra_test_3.csv . 64 /path/to/hubert_small_random_9.15k_cpscExtra3.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=3 --save_id=cpsc_extra3_small_random 

python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_small_random_9k_georgia0.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia0_small_random 
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_small_random_2.55k_georgia1.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia1_small_random 
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_small_random_13.5k_georgia2.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia2_small_random 
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_small_random_7.5k_georgia3.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia3_small_random 

python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_small_random_10.4k_chapman0.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman0_small_random 
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_small_random_6.75k_chapman1.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman1_small_random 
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_small_random_10.3k_chapman2.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman2_small_random 
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_small_random_9.45k_chapman3.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman3_small_random 

python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_random_34k_general.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_small --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_26k_cardiolearning_random_1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_small_1 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_32k_cardiolearning_random_2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_small_2 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_28.5k_cardiolearning_random_3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_small_3 --tta --tta_aggregation=max --n_augs=3

# base model size 

python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_base_random_26.5k_ningbo0.pt --downsampling_factor=5 --tta --n_augs=3 --save_id=ningbo0_base_random --tta_aggregation=max 
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_base_random_17.5k_ningbo1.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo1_base_random --tta_aggregation=max 
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_base_random_24.5k_ningbo2.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo2_base_random --tta_aggregation=max 
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_base_random_17.5k_ningbo3.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo3_base_random --tta_aggregation=max 

python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_base_random_5.05k_cpsc0.pt --downsampling_factor=5 --tta --save_id=cpsc0_base_random --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_base_random_5.1k_cpsc1.pt --downsampling_factor=5 --tta --save_id=cpsc1_base_random --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_base_random_4.2k_cpsc2.pt --downsampling_factor=5 --tta --save_id=cpsc2_base_random --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_base_random_3.45k_cpsc3.pt --downsampling_factor=5 --tta --save_id=cpsc3_base_random --tta_aggregation=max --n_augs=5

python test.py /path/to/sph_test.csv . 64 /path/to/hubert_base_random_14.5k_sph.pt --downsampling_factor=5 --label_start_index=4 --save_id=sph_base_random --tta --tta_aggregation=max --n_augs=5

python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_base_random_14k_hefei0.pt --downsampling_factor=5 --tta --save_id=hefei0_base_random --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_base_random_12k_hefei1.pt --downsampling_factor=5 --tta --save_id=hefei1_base_random --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_base_random_18k_hefei2.pt --downsampling_factor=5 --tta --save_id=hefei2_base_random --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_base_random_17k_hefei3.pt --downsampling_factor=5 --tta --save_id=hefei3_base_random --tta_aggregation=max --n_augs=5

python test.py /path/to/ptb_all_test.csv . 64 /path/to/hubert_base_random_19k_ptbAll.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_all_base_random --tta_aggregation=max 
python test.py /path/to/ptb_form_test.csv . 64 /path/to/hubert_base_random_9k_ptbForm.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_form_base_random --tta_aggregation=max --n_augs=5
python test.py /path/to/ptb_rhythm_test.csv . 64 /path/to/hubert_base_random_25.5k_ptbRhythm.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_rhythm_base_random --tta_aggregation=max 
python test.py /path/to/ptb_diag_test.csv . 64 /path/to/hubert_base_random_24.5k_ptbDiag.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_base_random --tta_aggregation=max 
python test.py /path/to/ptb_diag_superclass_test.csv . 64 /path/to/hubert_base_random_15.5k_ptbDiagSuperclass.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_superclass_base_random --tta_aggregation=max 
python test.py /path/to/ptb_diag_subclass_test.csv . 64 /path/to/hubert_base_random_24.5k_ptbDiagSubclass.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_subclass_base_random --tta_aggregation=max --n_augs=5

python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_base_random_5.9k_georgia0.pt --downsampling_factor=5 --n_augs=15 --tta --tta_aggregation=max --save_id=georgia0_base_random
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_base_random_6.2k_georgia1.pt --downsampling_factor=5 --n_augs=15 --tta --tta_aggregation=max --save_id=georgia1_base_random
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_base_random_7.1k_georgia2.pt --downsampling_factor=5 --n_augs=15 --tta --tta_aggregation=max --save_id=georgia2_base_random
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_base_random_6.3k_georgia3.pt --downsampling_factor=5 --n_augs=15 --tta --tta_aggregation=max --save_id=georgia3_base_random

python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_base_random_4.15k_chapman0.pt --downsampling_factor=5 --n_augs=9 --tta --tta_aggregation=max --save_id=chapman0_base_random
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_base_random_7.5k_chapman1.pt --downsampling_factor=5 --n_augs=9 --tta --tta_aggregation=max --save_id=chapman1_base_random
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_base_random_5.2k_chapman2.pt --downsampling_factor=5 --n_augs=9 --tta --tta_aggregation=max --save_id=chapman2_base_random
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_base_random_4.35k_chapman3.pt --downsampling_factor=5 --n_augs=9 --tta --tta_aggregation=max --save_id=chapman3_base_random

python test.py /path/to/ptb_test_0.csv . 64 /path/to/hubert_base_random_2.2k_ptb0.pt --downsampling_factor=5 --tta --save_id=ptb0_base_random --tta_aggregation=max
python test.py /path/to/ptb_test_1.csv . 64 /path/to/hubert_base_random_2.15k_ptb1.pt --downsampling_factor=5 --tta --save_id=ptb1_base_random --tta_aggregation=max
python test.py /path/to/ptb_test_2.csv . 64 /path/to/hubert_base_random_2.45k_ptb2.pt --downsampling_factor=5 --tta --save_id=ptb2_base_random --tta_aggregation=max
python test.py /path/to/ptb_test_3.csv . 64 /path/to/hubert_base_random_1.8k_ptb3.pt --downsampling_factor=5 --tta --save_id=ptb3_base_random --tta_aggregation=max

python test.py /path/to/samitrop_test0.csv . 64 /path/to/hubert_base_random_1.3k_samitrop0.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop0_base_random --tta_aggregation=max
python test.py /path/to/samitrop_test1.csv . 64 /path/to/hubert_base_random_1.05k_samitrop1.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop1_base_random --tta_aggregation=max
python test.py /path/to/samitrop_test2.csv . 64 /path/to/hubert_base_random_1.1k_samitrop2.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop2_base_random --tta_aggregation=max
python test.py /path/to/samitrop_test3.csv . 64 /path/to/hubert_base_random_0.75k_samitrop3.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop3_base_random --tta_aggregation=max

python test.py /path/to/cpsc_extra_test_0.csv . 64 /path/to/hubert_base_random_4.55k_cpscExtra0.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=7 --save_id=cpsc_extra0_base_random 
python test.py /path/to/cpsc_extra_test_1.csv . 64 /path/to/hubert_base_random_3.25k_cpscExtra1.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=cpsc_extra1_base_random 
python test.py /path/to/cpsc_extra_test_2.csv . 64 /path/to/hubert_base_random_5.45k_cpscExtra2.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=cpsc_extra2_base_random 
python test.py /path/to/cpsc_extra_test_3.csv . 64 /path/to/hubert_base_random_2.4k_cpscExtra3.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=3 --save_id=cpsc_extra3_base_random 

python test.py /path/to/ribeiro_test_set.csv /path/to/ribeiro_test/ 64 /path/to/hubert_base_random_55k_tnmg.pt --downsampling_factor=5 --save_id=ribeiro_base_random 

python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_random_33k_general.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_base --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_58k_cardiolearning_random_1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_base_1 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_28k_cardiolearning_random_2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_base_2 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_49.5k_cardiolearning_random_3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_base_3 --tta --tta_aggregation=max --n_augs=3

# large model size

python test.py /path/to/ningbo_test.csv .  64 /path/to/hubert_large_random_24k_ningbo0.pt --downsampling_factor=5  --tta  --save_id=ningbo0_large_random --tta_aggregation=max
python test.py /path/to/ningbo_test.csv .  64 /path/to/hubert_large_random_18k_ningbo1.pt --downsampling_factor=5  --tta  --save_id=ningbo1_large_random --tta_aggregation=max
python test.py /path/to/ningbo_test.csv .  64 /path/to/hubert_large_random_18.5k_ningbo2.pt --downsampling_factor=5  --tta --save_id=ningbo2_large_random --tta_aggregation=max
python test.py /path/to/ningbo_test.csv .  64 /path/to/hubert_large_random_19.5k_ningbo3.pt --downsampling_factor=5  --tta --save_id=ningbo3_large_random --tta_aggregation=max

python test.py /path/to/ptb_test_0.csv .  64 /path/to/hubert_large_random_1k_ptb0.pt --downsampling_factor=5  --tta --save_id=ptb0_large_random --tta_aggregation=max
python test.py /path/to/ptb_test_1.csv .  64 /path/to/hubert_large_random_1.6k_ptb1.pt --downsampling_factor=5  --tta --save_id=ptb1_large_random --tta_aggregation=max
python test.py /path/to/ptb_test_2.csv .  64 /path/to/hubert_large_random_1.85k_ptb2.pt --downsampling_factor=5  --tta --save_id=ptb2_large_random --tta_aggregation=max
python test.py /path/to/ptb_test_3.csv .  64 /path/to/hubert_large_random_1.85k_ptb3.pt --downsampling_factor=5  --tta --save_id=ptb3_large_random --tta_aggregation=max

python test.py /path/to/cpsc_test.csv .  64 /path/to/hubert_large_random_2.15k_cpsc0.pt --downsampling_factor=5  --tta --save_id=cpsc0_large_random --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv .  64 /path/to/hubert_large_random_3.15k_cpsc1.pt --downsampling_factor=5  --tta --save_id=cpsc1_large_random --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv .  64 /path/to/hubert_large_random_3.25k_cpsc2.pt --downsampling_factor=5  --tta --save_id=cpsc2_large_random --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv .  64 /path/to/hubert_large_random_4.1k_cpsc3.pt --downsampling_factor=5  --tta --save_id=cpsc3_large_random --tta_aggregation=max --n_augs=5

python test.py /path/to/sph_test.csv .  64 /path/to/hubert_large_random_21k_sph.pt --downsampling_factor=5  --label_start_index=4 --tta --n_augs=5 --save_id=sph_large_random --tta_aggregation=max

python test.py /path/to/hefei_test.csv .  64 /path/to/hubert_large_random_9.5k_hefei4.pt --downsampling_factor=5  --tta --n_augs=5 --save_id=hefei3_large_random --tta_aggregation=max
python test.py /path/to/hefei_test.csv .  64 /path/to/hubert_large_random_14k_hefei0.pt --downsampling_factor=5  --tta --n_augs=5 --save_id=hefei0_large_random --tta_aggregation=max
python test.py /path/to/hefei_test.csv .  64 /path/to/hubert_large_random_18.5k_hefei1.pt --downsampling_factor=5  --tta --n_augs=5 --save_id=hefei1_large_random --tta_aggregation=max
python test.py /path/to/hefei_test.csv .  64 /path/to/hubert_large_random_13k_hefei2.pt --downsampling_factor=5  --tta --n_augs=5 --save_id=hefei2_large_random --tta_aggregation=max

python test.py /path/to/ptb_all_test.csv .  64 /path/to/hubert_large_random_19k_ptbAll.pt --downsampling_factor=5  --label_start_index=4 --tta --save_id=ptb_all_large_random --tta_aggregation=max 
python test.py /path/to/ptb_form_test.csv .  64 /path/to/hubert_large_random_9k_ptbForm.pt --downsampling_factor=5  --label_start_index=4 --tta --save_id=ptb_form_large_random --tta_aggregation=max --n_augs=5
python test.py /path/to/ptb_rhythm_test.csv .  64 /path/to/hubert_large_random_20k_ptbRhythm.pt --downsampling_factor=5  --label_start_index=4 --tta --save_id=ptb_rhythm_large_random --tta_aggregation=max
python test.py /path/to/ptb_diag_test.csv .  64 /path/to/hubert_large_random_20k_ptbDiag.pt --downsampling_factor=5  --label_start_index=4 --tta --save_id=ptb_diag_large_random --tta_aggregation=max
python test.py /path/to/ptb_diag_subclass_test.csv .  64 /path/to/hubert_large_random_8k_ptbDiagSubclass.pt --downsampling_factor=5  --label_start_index=4 --tta --save_id=ptb_diag_subclass_large_random --tta_aggregation=max --n_augs=5
python test.py /path/to/ptb_diag_superclass_test.csv .  64 /path/to/hubert_large_random_16.5k_ptbDiagSuperclass.pt --downsampling_factor=5  --label_start_index=4 --tta --save_id=ptb_diag_superclass_large_random --tta_aggregation=max

python test.py /path/to/ribeiro_test_set.csv /path/to/ribeiro_test/  64 /path/to/hubert_large_random_40k_tnmg.pt --downsampling_factor=5  --save_id=ribeiro_large_random 

python test.py /path/to/samitrop_test0.csv .  64 /path/to/hubert_large_random_0.8k_samitrop0.pt --downsampling_factor=5  --tta --label_start_index=6 --task=multi_class --save_id=samitrop0_large_random --tta_aggregation=max
python test.py /path/to/samitrop_test1.csv .  64 /path/to/hubert_large_random_0.8k_samitrop1.pt --downsampling_factor=5  --tta --label_start_index=6 --task=multi_class --save_id=samitrop1_large_random --tta_aggregation=max
python test.py /path/to/samitrop_test2.csv .  64 /path/to/hubert_large_random_0.7k_samitrop2.pt --downsampling_factor=5  --tta --label_start_index=6 --task=multi_class --save_id=samitrop2_large_random --tta_aggregation=max
python test.py /path/to/samitrop_test3.csv .  64 /path/to/hubert_large_random_0.8k_samitrop3.pt --downsampling_factor=5  --tta --label_start_index=6 --task=multi_class --save_id=samitrop3_large_random --tta_aggregation=max

python test.py /path/to/cpsc_extra_test_0.csv .  64 /path/to/hubert_large_random_2.8k_cpscExtra0.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=7 --save_id=cpsc_extra0_large_random 
python test.py /path/to/cpsc_extra_test_1.csv .  64 /path/to/hubert_large_random_2.85k_cpscExtra1.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=15 --save_id=cpsc_extra1_large_random 
python test.py /path/to/cpsc_extra_test_2.csv .  64 /path/to/hubert_large_random_2.25k_cpscExtra2.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=15 --save_id=cpsc_extra2_large_random 
python test.py /path/to/cpsc_extra_test_3.csv .  64 /path/to/hubert_large_random_3.35k_cpscExtra3.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=3 --save_id=cpsc_extra3_large_random 

python test.py /path/to/georgia_test.csv .  64 /path/to/hubert_large_random_2.95k_georgia0.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=15 --save_id=georgia0_large_random 
python test.py /path/to/georgia_test.csv .  64 /path/to/hubert_large_random_4.6k_georgia1.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=15 --save_id=georgia1_large_random 
python test.py /path/to/georgia_test.csv .  64 /path/to/hubert_large_random_5.8k_georgia2.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=15 --save_id=georgia2_large_random 
python test.py /path/to/georgia_test.csv .  64 /path/to/hubert_large_random_6.15k_georgia3.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=15 --save_id=georgia3_large_random 

python test.py /path/to/chapman_test.csv .  64 /path/to/hubert_large_random_3.9k_chapman0.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=9 --save_id=chapman0_large_random 
python test.py /path/to/chapman_test.csv .  64 /path/to/hubert_large_random_3.4k_chapman1.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=9 --save_id=chapman1_large_random 
python test.py /path/to/chapman_test.csv .  64 /path/to/hubert_large_random_3.95k_chapman2.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=9 --save_id=chapman2_large_random 
python test.py /path/to/chapman_test.csv .  64 /path/to/hubert_large_random_2.8k_chapman3.pt --downsampling_factor=5  --tta --tta_aggregation=max --n_augs=9 --save_id=chapman3_large_random 

python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_random_27.5k_general.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_large --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_12.5k_cardiolearning_random_1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_large_1 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_13.5k_cardiolearning_random_2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_large_2 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_56k_cardiolearning_random_3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_large_3 --tta --tta_aggregation=max --n_augs=3
