#!/bin/bash

# more information with python test.py --help

python test.py /path/to/ribeiro_test_set.csv /path/to/ribeiro_test/ 64 /path/to/hubert_small_55k_tnmg.pt --downsampling_factor=5 --save_id=ribeiro_small 


python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_small_16.5k_ningbo0.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo0_small --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_small_18.5k_ningbo1.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo1_small --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_small_15.5k_ningbo2.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo2_small --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_small_13k_ningbo3.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo3_small --tta_aggregation=max


python test.py /path/to/ptb_test_0.csv . 64 /path/to/hubert_small_3.3k_ptb0.pt --downsampling_factor=5 --tta --save_id=ptb0_small --tta_aggregation=max
python test.py /path/to/ptb_test_1.csv . 64 /path/to/hubert_small_3.1k_ptb1.pt --downsampling_factor=5 --tta --save_id=ptb1_small --tta_aggregation=max
python test.py /path/to/ptb_test_2.csv . 64 /path/to/hubert_small_3.75k_ptb2.pt --downsampling_factor=5 --tta --save_id=ptb2_small --tta_aggregation=max
python test.py /path/to/ptb_test_3.csv . 64 /path/to/hubert_small_3.1k_ptb3.pt --downsampling_factor=5 --tta --save_id=ptb3_small --tta_aggregation=max


python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_small_5.95k_cpsc0.pt --downsampling_factor=5 --tta --save_id=cpsc0_small --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_small_6.05k_cpsc1.pt --downsampling_factor=5 --tta --save_id=cpsc1_small --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_small_5.2k_cpsc2.pt --downsampling_factor=5 --tta --save_id=cpsc2_small --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_small_6.3k_cpsc3.pt --downsampling_factor=5 --tta --save_id=cpsc3_small --tta_aggregation=max --n_augs=5


python test.py /path/to/sph_test.csv . 64 /path/to/hubert_small_10.5k_sph.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=sph_small --tta_aggregation=max --n_augs=5


python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_small_9.5k_hefei4.pt --downsampling_factor=5 --tta --save_id=hefei3_small --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_small_10k_hefei0.pt --downsampling_factor=5 --tta --save_id=hefei0_small --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_small_12.5k_hefei1.pt --downsampling_factor=5 --tta --save_id=hefei1_small --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_small_8.5k_hefei2.pt --downsampling_factor=5 --tta --save_id=hefei2_small --tta_aggregation=max --n_augs=5


python test.py /path/to/ptb_all_test.csv . 64 /path/to/hubert_small_12.5k_ptbAll.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_all_small --tta_aggregation=max
python test.py /path/to/ptb_form_test.csv . 64 /path/to/hubert_small_6.5k_ptbForm.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_form_small --tta_aggregation=max --n_augs=3
python test.py /path/to/ptb_rhythm_test.csv . 64 /path/to/hubert_small_7.5k_ptbRtythm.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_rhythm_small --tta_aggregation=max
python test.py /path/to/ptb_diag_superclass_test.csv . 64 /path/to/hubert_small_7k_ptbDiagSuperclass.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_superclass_small --tta_aggregation=max
python test.py /path/to/ptb_diag_subclass_test.csv . 64 /path/to/hubert_small_8k_ptbDiagSubclass.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_subclass_small --tta_aggregation=max --n_augs=5
python test.py /path/to/ptb_diag_test.csv . 64 /path/to/hubert_small_11k_ptbDiag.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_small --tta_aggregation=max


python test.py /path/to/samitrop_test0.csv . 64 /path/to/hubert_small_850_samitrop0.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=6 --task=multi_class --label_start_index=6
python test.py /path/to/samitrop_test1.csv . 64 /path/to/hubert_small_500_samitrop1.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=6 --task=multi_class --label_start_index=6
python test.py /path/to/samitrop_test2.csv . 64 /path/to/hubert_small_1250_samitrop2.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=6 --task=multi_class --label_start_index=6
python test.py /path/to/samitrop_test3.csv . 64 /path/to/hubert_small_1500_samitrop3.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=6 --task=multi_class --label_start_index=6

python test.py /path/to/cpsc_extra_test_0.csv . 64 /path/to/hubert_small_9.6k_cpscExtra0.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=13 --save_id=cpsc_extra0_small 
python test.py /path/to/cpsc_extra_test_1.csv . 64 /path/to/hubert_small_8.7k_cpscExtra1.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=cpsc_extra1_small 
python test.py /path/to/cpsc_extra_test_2.csv . 64 /path/to/hubert_small_9.4k_cpscExtra2.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=11 --save_id=cpsc_extra2_small 
python test.py /path/to/cpsc_extra_test_3.csv . 64 /path/to/hubert_small_9.1k_cpscExtra3.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=7 --save_id=cpsc_extra3_small 


python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_small_4.7k_georgia0.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia0_small 
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_small_5.6k_georgia1.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia1_small 
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_small_5.4k_georgia2.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia2_small 
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_small_5.3k_georgia3.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia3_small 

 
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_small_5.75k_chapman0.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman0_small 
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_small_4.65k_chapman1.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman1_small 
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_small_5k_chapman2.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman2_small 
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_small_4.8k_chapman3.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman3_small 

 
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_30.5k_cardiolearning0.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_base --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_28k_cardiolearning1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_small_1 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_31k_cardiolearning2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_small_2 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_28.5k_cardiolearning3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_small_3 --tta --tta_aggregation=max --n_augs=3

python test.py /path/to/ribeiro_test_set.csv /path/to/ribeiro_test/ 64 /path/to/hubert_base_45k_tnmg.pt --downsampling_factor=5 --save_id=ribeiro_base 

python test.py /path/to/ptb_all_test.csv . 64 /path/to/hubert_base_9k_ptbAll.pt --downsampling_factor=5 --label_start_index=4 --tta --tta_aggregation=max --save_id=ptb_all_base
python test.py /path/to/ptb_form_test.csv . 64 /path/to/hubert_base_6.5k_ptbForm.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_form_base --tta_aggregation=max --n_augs=5
python test.py /path/to/ptb_rhythm_test.csv . 64 /path/to/hubert_base_8.5k_ptbRhythm.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_rhythm_base --tta_aggregation=max
python test.py /path/to/ptb_diag_test.csv . 64 /path/to/hubert_base_9.5k_ptbDiag.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_base --tta_aggregation=max
python test.py /path/to/ptb_diag_superclass_test.csv . 64 /path/to/hubert_base_6k_ptbDiagSuperclass.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_superclass_base --tta_aggregation=max
python test.py /path/to/ptb_diag_subclass_test.csv . 64 /path/to/hubert_base_6.5k_ptbDiagSubclass.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_subclass_base --tta_aggregation=max --n_augs=5


python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_base_6.5k_hefei0.pt --downsampling_factor=5 --tta --save_id=hefei0_base --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_base_5.5k_hefei1.pt --downsampling_factor=5 --tta --save_id=hefei1_base --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_base_5.5k_hefei2.pt --downsampling_factor=5 --tta --save_id=hefei2_base --tta_aggregation=max --n_augs=5
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_base_6k_hefei3.pt --downsampling_factor=5 --tta --save_id=hefei3_base --tta_aggregation=max --n_augs=5


python test.py /path/to/sph_test.csv . 64 /path/to/hubert_base_5.5k_sph.pt --downsampling_factor=5 --label_start_index=4 --save_id=sph_base --tta --tta_aggregation=max --n_augs=5


python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_base_5k_cpsc0.pt --downsampling_factor=5 --tta --save_id=cpsc0_base --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_base_4.15k_cpsc1.pt --downsampling_factor=5 --tta --save_id=cpsc1_base --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_base_4.9k_cpsc2.pt --downsampling_factor=5 --tta --save_id=cpsc2_base --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_base_5.15k_cpsc3.pt --downsampling_factor=5 --tta --save_id=cpsc3_base --tta_aggregation=max --n_augs=5


python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_base_15k_ningbo0.pt --downsampling_factor=5 --tta --n_augs=3 --save_id=ningbo0_base --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_base_14.5k_ningbo1.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo1_base --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_base_12k_ningbo2.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo2_base --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_base_9k_ningbo3.pt --downsampling_factor=5 --tta --n_augs=7 --save_id=ningbo3_base --tta_aggregation=max


python test.py /path/to/ptb_test_0.csv . 64 /path/to/hubert_base_2.35k_ptb0.pt --downsampling_factor=5 --tta --save_id=ptb0_base --tta_aggregation=max
python test.py /path/to/ptb_test_1.csv . 64 /path/to/hubert_base_2.65k_ptb1.pt --downsampling_factor=5 --tta --save_id=ptb1_base --tta_aggregation=max
python test.py /path/to/ptb_test_2.csv . 64 /path/to/hubert_base_2.4k_ptb2.pt --downsampling_factor=5 --tta --save_id=ptb2_base --tta_aggregation=max
python test.py /path/to/ptb_test_3.csv . 64 /path/to/hubert_base_2.1k_ptb3.pt --downsampling_factor=5 --tta --save_id=ptb3_base --tta_aggregation=max


python test.py /path/to/samitrop_test0.csv . 64 /path/to/hubert_base_2.55k_samitrop0.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop0_base --tta_aggregation=max
python test.py /path/to/samitrop_test1.csv . 64 /path/to/hubert_base_2.6k_samitrop1.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop1_base --tta_aggregation=max
python test.py /path/to/samitrop_test2.csv . 64 /path/to/hubert_base_2.5k_samitrop2.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop2_base --tta_aggregation=max
python test.py /path/to/samitrop_test3.csv . 64 /path/to/hubert_base_2.3k_samitrop3.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop3_base --tta_aggregation=max

python test.py /path/to/cpsc_extra_test_0.csv . 64 /path/to/hubert_base_7k_cpscExtra0.pt --downsampling_factor=5 --tta --tta_aggregation=max  --n_augs=7  --save_id=cpsc_extra0_base 
python test.py /path/to/cpsc_extra_test_1.csv . 64 /path/to/hubert_base_6.7k_cpscExtra1.pt --downsampling_factor=5 --tta --tta_aggregation=max  --n_augs=9 --save_id=cpsc_extra1_base 
python test.py /path/to/cpsc_extra_test_2.csv . 64 /path/to/hubert_base_6.8k_cpscExtra2.pt --downsampling_factor=5 --tta --tta_aggregation=max  --n_augs=3 --save_id=cpsc_extra2_base 
python test.py /path/to/cpsc_extra_test_3.csv . 64 /path/to/hubert_base_6.55k_cpscExtra3.pt --downsampling_factor=5 --tta --tta_aggregation=max  --n_augs=7 --save_id=cpsc_extra3_base 


python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_base_5k_georgia0.pt --downsampling_factor=5 --n_augs=15 --tta --tta_aggregation=max --save_id=georgia0_base
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_base_5.05k_georgia1.pt --downsampling_factor=5 --n_augs=15 --tta --tta_aggregation=max --save_id=georgia1_base
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_base_5.7k_georgia2.pt --downsampling_factor=5 --n_augs=15 --tta --tta_aggregation=max --save_id=georgia2_base
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_base_5.95k_georgia3.pt --downsampling_factor=5 --n_augs=15 --tta --tta_aggregation=max --save_id=georgia3_base


python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_base_4.3k_chapman0.pt --downsampling_factor=5 --n_augs=9 --tta --tta_aggregation=max --save_id=chapman0_base
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_base_4.7k_chapman1.pt --downsampling_factor=5 --n_augs=9 --tta --tta_aggregation=max --save_id=chapman1_base
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_base_4.05k_chapman2.pt --downsampling_factor=5 --n_augs=9 --tta --tta_aggregation=max --save_id=chapman2_base
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_base_4.65k_chapman3.pt --downsampling_factor=5 --n_augs=9 --tta --tta_aggregation=max --save_id=chapman3_base


python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_30.5k_cardiolearning0.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_base_0 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_25.5k_cardiolearning1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_base_1 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_27.5k_cardiolearning2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_large_2 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_29k_cardiolearning3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_base_3 --tta --tta_aggregation=max --n_augs=3


python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_large_9k_ningbo0.pt --downsampling_factor=5 --tta --save_id=ningbo0_large --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_large_12k_ningbo1.pt --downsampling_factor=5 --tta --save_id=ningbo1_large --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_large_8k_ningbo2.pt --downsampling_factor=5 --tta --save_id=ningbo2_large --tta_aggregation=max
python test.py /path/to/ningbo_test.csv . 64 /path/to/hubert_large_11.5k_ningbo3.pt --downsampling_factor=5 --tta --save_id=ningbo3_large --tta_aggregation=max


python test.py /path/to/ptb_test_0.csv . 64 /path/to/hubert_large_1.8k_ptb0.pt --downsampling_factor=5 --tta --save_id=ptb0_large --tta_aggregation=max
python test.py /path/to/ptb_test_1.csv . 64 /path/to/hubert_large_1.85k_ptb1.pt --downsampling_factor=5 --tta --save_id=ptb1_large --tta_aggregation=max
python test.py /path/to/ptb_test_2.csv . 64 /path/to/hubert_large_1.95k_ptb2.pt --downsampling_factor=5 --tta --save_id=ptb2_large --tta_aggregation=max
python test.py /path/to/ptb_test_3.csv . 64 /path/to/hubert_large_1.85k_ptb3.pt --downsampling_factor=5 --tta --save_id=ptb3_large --tta_aggregation=max


python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_large_3.5k_cpsc0.pt --downsampling_factor=5 --tta --save_id=cpsc0_large --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_large_2.75k_cpsc1.pt --downsampling_factor=5 --tta --save_id=cpsc1_large --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_large_3.2k_cpsc2.pt --downsampling_factor=5 --tta --save_id=cpsc2_large --tta_aggregation=max --n_augs=5
python test.py /path/to/cpsc_test.csv . 64 /path/to/hubert_large_3.7k_cpsc3.pt --downsampling_factor=5 --tta --save_id=cpsc3_large --tta_aggregation=max --n_augs=5


python test.py /path/to/sph_test.csv . 64 /path/to/hubert_large_7.5k_sph.pt --downsampling_factor=5 --label_start_index=4 --tta --n_augs=5 --save_id=sph_large --tta_aggregation=max


python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_large_5k_hefei4.pt --downsampling_factor=5 --tta --n_augs=5 --save_id=hefei3_large --tta_aggregation=max
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_large_5.5k_hefei0.pt --downsampling_factor=5 --tta --n_augs=5 --save_id=hefei0_large --tta_aggregation=max
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_large_6k_hefei1.pt --downsampling_factor=5 --tta --n_augs=5 --save_id=hefei1_large --tta_aggregation=max
python test.py /path/to/hefei_test.csv . 64 /path/to/hubert_large_4.5k_hefei2.pt --downsampling_factor=5 --tta --n_augs=5 --save_id=hefei2_large --tta_aggregation=max


python test.py /path/to/ptb_all_test.csv . 64 /path/to/hubert_large_8.5k_ptbAll.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_all_large --tta_aggregation=max 
python test.py /path/to/ptb_form_test.csv . 64 /path/to/hubert_large_5.5k_ptbForm.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_form_large --tta_aggregation=max --n_augs=5
python test.py /path/to/ptb_rhythm_test.csv . 64 /path/to/hubert_large_5.5k_ptbRhythm.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_rhythm_large --tta_aggregation=max
python test.py /path/to/ptb_diag_test.csv . 64 /path/to/hubert_large_7.5k_ptbDiag.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_large --tta_aggregation=max
python test.py /path/to/ptb_diag_subclass_test.csv . 64 /path/to/hubert_large_8k_ptbDiagSubclass.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_subclass_large --tta_aggregation=max --n_augs=5
python test.py /path/to/ptb_diag_superclass_test.csv . 64 /path/to/hubert_large_4.5k_ptbDiagSuperclass.pt --downsampling_factor=5 --label_start_index=4 --tta --save_id=ptb_diag_superclass_large --tta_aggregation=max


python test.py /path/to/ribeiro_test_set.csv /path/to/ribeiro_test/ 64 /path/to/hubert_large_50k_tnmg.pt --downsampling_factor=5 --save_id=ribeiro_large 


python test.py /path/to/samitrop_test0.csv . 64 /path/to/hubert_large_1.85k_samitrop0.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop0_large --tta_aggregation=max
python test.py /path/to/samitrop_test1.csv . 64 /path/to/hubert_large_1.6k_samitrop1.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop1_large --tta_aggregation=max
python test.py /path/to/samitrop_test2.csv . 64 /path/to/hubert_large_1.9k_samitrop2.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop2_large --tta_aggregation=max
python test.py /path/to/samitrop_test3.csv . 64 /path/to/hubert_large_2.05k_samitrop3.pt --downsampling_factor=5 --tta --label_start_index=6 --task=multi_class --save_id=samitrop3_large --tta_aggregation=max


python test.py /path/to/cpsc_extra_test_0.csv . 64 /path/to/hubert_large_5.25k_cpscExtra0.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=7 --save_id=cpsc_extra0_large 
python test.py /path/to/cpsc_extra_test_1.csv . 64 /path/to/hubert_large_5.15k_cpscExtra1.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=cpsc_extra1_large 
python test.py /path/to/cpsc_extra_test_2.csv . 64 /path/to/hubert_large_5.7k_cpscExtra2.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=cpsc_extra2_large 
python test.py /path/to/cpsc_extra_test_3.csv . 64 /path/to/hubert_large_5.45k_cpscExtra3.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=3 --save_id=cpsc_extra3_large 

  
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_large_2.45k_georgia0.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia0_large 
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_large_2.85k_georgia1.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia1_large 
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_large_3.05k_georgia2.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia2_large 
python test.py /path/to/georgia_test.csv . 64 /path/to/hubert_large_3.2k_georgia3.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=15 --save_id=georgia3_large 


python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_large_3.45k_chapman0.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman0_large 
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_large_3.6k_chapman1.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman1_large 
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_large_3.35k_chapman2.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman2_large 
python test.py /path/to/chapman_test.csv . 64 /path/to/hubert_large_3.35k_chapman3.pt --downsampling_factor=5 --tta --tta_aggregation=max --n_augs=9 --save_id=chapman3_large 


python test.py /path/to/super_test_set_prova.csv . 64 /path/to/hubert_large_36.5k_cardiolearning_0.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_large_0 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/super_test_set_prova.csv . 64 /path/to/hubert_large_52.5k_cardiolearning_1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_large_1 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/super_test_set_prova.csv . 64 /path/to/hubert_large_26.5k_cardiolearning_2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_base_2 --tta --tta_aggregation=max --n_augs=3
python test.py /path/to/super_test_set_prova.csv . 64 /path/to/hubert_large_12k_cardiolearning_3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_large_3 --tta --tta_aggregation=max --n_augs=3


