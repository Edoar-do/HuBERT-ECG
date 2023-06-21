# Configs.py

import json
import torch

def get_configs(path_to_configs_json):
    configs = json.load(open(path_to_configs_json))
    configs['n_patches'] = configs['n_leads'] * configs['window'] // (configs['patch_height'] * configs['patch_width'])
    patch_size = (configs['patch_height'], configs['patch_width'])
    configs['mask_token'] = torch.Tensor([([-1] * (patch_size[1]//2)) + ([1] * (patch_size[1]//2))] * patch_size[0]).to(dtype=torch.float).cuda()
    configs['patch_size'] = patch_size
    return configs

# group0 = ['NSR']
# group1 = ['AAR', 'AED', 'ARH']
# group2 = ["AMI", "AMIs", "AnMIs", "AnMI", "CHD", "CMI", "ERe", "HTV", "IIs", "ISTD", "LIs",
#          "MI", "MIs", "NSSTTA", "OldMI", "STC", "STD", "STE", "STIAb", "TAb", "TInv"]
# group3 = ["AB", "AFAFL", "ATach", "BPAC", "BTS", "CAF", "CCR", "CVCL/CCVCL", "CD", "CR", "PAF", "PSVT", "PWC", 
#           "RAF", "SAAWR", "WAP", "AF", "AFL", "PAC", "SVPB", "+ROT", "-ROT"]
# group4 = ["AVB", "AVD", "AVJR", "CHB", "CIAHB", "DIB", "IIAVB", "IIAVBII", "ILBBB", "IR", "JE", "LPFB", "MoI", "SAB", "SARR", "SND", "VEsR",
#            "BBB", "Brady", "CLBBB", "CRBBB", "IAVB", "IRBBB", "LAnFB", "LBBB", "LPR", "RBBB", "SB"]
# group5 = ["AVNRT", "AVRT", "SPRI", "SVB", "SVT", "VPEx", "WPW", "SA", "STach"]
# group6 = ['abQRS','AH','BRU','FQRS','HF','HVD','ICA','LAA','LAE','LAH','LVH','LVHV','LVS','PPW','RAAb','RAb','RAH',
#           'RAHV','RVH','SQT','TPW','UAb','VH','LAD','LQRSV','NSIVCB','PRWP','LQT','QAb','RAD']
# group7 = ['AP', 'VPP', 'PR']
# group8 = ['AIVR','AJR','FB','JPC','JTach','PVT','VBig','VEB','VEsB','VF','VFL','VPVC','VTach','VTrig','PVC','VPB']

# n_classes = 9

# target_fs = 500
# filter_bandwidth = [0.05, 47]
# window = 10*target_fs
# patch_size = (2, 1000)
# patch_height, patch_width = patch_size
# mask_perc = 0.7
# mask_token = [([-1] * (patch_size[1]//2)) + ([1] * (patch_size[1]//2))] * patch_size[0]
# d_model = 128
# cls_token = [0] * d_model 
# n_leads = 12
# n_patches = n_leads * window // (patch_height * patch_width)
# epochs = 1
# patience = 5
# n_encoder_layers = 4
# warmup = 4000

#available cuda

pin_memory = False


