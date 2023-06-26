### guardare se piacciono tutti i dati ###
import torch
from Configs import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from ECGDatasetSelfSupervised import ECGDatasetSelfSupervised

from loguru import logger


logger.info("creating datasets for self-supervised pre-training...")
train_set = ECGDatasetSelfSupervised("./train_self_supervised_processed.csv", "./train_self_supervised", reduced=False)


logger.info("creating dataloaders to generate batches...")
# no shuffle if False
train_dl = DataLoader(train_set, batch_size=512, shuffle=(not False), collate_fn=lambda x: x, num_workers=4)


for i, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
    batch_ecg_data, batch_age, batch_sex = tuple(zip(*batch))

    batch_age = torch.Tensor(batch_age) # (batch_size, 1)
    batch_sex = torch.Tensor(batch_sex) # (batch_size, 1)
    batch_ecg_data = torch.Tensor(batch_ecg_data) #(batch_size, 1) full of nan in case of ValueError, (batch_size, 12, 5000) when all's fine

logger.info("End of data validation")
### da rimuovere quanto sopra ###
