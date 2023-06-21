### guardare se piacciono tutti i dati ###
import torch
from Configs import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from ECGDatasetSelfSupervised import ECGDatasetSelfSupervised

from loguru import logger #todo da importare


print("creating datasets for self-supervised pre-training...")
train_set = ECGDatasetSelfSupervised("./train_self_supervised_processed.csv", "./train_self_supervised")
val_set = ECGDatasetSelfSupervised("./val_self_supervised_processed.csv", "./val_self_supervised")
test_set = ECGDatasetSelfSupervised("./test_self_supervised_processed.csv", "./test_self_supervised")

print("creating dataloaders to generata batches...")
# no shuffle if False
train_dl = DataLoader(train_set, batch_size=512, shuffle=(not False), collate_fn=lambda x: x, num_workers=4)
val_dl = DataLoader(val_set, batch_size=512, shuffle=(not False), collate_fn=lambda x: x, num_workers=4)
test_dl = DataLoader(test_set, batch_size=512, shuffle=True, collate_fn=lambda x: x, num_workers=4)

  
for i, batch in tqdm(enumerate(train_dl), total=len(val_dl)):
    batch_ecg_data, batch_age, batch_sex = tuple(zip(*batch))

    batch_age = torch.Tensor(batch_age).cuda().to(dtype=torch.float) # (batch_size, 1)
    batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)
    batch_ecg_data = torch.stack(batch_ecg_data, dim=0).cuda().to(dtype=torch.float)

exit()
### da rimuovere quanto sopra ###