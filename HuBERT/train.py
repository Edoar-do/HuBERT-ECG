import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from loguru import logger
import argparse
from tqdm import tqdm
from dataset import ECGDatasetSelfSupervised as ECGDataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from hubert_ecg import Hubert
import torch.optim as optim
from utils import load_checkpoint

BACKEND = 'nccl'
INIT_METHOD = "tcp://localhost:????"
EPS = 1E-06

SELF_SUPERVISED_MODEL_CKPT_PATH = "/data/ECG_AF/ECG_pretraining/models/checkpoints/self-supervised"

def train(rank, world_size, args):

    wandb.init(entity="cardi-ai", project="ECG-pretraining", config=wandb.config, group="self-supervised")

    distributed = world_size > 1

    dist.init_process_group(
        BACKEND,
        rank=rank,
        world_size=world_size,
        init_method=INIT_METHOD
    )

    #configs
    lr = wandb.config['lr'] # 1e-5
    betas = (wandb.config['beta1'], wandb.config['beta2']) # (0.9, 0.98)
    batch_size = wandb.config['batch_size'] # 512 --> remember to sert bs so to saturate gpu memory
    weight_decay = wandb.config['weight_deacay'] # 0.01
    #the following can be reduced for exploratory training phases
    first_iter_steps = wandb.config['first_iter_steps'] # 250000
    second_iter_steps = wandb.config['second_iter_steps'] #400000
    third_iter_steps = wandb.config['third_iter_steps'] # 100000
    val_interval = wandb.config['val_interval'] # 1000

    #assertions necessary to validate untill the end of each training phase
    assert first_iter_steps % val_interval == 0
    assert second_iter_steps % val_interval == 0
    assert third_iter_steps % val_interval == 0
    
    #device = torch.device("cuda")
    dtype = torch.float

    if distributed:
        dist.init_process_group(
            backend=BACKEND,
            init_method=INIT_METHOD,
            rank=rank,
            world_size=world_size)

    if args.resume:
        hubert = torch.jit.load(args.load_path)
        tmp = torch.load(args.load_path[:-3] + ".pth")
        global_step, best_val_loss = tmp['global_step'], tmp['best_val_loss']
    else:
        hubert = Hubert() #default args
        global_step = 0
        best_val_loss = float("inf")

    hubert.to(rank)
    hubert = DDP(hubert, device_ids=[rank])

    optimizer = optim.AdamW(
        hubert.parameters(),
        lr=lr,
        betas=betas,
        eps=EPS,
        weight_decay=weight_decay,
    )

    scaler = amp.GradScaler()

    ### FIRST TRAINING ITERATION ###

    train_iteration = args.train_iteration

    train_set = ECGDataset(
        path_to_dataset_csv="/data/ECG_AF/train_self_supervised_processed.csv",
        ecg_dir_path="/data/ECG_AF/train_self_supervised",
        train_iteration=train_iteration,
        pretrain=True
        )

    val_set = ECGDataset(
        path_to_dataset_csv="/data/ECG_AF/val_self_supervised_processed.csv",
        ecg_dir_path="/data/ECG_AF/val_self_supervised",
        train_iteration=train_iteration,
        pretrain=True
        )

    train_dl = DataLoader(
        train_set,
        collate_fn=lambda x: x,
        num_workers=4,
        batch_size=batch_size,
        shuffle=(not distributed),
        sampler=(DistributedSampler(train_set) if distributed else None),
        pin_memory=True
        )

    val_dl = DataLoader(
        val_set,
        collate_fn=lambda x: x,
        num_workers=4,
        batch_size=batch_size,
        shuffle=(not distributed),
        pin_memory=True
        )


    if train_iteration == 1:
        epochs = first_iter_steps // len(train_dl) + 1
    elif train_iteration == 2:
        epochs = second_iter_steps // len(train_dl) + 1
    else:
        epochs = third_iter_steps // len(train_dl) + 1

    start_epoch = global_step // len(train_dl) + 1
    
    for epoch in range(start_epoch, epochs):

        hubert.train()
        logger.info(f"Epoch {epoch+1}/{epochs}")

        epoch_losses = []
        #epoch_accuracies = []

        train_dl.sampler.set_epoch(epoch)

        for batch in tqdm(train_dl, total=len(train_dl)):

            global_step += 1

            optimizer.zero_grad()

            batch_ecg, batch_labels = tuple(zip(*batch)) #2 tuples of tensors
            # batch_ecg = torch.stack(batch_ecg).to(rank)
            # batch_labels = torch.stack(batch_labels).to(rank)
            batch_ecg = torch.stack(batch_ecg).to(rank, dtype=dtype)
            batch_labels = torch.stack(batch_labels).to(rank, dtype=dtype)

            with amp.autocast():
                batch_logits, batch_mask = hubert(batch_ecg)

                length = min(batch_mask.size[-1], batch_labels.size[-1])
                batch_logits = batch_logits[:, :length, :]
                batch_labels = batch_labels[:, :length]
                batch_mask = batch_mask[:, :length]

                masked_loss = F.cross_entropy(batch_logits[batch_mask], batch_labels[batch_mask])
                unmasked_loss = F.cross_entropy(batch_logits[~batch_mask], batch_labels[~batch_mask])
                loss = args.alpha * masked_loss +  (1 - args.alpha) * unmasked_loss


            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(hubert.parameters(), 10.)

            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

            # accuracy = (batch_logits[mask].argmax(dim=-1) == batch_labels[mask]).float().mean()
            # epoch_accuracies.append(accuracy.item())

            ### VALIDATION LOOP EVERY val_interval STEPS + LOGGING ###

            if global_step % val_interval == 0:

                hubert.eval()
                val_loss = []
                # val_accuracy = []

                for batch in tqdm(val_dl, total=len(val_dl)):
                    batch_ecg, batch_labels = tuple(zip(*batch))
                    batch_ecg = torch.stack(batch_ecg).to(rank, dtype=dtype)
                    batch_labels = torch.stack(batch_labels).to(rank, dtype=dtype)

                    with torch.no_grad():
                        batch_logits, batch_mask = hubert(batch_ecg)

                        length = min(batch_mask.size[-1], batch_labels.size[-1])
                        batch_logits = batch_logits[:, :length, :]
                        batch_labels = batch_labels[:, :length]
                        batch_mask = batch_mask[:, :length]

                        loss = F.cross_entropy(batch_logits[mask], batch_labels[mask])
                    
                    val_loss.append(loss.item())

                    #accuracy = (batch_logits[mask].argmax(dim=-1) == batch_labels[mask]).float().mean()
                    #val_accuracy.append(accuracy.item())
                val_loss = np.mean(val_loss)
                logger.info(f"Step {global_step}: train_loss_{train_iteration}: {np.mean(epoch_losses)}, val_loss_{train_iteration}: {val_loss}")
                wandb.log({f"first_train_{train_iteration}": np.mean(epoch_losses), f"val_loss_{train_iteration}": np.mean(val_loss)})
                # wandb.log({f"train_accuracy_{train_iteration}": np.mean(epoch_accuracies), f"val_accuracy_{train_iteration}": np.mean(val_accuracy)})

                hubert.train()

                ### EARLY STOPPING ###
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_name = f"hubert_{train_iteration}_iteration_{global_step}.pt"
                    model_scripted = torch.jit.script(hubert)
                    model_scripted.save(SELF_SUPERVISED_MODEL_CKPT_PATH + checkpoint_name)
                    torch.save({"global_step":global_step, "best_val_loss":best_val_loss},
                     SELF_SUPERVISED_MODEL_CKPT_PATH + checkpoint_name[:-3] + ".pth")
                    logger.info(f"New best (best_val_loss = {best_val_loss}) model saved at step {global_step}")

    ### END OF TRAINING ITERATION ###

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Hubert-ECG")

    #1, 2 or 3
    parser.add_argument(
        "train_iteration",
        help="1, 2 or 3 HuBERT train iteration",
        type=int,
    )

    # in [0, 1]
    parser.add_argument(
        "alpha",
        help="The alpha value for the weighted loss function. \n alpha=1 suggested when train_iteration=1",
        type=float
    )

    parser.add_argument(
        "--resume",
        action="store_false"
        help="whether to resume training or not",
    )

    # ECG_HUBERT_FEATURES_PATH = "/data/ECG_AF/hubert_features"
    # ENCODER_6_FEATURES_PATH = "/data/ECG_AF/ECG_pretraining/HuBERT/encoder_6_features"
    # ENCODER_9_FEATURES_PATH = "/data/ECG_AF/ECG_pretraining/HuBERT/encoder_9_features"

    parser.add_argument(
        "--load_path",
        help="path to load model from, if it has to resume training",
        type=str
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    
    if not torch.cuda.is_available():
        logger.error("CUDA not available. CPU training not supported")
        exit(1)
    if args.alpha > 1.0 or args.alpha < 0.0:
        logger.error("alpha parameter should be in [0,1] interval")

    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
        

    