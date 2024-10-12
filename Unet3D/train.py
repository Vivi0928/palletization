import torch
import os
import numpy as np
import random
from model import UNet3D
import torch.optim as optim
import torch.nn as nn
from feasible_dataset import FeasibleDataset, get_dataloader
from utils import get_logger, load_checkpoint
from trainer import Trainer

logger = get_logger("TrainingSetup")
seed = 7

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model: UNet3D, loaders, logdir, writer, device):
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, loss_criterion=criterion, 
                      loaders=loaders, max_num_epochs=10, writer=writer, logdir=logdir, device=device)
    train_loss, val_loss, val_iou = trainer.fit()
    return train_loss, val_loss, val_iou

if __name__=="__main__":
    experiment_id = 5
    logger.info(f"experiment_id: {experiment_id}")

    # dataset_path = "../feasible_set/feasible_anno_2"
    dataset_path = "../feasible_anno"
    logger.info(f"loading dataset : {dataset_path}")
    dataset = FeasibleDataset(dataset_path)
    loaders = get_dataloader(dataset=dataset, split=[0.8, 0.1, 0.1], seed=seed)
    logger.info(f"Dataloader prepared! Dataset len: {len(dataset)}")

    model = UNet3D(5, 1, f_maps=8, num_levels=3, final_sigmoid=True).cuda()
    # # Load model state
    # model_path = "./logs/4/checkpoint_60.pytorch"
    # load_checkpoint(model_path, model)

    train(model=model, loaders=loaders, experiment_id=experiment_id)
