from feasible_dataset import get_dataloader
from model import UNet3D
from trainer import eval
import torch
import numpy as np
import utils
from utils import Heuri

logger = utils.get_logger('UNet3DTest')

if __name__ == "__main__":
    # load data
    model = Heuri(pallet_size_discrete=(24,40), max_pallet_height=20)
    loader = get_dataloader(split=[0.8, 0.1, 0.1], seed=7)
    logger.info("Dataloader prepared!")

    # Create the model
    model = UNet3D(5, 1, f_maps=8, num_levels=3, final_sigmoid=True).cuda()

    # Load model state
    model_path = "./logs/3/checkpoint.pytorch"
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)

    # model = Heuri(pallet_size_discrete=(24,40), max_pallet_height=20)
    # Eval on test set
    iou = eval(model, loader=loader['Test'], gpu_inference=True)
    logger.info(f"IoU Score: {iou}")