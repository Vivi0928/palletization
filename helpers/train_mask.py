import numpy as np
import os, torch, sys, time
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

from generate_annotation import imap_gen, GenAnno
sys.path.append("Unet3D")
import Unet3D.train
import Unet3D.model
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split

class FeasibleDataset(Dataset):
    def __init__(self, max_data_count):
        self.max_data_count = max_data_count
        self.data_list = deque(maxlen=max_data_count)
        self.mean = np.array([0.21183236, 6.6072526, 5.9993353, 5.5127134, 2.9000595], dtype=np.float32)
        self.std = np.array([9.9945396e-01, 9.1979831e-01, 6.6459063e-04, 8.5840923e-01, 2.4248180e+00], dtype=np.float32)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = (self.data_list[idx][0] - self.mean[:, None, None, None]) / self.std[:, None, None, None]
        label = self.data_list[idx][1]
        return data, label
    
    def update(self, new_data):
        self.data_list.extend(new_data)
        # self.update_mean_std()

    def update_mean_std(self):
        # Stack all images from data_list
        all_images = np.stack([data[0] for data in self.data_list], axis=0)

        # Calculate mean and std across the dataset
        self.mean = all_images.mean(axis=(0, 2, 3, 4))  # Mean per channel
        self.std = all_images.std(axis=(0, 2, 3, 4))    # Std per channel
        print(f"mean: {self.mean}")
        print(f"std: {self.std}")


def get_dataloader(dataset: Dataset, split:list, seed, num_workers=2):
    generator = torch.Generator().manual_seed(seed)
    trainset, testset, valset = random_split(dataset, split, generator=generator)
    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=1024, shuffle=False, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=1024, shuffle=False, num_workers=num_workers)
    return {"Train":trainloader, "Test":testloader, "Val":valloader}


class TrainFeasibleMaskCallback(BaseCallback):
    def __init__(self, pallet_size_discrete, max_pallet_height, logdir, verbose=1):
        super(TrainFeasibleMaskCallback, self).__init__(verbose)
        self.n_gen = 20
        self.n_epoch = 0
        self.max_data_count = 16000
        # self.generators = get_generators(pallet_size_discrete, max_pallet_height, self.n_gen)
        self.dataset = FeasibleDataset(max_data_count=self.max_data_count)
        self.logdir = os.path.join(logdir, "nn_training")
        self.writer = SummaryWriter(self.logdir)

        self.device = torch.device("cuda:1")
        self.unet_model = Unet3D.model.UNet3D(5, 1, f_maps=8, num_levels=3, final_sigmoid=True).to(self.device)
    
    def _on_step(self) -> bool:
        # 遍历所有环境的 record_data
        for env_info in self.locals['infos']:
            record_data = env_info.get('record_data', None)
            if record_data is not None:
                box_num = len(record_data["pallet_config"])
                terminate_reason = env_info.get('termination_reason', -1)
                if terminate_reason == 2:
                    GenAnno.data_list.append(record_data)
                else:
                    if np.random.rand() < 0.1 and box_num >= 1:
                        GenAnno.data_list.append(record_data)
        
        return True
    
    def _on_rollout_end(self):
        self.n_epoch += 1

        self.update_dataset()
        
        self.train_unet()
        self.update_env()


    def update_dataset(self):
        print(f"Gennerate anno count: {len(GenAnno.data_list)}")
        t = time.time()
        data_list = imap_gen()
        GenAnno.data_list.clear()
        self.dataset.update(data_list)
        print(f"Gennerate anno time cost: {time.time() - t}")

    def train_unet(self):
        t = time.time()
        loaders = get_dataloader(dataset=self.dataset, split=[0.8, 0.0, 0.2], seed=7, num_workers=2)
        train_loss, val_loss, val_iou = Unet3D.train.train(self.unet_model, loaders, self.logdir, self.writer, self.device)
        self.logger.record('nn_training/train_loss', train_loss)
        self.logger.record('nn_training/val_loss', val_loss)
        self.logger.record('nn_training/val_iou', val_iou.item())

        print(f"Update Unet cost time: {time.time() - t}")

    def update_env(self):
        env = self.training_env
        state_dict = self.unet_model.state_dict()
        nn_mask_state_dict = {key: value.to(torch.device("cuda:0")) for key, value in state_dict.items()}
        env.env_method('update_nn_mask', nn_mask_state_dict)
        # env.env_method('update_mean_std', self.dataset.mean, self.dataset.std)