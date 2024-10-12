import os
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np
import torch

class FeasibleDataset(Dataset):
    def __init__(self, data_dir_path):
        all_filenames = [os.path.join(data_dir_path, fname) for fname in os.listdir(data_dir_path) if fname.endswith("npz")]
        random.shuffle(all_filenames)
        self.data = []
        self.label = []
        for fname in all_filenames:
            try:
                data, label = self.load_data(fname)
                self.data.append(data)
                self.label.append(label)
            except (AssertionError, ValueError) as e:
                print(f"跳过文件 {fname}，原因: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label[idx]
        return sample, label

def get_dataloader(dataset: Dataset, split:list, seed, num_workers=2):
    generator = torch.Generator().manual_seed(seed)
    trainset, testset, valset = random_split(dataset, split, generator=generator)
    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=1024, shuffle=False, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=1024, shuffle=False, num_workers=num_workers)
    return {"Train":trainloader, "Test":testloader, "Val":valloader}

if __name__ == "__main__":
    dataset = FeasibleDataset("../feasible_anno_2")
    print(dataset.__len__())