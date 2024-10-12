import numpy as np
import random
import os
import sys
import shutil
import pickle
import time
import torch.optim as optim
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from collections import deque
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from gymnasium import spaces
import torch


class CustomExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Dict, n_properties, buffer_size):
        super().__init__(observation_space, features_dim=1)
        # expect shape:[n_envs, channels, pallet_size_discrete[0], pallet_size_discrete[1], max_height]
        extractors = {}
        cnn = nn.Sequential(
            nn.Unflatten(1,(1,25)), # Channel=1

            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),

            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),

            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),

            nn.Flatten(),
        )
        mlp = nn.Sequential(
            nn.Linear(n_properties * buffer_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )
        extractors["pallet_obs_density"] = cnn
        extractors["buffer"] = mlp

        self.extractors = nn.ModuleDict(extractors)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = cnn(
                torch.as_tensor(observation_space["pallet_obs_density"].sample()[None]).float()
            ).shape[1]
        self._features_dim = 64 + n_flatten
        print(f"Feature dim: {self._features_dim}")


    def forward(self, observations) -> torch.Tensor:
        encoded = []
        for key, extractor in self.extractors.items():
            encoded.append(extractor(observations[key]))
        return torch.cat(encoded, dim=1)
    

# Callback for logging
class LogRewardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(LogRewardCallback, self).__init__(verbose)
        self.window_length = 100
        self.episode_reward_box_size_buffer = deque(maxlen=self.window_length)
        self.episode_reward_box_size = 0
        self.episode_done_reason_buffer = deque(maxlen=self.window_length)

    def _on_step(self) -> bool:
        # 更新当前episode的reward和长度
        self.episode_reward_box_size += self.locals['infos'][0].get('reward_box_size', 0)

        # 如果episode结束，记录平均reward
        if self.locals['dones'][0]:
            self.episode_reward_box_size_buffer.append(self.episode_reward_box_size)
            self.episode_reward_box_size= 0
            self.episode_done_reason_buffer.append(self.locals['infos'][0].get('termination_reason', -1))

        return True
    
    def _on_rollout_end(self):
        mean_reward_box_size = np.mean(self.episode_reward_box_size_buffer)

        self.logger.record('rollout/mean_rew_box_size', mean_reward_box_size)
        window_length = len(self.episode_done_reason_buffer)
        self.logger.record('rollout/infeasible_rate', self.episode_done_reason_buffer.count(1)/window_length)
        self.logger.record('rollout/unstable_rate', self.episode_done_reason_buffer.count(2)/window_length)
        self.logger.record('rollout/success_rate', self.episode_done_reason_buffer.count(3)/window_length)
        

class Heuri:
    def __init__(self, pallet_size_discrete, max_pallet_height) -> None:
        self.pallet_size = pallet_size_discrete
        self.max_pallet_height = max_pallet_height
        

    def is_place_stable(self, height_map, size_after_rotate, x, y, density):
        def is_qualified(support_ratio, num_support_corners, height_to):
            return (support_ratio>0.6 and num_support_corners==4 and height_to<=self.max_pallet_height) or \
                (support_ratio>0.8 and num_support_corners>=3 and height_to<=self.max_pallet_height) or \
                (support_ratio>0.95 and height_to<=self.max_pallet_height)
    
        size_after_rotate = size_after_rotate.astype(int)
        length, width, height = size_after_rotate[0], size_after_rotate[1], size_after_rotate[2]
        
        area_height = np.max(height_map[x:x+length, y:y+width])
        area = height_map[x:x+length, y:y+width]
        vals, counts = np.unique(area, return_counts=True)
        count = np.sum(area == area_height)
        support_area_ratio = count / (area.shape[0] * area.shape[1])
        num_support_corners = (area_height == height_map[x, y]) \
                            + (area_height == height_map[x+length-1, y]) \
                            + (area_height == height_map[x, y+width-1]) \
                            + (area_height == height_map[x+length-1, y+width-1])
        height_to = area_height + height

        if is_qualified(support_area_ratio, num_support_corners, height_to):
            return 1
        else:
            return 0

    def heuri_annotation(self, pallet_obs_density, size_after_rotate, box_density):
        feasible_map = np.zeros(self.pallet_size)
        height_map = self.get_map(pallet_obs_density)
        
        for x in range(int(feasible_map.shape[0]-size_after_rotate[0]+1)):
            for y in range(int(feasible_map.shape[1]-size_after_rotate[1]+1)):
                feasible_map[x, y] = self.is_place_stable(height_map, size_after_rotate, x, y, box_density)
        
        return feasible_map
    
    def get_map(self, pallet_obs_density):
        height_map = np.max((pallet_obs_density > 0) * np.arange(1, pallet_obs_density.shape[2] + 1), axis=2)

        return height_map


    def __call__(self, x):
        anno = []
        for i in range(x.shape[0]):
            image = x[i]
            obs_density = image[0].numpy()
            box_size = np.array([image[1][0][0][0], image[2][0][0][0], image[3][0][0][0]])
            anno.append(self.heuri_annotation(obs_density ,box_size))
        return torch.from_numpy(np.stack(anno, axis=0))
    
    def eval(self):
        pass

    def train(self):
        raise NotImplementedError("The heuristic model can't be trained!")

def save_config(config_path, log_dir):
    txt_path = os.path.join(log_dir, "task_config.txt")
    shutil.copyfile(config_path, txt_path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def read_data(datapath):
    with open(datapath, "rb") as f:
        return pickle.load(f)