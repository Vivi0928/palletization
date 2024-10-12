import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from gymnasium.envs.registration import register
from task_config import TaskConfig

from helpers.utils import LogRewardCallback, CustomExtractor, save_config, set_seed
from helpers.train_mask import TrainFeasibleMaskCallback

register(
    id='BoxPlanning',
    entry_point='env:BoxPlanningEnvWrapper',
    max_episode_steps=75,
)

if __name__ == "__main__":
    # Experiment setup
    set_seed(TaskConfig.train.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = TaskConfig.train.CUDA_VISIBLE_DEVICES
    exp_save_dir = TaskConfig.train.mask_type + TaskConfig.train.exp_id
    log_dir = os.path.join("logs", exp_save_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Make environment
    env_kwargs = {"save_video_path": None, "mask_type":TaskConfig.train.mask_type, 'nn_mask_path':None, 'device':torch.device("cuda:0")}
    env = make_vec_env("BoxPlanning", n_envs=TaskConfig.train.n_envs, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv, seed=TaskConfig.train.seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=TaskConfig.train.gamma)
    pallet_size_discrete = (np.array(TaskConfig.pallet.size)[:2] / TaskConfig.bin_size).astype(int)
    print(f"Total envs: {TaskConfig.train.n_envs}")

    # Save stats
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)

    # Save config
    config_path = "./task_config.py"
    save_config(config_path, log_dir)

    # Define model and train
    policy_kwargs = dict(
        features_extractor_class=CustomExtractor,
        features_extractor_kwargs=dict(
            n_properties=TaskConfig.box.n_properties,
            buffer_size=TaskConfig.buffer_size,
        ),
    )
    model = PPO("MultiInputPolicy", env=env, 
                learning_rate=TaskConfig.train.learning_rate, 
                n_steps=TaskConfig.train.n_steps, 
                batch_size=TaskConfig.train.batch_size, 
                gamma=TaskConfig.train.gamma, 
                use_sde=TaskConfig.train.use_sde,
                policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir, seed=TaskConfig.train.seed)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=12500, save_path=log_dir, name_prefix="model")
    callback = CallbackList([LogRewardCallback(), checkpoint_callback])
    if TaskConfig.train.mask_type == 'nn':
        train_mask_callback = TrainFeasibleMaskCallback(
            pallet_size_discrete=pallet_size_discrete, 
            max_pallet_height=TaskConfig.pallet.max_pallet_height ,
            logdir=log_dir)
        callback.callbacks.append(train_mask_callback)

    model.learn(total_timesteps=TaskConfig.train.total_timesteps, log_interval=1, callback=callback)

    print("######  Train finished, save the last model  ######")
    model.save(os.path.join(log_dir, "model"))
