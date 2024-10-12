import os, argparse, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env

register(
    id='BoxPlanning',
    entry_point='env:BoxPlanningEnvWrapper',
    max_episode_steps=75,
    nondeterministic=False,
)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--exp_id', type=str, required=True, help='Experiment ID')
    parser.add_argument('--check_point_name', type=str, required=True, help='Checkpoint name')
    args = parser.parse_args()

    device = torch.device(args.device)
    exp_id = args.exp_id
    check_point_name = args.check_point_name

    logdir = os.path.join("logs", exp_id)
    save_video_path = f"video/test_policy_{exp_id}.mp4"
    model_path = os.path.join(logdir, check_point_name)
    stats_path = os.path.join(logdir, "vec_normalize.pkl")
    nn_mask_path = os.path.join(logdir, "nn_training", 'checkpoint.pytorch')

    env_kwargs = {"save_video_path": save_video_path, "mask_type": 'nn', 'nn_mask_path': nn_mask_path, 'device':device}
    env = make_vec_env("BoxPlanning", n_envs=1, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)
    env = VecNormalize.load(stats_path, env)
    model = PPO.load(model_path, device='cpu', env=env)

    obs = env.reset()
    for i in range(5):
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)