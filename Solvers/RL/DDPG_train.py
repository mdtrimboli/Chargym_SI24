import gym
import Chargym_Charging_Station
import argparse

import numpy
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import gym
import numpy as np
import os

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import time
models_dir = f"../../models/DDPG-{int(time.time())}"
logdir = f"logs/DDPG-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = gym.make('ChargingEnv-v0')
from stable_baselines3.common.env_checker import check_env

# It will check your custom environment and output additional warnings if needed
check_env(env)
# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise, tensorboard_log=logdir)

# model.learn(total_timesteps=2000000, reset_num_timesteps=False, tb_log_name="DDPG")
# model.save(f"{models_dir}/{2000000}")

#This will save every 20000 steps the models
TIMESTEPS = 20000
for i in range(1, 50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG")
    model.save(f"{models_dir}/{TIMESTEPS * i}")



env.close
#del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_Chargym", env=env)
#
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#
# # Enjoy trained agent
# obs = env.reset()
# for i in range(24):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     # env.render(



#aaaaa=1