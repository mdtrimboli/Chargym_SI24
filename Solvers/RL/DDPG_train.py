import gym
import Chargym_Charging_Station
import argparse
import gym
import numpy as np
import os
import time
import torch

from functional import seq
from datetime import datetime

from Solvers.ddpg.core.config import Config
from Solvers.ddpg.actor import Actor
from Solvers.ddpg.critic import Critic
from Solvers.ddpg.ddpg import DDPG

#from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy


SAVE = True     # True para Guardar - False para Cargar modelo


fecha_actual = datetime.now().date()
fecha_carga = fecha_actual

config = Config.get().main.trainer
models_dir = f"models/DDPG-{int(time.time())}"
logdir = f"logs/DDPG-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = gym.make('ChargingEnv-v0')            # Creación del entorno gym

# It will check your custom environment and output additional warnings if needed
check_env(env)
# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
################################################################
# PARTE DE DDPG NUEVO

#observation_dim = (seq(env.observation_space.spaces.values()).map(lambda x: x.shape[0]).sum())
observation_dim = env.observation_space.shape[0]

actor = Actor(observation_dim, env.action_space.shape[0])
critic = Critic(observation_dim, env.action_space.shape[0])
ddpg = DDPG(env, actor, critic)

if SAVE:
    ddpg.train()

    directory = 'model'
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(ddpg._actor.state_dict(), f'model/actor_weights_{fecha_actual}.pth')
    torch.save(ddpg._critic.state_dict(), f'model/critic_weights_{fecha_actual}.pth')
else:
    ddpg._actor.load_state_dict(torch.load(f'model/actor_weights_{fecha_carga}.pth'))
    ddpg._critic.load_state_dict(torch.load(f'model/critic_weights_{fecha_carga}.pth'))
    ddpg.evaluate()

if SAVE:
    directory_2 = 'curves'
    if not os.path.exists(directory_2):
        os.makedirs(directory_2)
    np.savetxt("curves/Rew_DDPG.csv", ddpg.episodic_reward_buffer, delimiter=", ", fmt='% s')
    #np.savetxt("curves/ALVConst_Eval_DDPG_SL.csv", ddpg.accum_lv_eval, delimiter=", ", fmt='% s')
else:
    #np.savetxt("curves/Price.csv", ddpg.temp, delimiter=", ", fmt='% s')
    #Gráfico b) Evolución Almacenamiento Energía
    np.savetxt("curves/Precio.csv", np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                              0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1]), delimiter=", ", fmt='% s')
    np.savetxt("curves/E_almacenada_red.csv", env.Grid_Evol, delimiter=", ", fmt='% s')
    np.savetxt("curves/E_almacenada_PV.csv", env.E_almac_pv, delimiter=", ", fmt='% s')
    #gráfico c) Perfil de carga
    np.savetxt("curves/Presencia_autos.csv", env.Invalues['present_cars'], delimiter=", ", fmt='% s')
    np.savetxt("curves/Presencia_autos.csv", env.BOC, delimiter=", ", fmt='% s')





#####################################################################

# PARTE DE SB3
"""
model = DDPG(MlpPolicy, env, verbose=1, learning_rate=1e-6,
             batch_size=64, action_noise=action_noise, tensorboard_log=logdir)


TIMESTEPS = 150000
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG", log_interval=10)
model.save(f"{models_dir}/{TIMESTEPS}")

#This will save every 20000 steps the models
#TIMESTEPS = 20000

"""
"""
for i in range(1, 50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG")
    model.save(f"{models_dir}/{TIMESTEPS * i}")
"""

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



