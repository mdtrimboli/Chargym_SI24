import gym
import numpy as np
import Chargym_Charging_Station

import os
from RBC import RBC
import argparse

from datetime import datetime

fecha_actual = datetime.now().date()
#models_dir = f"models/DDPG-{int(time.time())}"
#logdir = f"logs/DDPG-{int(time.time())}"

#if not os.path.exists(models_dir):
    #os.makedirs(models_dir)

#if not os.path.exists(logdir):
    #os.makedirs(logdir)

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="ChargingEnv-v0")
parser.add_argument("--reset_flag", default=1, type=int)
args = parser.parse_args()
env = gym.make(args.env)

i = 0
len_test = 50


rewards_list = []
for j in range(len_test):
    state = env.reset(reset_flag=0)
    done = False
    while not done:
        i += 1
        action = RBC.select_action(env.env, state)
        next_state, rewards, done, info = env.step(action)
        #print(rewards)
        state = next_state
        rewards_list.append(rewards)

SOC = info['SOC']
Presence = info['Presence']
np.savetxt("curves/E_almacenada_red_rbc.csv", env.Grid_Evol_mem, delimiter=", ", fmt='% s')
np.savetxt("curves/E_almacenada_PV_rbc.csv", env.Energy['Renewable'][0][:24], delimiter=", ", fmt='% s')
np.savetxt("curves/Presencia_autos_rbc.csv", Presence, delimiter=", ", fmt='% s')
np.savetxt("curves/SOC_rbc.csv", SOC, delimiter=", ", fmt='% s')
np.savetxt("curves/E_almacenada_total_rbc.csv", env.Lista_E_Almac_Total, delimiter=", ", fmt='% s')


final_reward = sum(rewards_list)
avg_reward = final_reward / len_test
print(avg_reward)
