import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from numpy import loadtxt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d

sns.set_theme()

actual_date = datetime.now().date()


### COMPARACION DE REWARD

rew_curves_DDPG = open('Solvers/RL/curves/Rew_DDPG.csv', 'rb')
rew_curves_PPO = open('Solvers/RL/curves/Rew_PPO.csv', 'rb')
data_DDPG = gaussian_filter1d(loadtxt(rew_curves_DDPG, delimiter=","), sigma=5)
data_PPO = gaussian_filter1d(loadtxt(rew_curves_PPO, delimiter=","), sigma=5)

plt.plot(data_DDPG, label='DDPG', color='tab:red')
plt.plot(data_PPO, label='PPO', color='tab:blue')

plt.legend(loc="lower right")
plt.xlabel("Training Episodes")
plt.ylabel("Episodic reward")
plt.savefig(f'Solvers/RL/curves/Reward_comp_{actual_date}.png', dpi=600)
plt.show()
