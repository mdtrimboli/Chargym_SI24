import torch

from Solvers.RL.ddpg.core.config import Config
from Solvers.RL.ddpg.core.net import Net
from Solvers.RL.ddpg.utils.utils import init_fan_in_uniform


class Actor(Net):
    def __init__(self, observation_dim, action_dim):
        config = Config.get().ddpg.actor

        super(Actor, self).__init__(observation_dim,
                                    action_dim,
                                    config.layers,
                                    config.init_bound,
                                    init_fan_in_uniform,
                                    torch.tanh)
