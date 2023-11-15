import torch

from Solvers.ddpg.core.config import Config
from Solvers.ddpg.core.net import Net
from Solvers.ddpg.utils.utils import init_fan_in_uniform


class Actor(Net):
    def __init__(self, observation_dim, action_dim):
        config = Config.get().ddpg.actor

        super(Actor, self).__init__(observation_dim,
                                    action_dim,
                                    config.layers,
                                    config.init_bound,
                                    init_fan_in_uniform,
                                    torch.tanh)
