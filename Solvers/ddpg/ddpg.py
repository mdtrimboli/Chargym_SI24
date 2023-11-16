import copy
from datetime import datetime
from functional import seq
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam

from Solvers.ddpg.core.config import Config
from Solvers.ddpg.core.replay_buffer import ReplayBuffer
from Solvers.ddpg.utils.list import for_each, select_with_predicate

class DDPG:
    def __init__(self,
                 env,
                 actor,
                 critic):
        self._env = env
        self._actor = actor
        self._critic = critic

        self._config = Config.get().ddpg.trainer
        self.episodic_reward_buffer = []
        self.episodic_length_buffer = []

        self._initialize_target_networks()
        self._initialize_optimizers()

        self._models = {
            'actor': self._actor,
            'critic': self._critic,
            'target_actor': self._target_actor,
            'target_critic': self._target_critic
        }

        self._replay_buffer = ReplayBuffer(self._config.replay_buffer_size)

        self._train_global_step = 0
        self._eval_global_step = 0

        if self._config.use_gpu:
            print("CUDA Activated")
            self._cuda()

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad

        if self._config.use_gpu:
            self._cuda()
            tensor = tensor.cuda()

        return tensor

    def _initialize_target_networks(self):
        self._target_actor = copy.deepcopy(self._actor)
        self._target_critic = copy.deepcopy(self._critic)
    
    def _initialize_optimizers(self):
        self._actor_optimizer = Adam(self._actor.parameters(), lr=self._config.actor_lr)
        self._critic_optimizer = Adam(self._critic.parameters(), lr=self._config.critic_lr)
    
    def _eval_mode(self):
        for_each(lambda x: x.eval(), self._models.values())

    def _train_mode(self):
        for_each(lambda x: x.train(), self._models.values())

    def _cuda(self):
        for_each(lambda x: x.cuda(), self._models.values())

    def _get_action(self, observation, is_training=True):
        # Action + random gaussian noise (as recommended in spining up)
        action = self._actor(self._as_tensor(self._flatten_dict(observation)))
        if is_training:
            action += self._config.action_noise_range * torch.randn(self._env.action_space.shape).cuda()

        action = action.cpu().data.numpy()

        return action

    def _get_q(self, batch):
        return self._critic(self._as_tensor(batch["observation"]))

    def _get_target(self, batch):
        # For each observation in batch:
        # target = r + discount_factor * (1 - done) * max_a Q_tar(s, a)
        # a => actions of actor on current observations
        # max_a Q_tar(s, a) = output of critic
        observation_next = self._as_tensor(batch["observation_next"])
        reward = self._as_tensor(batch["reward"]).reshape(-1, 1)
        done = self._as_tensor(batch["done"]).reshape(-1, 1)

        action = self._target_actor(observation_next).reshape(-1, *self._env.action_space.shape)

        q = self._target_critic(observation_next, action)

        return reward + self._config.discount_factor * (1 - done) * q

    def _flatten_dict(self, inp):
        if type(inp) == dict:
            inp = np.concatenate(list(inp.values()))
        return inp

    def _update_targets(self, target, main):
        for target_param, main_param in zip(target.parameters(), main.parameters()):
            target_param.data.copy_(self._config.polyak * target_param.data + \
                                    (1 - self._config.polyak) * main_param.data)

    def _update_batch(self):
        batch = self._replay_buffer.sample(self._config.batch_size)
        # Only pick steps in which action was non-zero
        # When a constraint is violated, the safety layer makes action 0 in
        # direction of violating constraint
        # valid_action_mask = np.sum(batch["action"], axis=1) > 0
        # batch = {k: v[valid_action_mask] for k, v in batch.items()}

        # Update critic
        self._critic_optimizer.zero_grad()
        q_target = self._get_target(batch)
        q_predicted = self._critic(self._as_tensor(batch["observation"]),
                                   self._as_tensor(batch["action"]))
        # critic_loss = torch.mean((q_predicted.detach() - q_target) ** 2)
        # Seems to work better
        critic_loss = F.smooth_l1_loss(q_predicted, q_target)

        critic_loss.backward()
        self._critic_optimizer.step()

        # Update actor
        self._actor_optimizer.zero_grad()
        # Find loss with updated critic
        new_action = self._actor(self._as_tensor(batch["observation"])).reshape(-1, *self._env.action_space.shape)
        actor_loss = -torch.mean(self._critic(self._as_tensor(batch["observation"]), new_action))
        actor_loss.backward()
        self._actor_optimizer.step()

        # Update targets networks
        self._update_targets(self._target_actor, self._actor)
        self._update_targets(self._target_critic, self._critic)

        self._train_global_step +=1

    def _update(self, episode_length):
        # Update model #episode_length times
        for_each(lambda x: self._update_batch(),
                 range(min(episode_length, self._config.max_updates_per_episode)))

    def evaluate(self):
        episode_rewards = []
        episode_lengths = []
        episode_actions = []

        observation = self._env.reset()
        episode_reward = 0
        episode_length = 0
        episode_action = 0

        self._eval_mode()

        for step in range(self._config.evaluation_steps):
            action = self._get_action(observation, is_training=False)
            episode_action += np.absolute(action)
            observation, reward, done, _ = self._env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done or (episode_length == self._config.max_episode_length):

                self.episodic_reward_buffer.append(episode_reward)
                self.episodic_length_buffer.append(episode_length)

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_actions.append(episode_action / episode_length)

                observation = self._env.reset()
                episode_reward = 0
                episode_length = 0
                episode_action = 0

        mean_episode_reward = np.mean(episode_rewards)
        mean_episode_length = np.mean(episode_lengths)

        self._eval_global_step += 1

        self._train_mode()

        print("Validation completed:\n"
              f"Number of episodes: {len(episode_actions)}\n"
              f"Average episode length: {mean_episode_length}\n"
              f"Average reward: {mean_episode_reward}\n"
              f"Average action magnitude: {np.mean(episode_actions)}")

    def train(self):
        
        start_time = time.time()

        print("==========================================================")
        print("Initializing DDPG training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        observation = self._env.reset()
        episode_reward = 0
        episode_length = 0

        number_of_steps = self._config.steps_per_epoch * self._config.epochs

        for step in range(number_of_steps):
            # Randomly sample episode_ for some initial steps
            action = self._env.action_space.sample() if step < self._config.start_steps \
                     else self._get_action(observation)
            
            observation_next, reward, done, _ = self._env.step(action)
            episode_reward += reward
            episode_length += 1

            self._replay_buffer.add({
                "observation": self._flatten_dict(observation),
                "action": action,
                "reward": np.asarray(reward) * self._config.reward_scale,
                "observation_next": self._flatten_dict(observation_next),
                "done": np.asarray(done),
            })

            observation = observation_next

            # Make all updates at the end of the episode
            if done or (episode_length == self._config.max_episode_length):
                if step >= self._config.min_buffer_fill:
                    self._update(episode_length)
                # Reset episode
                observation = self._env.reset()
                episode_reward = 0
                episode_length = 0

            # Check if the epoch is over
            if step != 0 and step % self._config.steps_per_epoch == 0: 
                print(f"Finished epoch {step / self._config.steps_per_epoch}. Running validation ...")
                self.evaluate()
                print("----------------------------------------------------------")

        print("==========================================================")
        print(f"Finished DDPG training. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")