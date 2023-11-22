import numpy as np
import gym
import argparse
import os
import torch
from Solvers import check_main
from stable_baselines3.common.env_checker import check_env
from Solvers.RL.ppo.normalization import Normalization, RewardScaling
from Solvers.RL.ppo.replay_buffer import ReplayBuffer
from Solvers.RL.ppo.ppo_continuous import PPO_continuous


def save_models(actor, critic, directory, actor_filename='ppo_actor.pth', critic_filename='ppo_critic.pth'):
    """
    Guarda los modelos del actor y del crítico en el directorio especificado.

    Args:
    - actor: Modelo del actor de PyTorch.
    - critic: Modelo del crítico de PyTorch.
    - directory: Directorio donde se guardarán los modelos.
    - actor_filename: Nombre del archivo para el modelo del actor.
    - critic_filename: Nombre del archivo para el modelo del crítico.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    actor_path = os.path.join(directory, actor_filename)
    critic_path = os.path.join(directory, critic_filename)

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)


def load_models(actor, critic, directory, actor_filename='actor.pth', critic_filename='critic.pth'):
    """
    Carga los modelos del actor y del crítico desde el directorio especificado.

    Args:
    - actor: Modelo del actor de PyTorch.
    - critic: Modelo del crítico de PyTorch.
    - directory: Directorio desde donde se cargarán los modelos.
    - actor_filename: Nombre del archivo para el modelo del actor.
    - critic_filename: Nombre del archivo para el modelo del crítico.
    """
    actor_path = os.path.join(directory, actor_filename)
    critic_path = os.path.join(directory, critic_filename)

    actor.load_state_dict(torch.load(actor_path))
    critic.load_state_dict(torch.load(critic_path))


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, number, seed):
    env_name = 'ChargingEnv-v0'
    env = gym.make('ChargingEnv-v0')
    env_evaluate = gym.make('ChargingEnv-v0')  # When evaluating the policy, we need to rebuild an environment

    # It will check your custom environment and output additional warnings if needed
    #check_main(env)

    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format('ChargingEnv-v0'))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                #writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                directory = 'data'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                if evaluate_num % args.save_freq == 0:
                    np.save('data/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))
    directory_2 = 'curves'
    if not os.path.exists(directory_2):
        os.makedirs(directory_2)
    np.savetxt("curves/Rew_PPO.csv", evaluate_rewards, delimiter=", ", fmt='% s')
    SAVE = True


    if SAVE:
        # Guardar modelos
        save_models(agent.actor, agent.critic, 'model')
    else:
        pass
        # TODO: Armar una clase que cargue los modelos de torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for ppo-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(480000), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=24, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=1200, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="ppo clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="ppo parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    main(args, number=1, seed=10)
