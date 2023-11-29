import numpy as np
import gym
import argparse
import os
import torch
from datetime import datetime

from Solvers import check_main
from stable_baselines3.common.env_checker import check_env
from Solvers.RL.ppo.normalization import Normalization, RewardScaling
from Solvers.RL.ppo.replay_buffer import ReplayBuffer
from Solvers.RL.ppo.ppo_continuous import PPO_continuous
from Solvers.RL.ppo.ppo import PPO


def save_models(actor, critic, directory):
    """
    Guarda los modelos del actor y del crítico en el directorio especificado.

    Args:
    - actor: Modelo del actor de PyTorch.
    - critic: Modelo del crítico de PyTorch.
    - directory: Directorio donde se guardarán los modelos.
    - actor_filename: Nombre del archivo para el modelo del actor.
    - critic_filename: Nombre del archivo para el modelo del crítico.
    """
    fecha_actual = datetime.now().date()
    actor_filename = f'ppo_actor_{fecha_actual}.pth'
    critic_filename = f'ppo_critic_{fecha_actual}.pth'

    if not os.path.exists(directory):
        os.makedirs(directory)

    actor_path = os.path.join(directory, actor_filename)
    critic_path = os.path.join(directory, critic_filename)

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)


def load_models(actor, critic, fecha_carga, directory='model/'):
    """
    Carga los modelos del actor y del crítico desde el directorio especificado.

    Args:
    - actor: Modelo del actor de PyTorch.
    - critic: Modelo del crítico de PyTorch.
    - directory: Directorio desde donde se cargarán los modelos.
    - actor_filename: Nombre del archivo para el modelo del actor.
    - critic_filename: Nombre del archivo para el modelo del crítico.
    """

    fecha_carga = fecha_carga
    actor_filename = f'ppo_actor_{fecha_carga}.pth'
    critic_filename = f'ppo_critic_{fecha_carga}.pth'

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

    SAVE = False
    fecha_carga = '2023-11-29'
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

    agent = PPO_continuous(args)
    agent_ppo = PPO(env, agent, args)

    directory_2 = 'curves'

    if SAVE:
        agent_ppo.train()
        save_models(agent.actor, agent.critic, 'model')     # Guardar modelos
        if not os.path.exists(directory_2):
            os.makedirs(directory_2)
        np.savetxt("curves/Rew_PPO.csv", agent_ppo._evaluate_rewards, delimiter=", ", fmt='% s')

    else:
        cwb_actor = agent.actor.state_dict()
        cwb_critic = agent.critic.state_dict()
        load_models(agent.actor, agent.critic, fecha_carga, 'model')
        agent_ppo.evaluate(agent)

    if not SAVE:
        np.savetxt("curves/Precio.csv", np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1,
                                                  0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.06, 0.06, 0.06, 0.1, 0.1, 0.1, 0.1]),
                   delimiter=", ", fmt='% s')
        np.savetxt("curves/E_almacenada_red_ppo.csv", env.Grid_Evol_mem, delimiter=", ", fmt='% s')
        np.savetxt("curves/E_almacenada_PV_ppo.csv", env.E_almac_pv, delimiter=", ", fmt='% s')
        # gráfico c) Perfil de carga
        # np.savetxt("curves/Presencia_autos.csv", env.Invalues['present_cars'], delimiter=", ", fmt='% s')
        np.savetxt("curves/Presencia_autos_ppo.csv", agent_ppo.presence, delimiter=", ", fmt='% s')
        # np.savetxt("curves/SOC.csv", env.SOC, delimiter=", ", fmt='% s')
        np.savetxt("curves/SOC_ppo.csv", agent_ppo.soc, delimiter=", ", fmt='% s')
        np.savetxt("curves/E_almacenada_total_ppo.csv", env.Lista_E_Almac_Total, delimiter=", ", fmt='% s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for ppo-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(480000), help=" Maximum number of training steps")   #48k para entrenar
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
