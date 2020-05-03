import argparse
import datetime
import itertools
import os
import pickle

import gym
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from traci import FatalTraCIError

from sumoProject.SAC.replay_memory import ReplayMemory
from sumoProject.SAC.sac import SAC
from sumoProject.agents.policyGradient import network_plot

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default='EPHighWay-v1',
                    help='SUMO Gym environment (default: EPHighWay-v1')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 100 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=100000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_episode', type=int, default=100, metavar='N',
                    help='model updates per episode (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=100, metavar='N',
                    help='Value target update per no. of updates  (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', default=True, action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--model_path',
                    default=None,
                    # "/home/st106/workspace/RLHighWay/sumoProject/agents/archive/20200212_191313/model_final.weight",
                    help='path to convolutional weights')
args = parser.parse_args()

# Tesnorboard
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = 'conv_train/{}_SAC_{}_{}_{}'.format(date, args.env_name, args.policy,
                                             "autotune" if args.automatic_entropy_tuning else "")
setattr(args, "log_dir", logdir)
writer = SummaryWriter(log_dir=logdir)
# write args file
with open(os.path.join(logdir, "params.pkl"), "bw") as file:
    pickle.dump(args, file)
    file.close()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)
env.render('none')

# Memory
memory = ReplayMemory(args.replay_size)

# Agent
observation_space = env.observation_space.flatten().shape[0]
agent = SAC(num_inputs=50, action_space=env.action_space, args=args)
# Training Loop
total_numsteps = 0
updates = 0
top_reward = -np.inf
for i_episode in itertools.count(1):

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    try:
        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                with torch.no_grad():
                    action = agent.select_action(state)  # Sample action from policy

            next_state, reward, done, info = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if info['cause'] is None and done else float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        writer.add_scalar('Episode/reward', episode_reward, i_episode)
        writer.add_scalar('Episode/lane_change', info['lane_change'], i_episode)
        writer.add_scalar('Episode/distance', info['distance'], i_episode)

        print(f"Episode: {i_episode}, total steps: {total_numsteps}, episode steps: {episode_steps}, "
              f"reward: {round(episode_reward, 3)}, {info['cause']}, lane_change: {info['lane_change']}")
    except FatalTraCIError:
        env.stop()
        env = gym.make(args.env_name)
        env.render(mode='none')
        continue
    if i_episode % 50 == 0:

        if len(memory) > args.batch_size * args.updates_per_episode / 10:
            # Number of updates per step in environment
            trange_ = tqdm.trange(args.updates_per_episode, desc="Updating network weights")
            for i in trange_:
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent(memory,
                                                                                   updates)
                network_plot(agent.convolution, writer, updates, name='conv')
                network_plot(agent.critic, writer, updates, "critic1")
                network_plot(agent.policy, writer, updates, 'policy')
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy', ent_loss, updates)
                writer.add_scalar('loss/alpha', alpha, updates)
                updates += 1

        if total_numsteps > args.num_steps:
            agent.save_model('sumo', f"e{i_episode}")
            break

        if i_episode % 500 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 100
            done = 0
            try:
                with torch.no_grad():
                    for _ in range(episodes):
                        state = env.reset()
                        episode_reward = 0
                        done = False
                        while not done:
                            action = agent.select_action(state, evaluate=True)

                            next_state, reward, done, info = env.step(action)
                            episode_reward += reward

                            state = next_state
                        if info["cause"] is None:
                            done += 1
                        avg_reward += episode_reward
            except FatalTraCIError:
                env.stop()
                env = gym.make(args.env_name)
                env.render(mode='none')
                continue
            done /= episodes
            avg_reward /= episodes
            if avg_reward >= top_reward:
                agent.save_model('sumo', f"e{i_episode}")
                top_reward = avg_reward
            writer.add_scalar('test/reward', avg_reward, i_episode)
            writer.add_scalar('test/success', done, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

env.stop()
