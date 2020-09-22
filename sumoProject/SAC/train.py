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

from sumoProject.SAC.replay_memory import ReplayMemory
from sumoProject.SAC.sac import SAC

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
parser.add_argument('--lr', type=float, default=0.1, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=10000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_episode', type=int, default=50, metavar='N',
                    help='model updates per episode (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=100, metavar='N',
                    help='Value target update per no. of updates  (default: 1)')
parser.add_argument('--replay_size', type=int, default=50000, metavar='N',
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
continue_training = False
# Agent
observation_space = env.observation_space.flatten().shape[0]
agent = SAC(num_inputs=50, action_space=env.action_space, args=args)
if continue_training:
    agent.load_model(
        conv_path="/home/st106/workspace/RLHighWay/sumoProject/SAC/conv_train/2020-05-15_18-13-30_SAC_EPHighWay-v1_Gaussian_/models/conv_e41500.pkl",
        actor_path="/home/st106/workspace/RLHighWay/sumoProject/SAC/conv_train/2020-05-15_18-13-30_SAC_EPHighWay-v1_Gaussian_/models/actor_e41500.pkl",
        critic_path="/home/st106/workspace/RLHighWay/sumoProject/SAC/conv_train/2020-05-15_18-13-30_SAC_EPHighWay-v1_Gaussian_/models/critic_e41500.pkl")
# Training Loop
total_numsteps = 0
updates = 0
top_reward = 0
avg_reward = 0.
done_episodes = 0
for i_episode in itertools.count(1):

    episode_reward = 0
    episode_steps = 0
    done = False

    try:
        state = env.reset()
        while not done:
            if args.start_steps > total_numsteps:
                if continue_training:
                    action = agent.select_action(state, evaluate=True)
                else:
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
    except RuntimeError:
        env.stop()
        env = gym.make(args.env_name)
        env.render(mode='none')

    if info["cause"] is None:
        done_episodes += 1
    avg_reward += episode_reward

    if i_episode % 500 == 0 and args.eval is True:
        done_episodes /= 500
        avg_reward /= 500
        if abs(avg_reward) >= top_reward:
            agent.save_model('sumo', f"e{i_episode}")
            top_reward = abs(avg_reward)
        writer.add_scalar('test/reward', avg_reward, i_episode)
        writer.add_scalar('test/success', done_episodes, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(500, round(avg_reward, 2)))
        print("----------------------------------------")
        avg_reward = 0.
        done_episodes = 0

    if i_episode % 50 == 0 and total_numsteps > args.start_steps:

        if len(memory) > args.batch_size * args.updates_per_episode / 10:
            # Number of updates per step in environment
            trange_ = tqdm.trange(args.updates_per_episode, desc="Updating network weights")
            for i in trange_:
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent(memory,
                                                                                   updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy', ent_loss, updates)
                writer.add_scalar('loss/alpha', alpha, updates)
                updates += 1

        if updates > args.num_steps:
            agent.save_model('sumo', f"e{i_episode}")
            break

env.stop()
