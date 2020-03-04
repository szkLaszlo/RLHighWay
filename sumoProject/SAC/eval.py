import argparse
import os
import pickle

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sumoProject.SAC.model import ImageProcessor
from sumoProject.SAC.sac import SAC
from sumoProject.agents.eval_policyGrad import find_latest_weight


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

parser.add_argument('--episode_num', type=int, default=100, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--model_path',
                    default="/home/st106/workspace/RLHighWay/sumoProject/SAC/runs/2020-03-02_19-38-41_SAC_EPHighWay-v1_Gaussian_autotune",
                    help='path to actor and critic weights')
parser.add_argument('--conv_path',
                    default="/home/st106/workspace/RLHighWay/sumoProject/agents/torchSummary/20200212_191313/model_final.weight",
                    help='path to convolutional weights')

args = parser.parse_args()
with open(os.path.join(args.model_path, 'params.pkl'), 'br')as file:
    args_ = pickle.load(file)
    file.close()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args_.env_name)
env.render('none')

# Tensorboard
logdir = f'evals/{args.model_path.split("runs/")[1]}'
if not os.path.exists(logdir):
    os.mkdir(logdir)
writer = SummaryWriter(log_dir=logdir)

# Agent
observation_space = env.observation_space.flatten().shape[0]
agent = SAC(20, env.action_space, args_)
path = find_latest_weight(path=args.model_path, file_end='.pkl')
agent.load_model(actor_path=path if 'actor' in path else path.replace('critic', 'actor'),
                 critic_path=path if 'critic' in path else path.replace('actor', 'critic'))
convolutional_prepare = ImageProcessor(model_path=args_.conv_path)

avg_reward = 0
success = 0

# Eval Loop
for episode in range(args.episode_num):
    try:
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        state = convolutional_prepare(state)
        with torch.no_grad():
            while not done:
                action = agent.select_action(state[np.newaxis, np.newaxis, :])  # Sample action from policy

                next_state, reward, done, info = env.step(action)  # Step
                next_state = convolutional_prepare(next_state)
                episode_steps += 1
                episode_reward += reward
                state = next_state

        if info['cause'] is None:
            success += 1
        writer.add_scalar('eval/reward', episode_reward, episode)
        writer.add_scalar('eval/lane_change', info['lane_change'], episode)
        writer.add_scalar('eval/distance', info['distance'], episode)

        print(f"Episode: {episode}, episode steps: {episode_steps}, "
              f"reward: {round(episode_reward, 2)}, cause: {info['cause']}, lane_change: {info['lane_change']}")
    except RuntimeError:
        env.stop()
        env = gym.make(args.env_name)
        env.render(mode='none')
        continue

    avg_reward += episode_reward
avg_reward /= args.episode_num
print(f'Average evaluation reward: {avg_reward} in {args.episode_num}. Successful episodes: {success}')

env.stop()
