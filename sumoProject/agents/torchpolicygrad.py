import itertools
import os
import time

import easygui as easygui
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


class Policy(nn.Module):

    def __init__(self, env, episodes):
        super(Policy, self).__init__()
        self.env = env
        self.model = None
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = gamma
        self.loss = 10

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

    def forward(self, x):
        self.model = torch.nn.Sequential(
            self.l1,
            # nn.Dropout(p=0.4),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return self.model(x)

    def update(self):
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in self.reward_episode[::-1]:
            R = r + policy.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss
        self.loss = (torch.sum(torch.mul(self.policy_history, Variable(rewards)).mul(-1), -1))

        # Update network weights
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


        # Save and intialize episode history counters
        self.loss_history.append(self.loss.item())
        self.reward_history.append(np.sum(self.reward_episode))
        if len(self.loss_history) > 5000:
            self.loss_history.pop(0)
            self.reward_history.pop(0)
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []

    def select_action(self, state):
        # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
        state = self.env.state_to_tuple(state)
        state = torch.from_numpy(np.asarray(state)).type(torch.FloatTensor)
        state = Variable(state)
        state = self(state)
        c = Categorical(state)
        action = c.sample()

        # Add log probability of our chosen action to our history
        if self.policy_history.dim() != 0:
            self.policy_history = torch.cat([self.policy_history, c.log_prob(action).reshape(-1, )])
        else:
            self.policy_history = (c.log_prob(action))
        return action


def main(pol, save_path, episodes=100):
    running_reward = 0
    writer = SummaryWriter(save_path)
    episode_reward = 0

    for episode in range(episodes):
        state = env.reset()  # Reset environment and record the starting state
        done = False
        episode_reward = 0
        reward = 0
        for t in itertools.count():
            action = pol.select_action(state)
            # Step through environment using chosen action
            state, reward, done, info = pol.env.step(action.item())

            # Save reward

            pol.reward_episode.append(reward)
            if done:
                print(info)
                print(t)
                episode_reward = reward
                break

        # Used to determine when the environment is solved.
        running_reward += reward
        writer.add_scalar('episode_reward', episode_reward, episode)
        writer.add_scalar('episode_length', t, episode)
        writer.add_scalar('finished', 1 if info['cause'] is None else 0, episode)

        # writer.close()
        # writer.add_scalar('episode_length', t, 1)

        pol.update()
        pol.scheduler.step()

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage reward: {:.2f}'.format(episode, t, running_reward / 50))
            running_reward = 0
        if not episode % (episodes // 10):
            torch.save(policy, os.path.join(save_path, 'model_{}.weight'.format(episode)))


if __name__ == "__main__":
    train = True
    episode_nums = 10
    if train:
        env = gym.make('EPHighWay-v1')
        env.render(mode='none')

        torch.manual_seed(1)
        # Hyperparameters
        learning_rate = 0.0001
        gamma = 0.99
        episodes = 100000
        save_path = 'torchSummary/{}'.format(time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime()))
        policy = Policy(env=env,episodes=episodes)

        main(pol=policy,
             save_path=save_path,
             episodes=episodes,
             )
        torch.save(policy, os.path.join(save_path, 'model_final.weight'))
    else:
        path = easygui.fileopenbox()
        model = torch.load(path)
        env = gym.make('EPHighWay-v1')
        env.render()

        for _ in range(episode_nums):
            state = env.reset()  # Reset environment and record the starting state
            for t in itertools.count():
                action = model.select_action(state)
                # Step through environment using chosen action
                state, reward, done, info = env.step(action.item())

                # Save reward
                model.reward_episode.append(reward)
                if done:
                    print(info)
                    print(t)
                    episode_reward = reward
                    break
