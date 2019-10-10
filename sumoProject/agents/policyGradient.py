import itertools
import os
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def network_plot(model, writer, epoch):
    for f in model.parameters():
        hist_name = 'gradients/' + str(list(f.grad.data.size()))
        writer.add_histogram(hist_name, f, epoch)


class Policy(nn.Module):

    def __init__(self, env, save_path=None, update_freq=10, gamma=0.99, learning_rate=0.001, use_gpu=True,
                 tb_summary=True):
        super(Policy, self).__init__()
        torch.manual_seed(int(time.time()))
        self.env = env
        self.save_path = save_path \
            if save_path is not None else 'torchSummary/{}'.format(time.strftime("%Y%m%d_%H%M%S", time.gmtime()))
        if not os.path.exists(self.save_path) and tb_summary:
            os.mkdir(self.save_path)
        self.writer = SummaryWriter(self.save_path) if tb_summary else None
        self.current_episode = 0
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.hidden_size = 128
        self.hidden_size2 = 64
        self.update_freq = update_freq
        self.l1 = nn.Linear(self.state_space, self.hidden_size, bias=True)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size2, bias=True)
        self.l3 = nn.Linear(self.hidden_size2, self.action_space, bias=True)

        self.gamma = gamma
        self.loss = 10
        self.use_gpu = use_gpu if torch.cuda.is_available() else False
        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.action_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.model = torch.nn.Sequential(
            self.l1,
            # nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            self.l2,
            # nn.BatchNorm1d(self.hidden_size2),
            nn.ReLU(),
            self.l3,
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if self.use_gpu:
            self.policy_history = self.policy_history.cuda()
            self.action_history = self.action_history.cuda()
            self.model = self.model.cuda()
            print("Using CUDA backend.")
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=200,
                                                              verbose=True, threshold=0.001, threshold_mode='rel',
                                                              cooldown=5, min_lr=0, eps=1e-08)

    def forward(self, x):
        return self.model(x)

    def update(self):
        self.current_episode += 1
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in self.reward_episode[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        if self.use_gpu:
            rewards = rewards.cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss
        self.loss = (torch.sum(torch.mul(self.policy_history, Variable(rewards)).mul(-1), -1))
        self.loss /= self.update_freq
        # Update network weights
        self.loss.backward()
        # plot_grad_flow(self.model.named_parameters())
        self.optimizer.step() if self.current_episode % self.update_freq == 0 else None
        self.optimizer.zero_grad() if self.current_episode % self.update_freq == 0 else None

        # network_plot(self.model, self.writer, self.current_episode)

        self.policy_history = Variable(torch.Tensor())
        self.action_history = Variable(torch.Tensor())
        if self.use_gpu:
            self.action_history = self.action_history.cuda()
            self.policy_history = self.policy_history.cuda()
        self.reward_episode = []

    def select_action_probabilities(self, state_):
        state_ = self.env.state_to_tuple(state_)
        state_ = torch.from_numpy(np.asarray(state_)).type(torch.FloatTensor)
        state_ = Variable(state_)
        if self.use_gpu:
            state_ = state_.cuda()
        action_probs = self.forward(state_.reshape(1, -1))
        c = Categorical(action_probs)  # np.random.choice(self.action_space, 1, p=action_probs.detach().numpy())
        action_ = c.sample()

        # Add probability of our chosen action to our history #not log because log1 = 0
        self.policy_history = torch.cat([self.policy_history, c.log_prob(action_).reshape(-1, )], 0)
        self.action_history = torch.cat([self.action_history, action_.float()], 0)
        return action_.detach().cpu() if action_.is_cuda else action_

    def train_network(self, episodes=100):
        running_reward = 0
        done_average = 0
        max_reward = -100000
        stopping_counter = 0

        for episode in range(episodes):
            state = self.env.reset()  # Reset environment and record the starting state
            done = False
            episode_reward = 0
            running_speed = []
            reward = 0
            for t in itertools.count():

                action = self.select_action_probabilities(state)

                # Step through environment using chosen action
                state, reward, done, info = self.env.step(action.item())
                running_speed.append(state['speed'])
                # Save reward
                self.reward_episode.append(reward)
                if done:
                    print(f"Steps:{t}, distance: {info['distance']:.3f}, "
                          f"average speed: {sum(running_speed) / len(running_speed):.2f} "
                          f"cause: {info['cause']}, reward: {info['rewards']:.3f} \n")
                    episode_reward = info['rewards']
                    print(f"Episode {episode + 1}:")
                    break

            running_reward += episode_reward
            if self.writer is not None:
                self.writer.add_scalar('episode/reward', episode_reward, episode)
                self.writer.add_scalar('episode/length', t, episode)
                self.writer.add_scalar('episode/speed', sum(running_speed) / len(running_speed), episode)
                self.writer.add_scalar('episode/finished', 1 if info['cause'] is None else 0, episode)
                done_average += 1 if info['cause'] is None else 0
                self.writer.add_scalar('episode/distance', info["distance"], episode)
                # writer.add_histogram('history/policy', pol.policy_history, episode)
                # writer.add_histogram('history/action', pol.action_history, episode)

            self.update()

            if episode % 50 == 0 and episode != 0:
                running_reward = running_reward / 50
                done_average = done_average / 50
                print('Episode {}\tLast length: {:5d}\tAverage reward: {:.2f}'.format(episode + 1, t, running_reward))
                self.writer.add_scalar('average/reward', running_reward,
                                       episode + 1) if self.writer is not None else None
                self.writer.add_scalar('average/done', done_average, episode + 1) if self.writer is not None else None
                self.scheduler.step(running_reward)
                if running_reward > max_reward:
                    max_reward = running_reward
                    stopping_counter = 0
                elif stopping_counter > 200:
                    print(f"The rewards did not improve since {50 * stopping_counter - 1} episodes")
                    self.env.stop()
                    break
                else:
                    stopping_counter += 1

                # pol.scheduler.step(running_reward)
                running_reward = 0
                done_average = 0

            if not episode % (episodes // 100):
                torch.save(self.model, os.path.join(self.save_path, 'model_{}.weight'.format(episode + 1)))
        torch.save(self.model, os.path.join(self.save_path, 'model_final.weight'))

    def eval_model(self, path, episode_nums):
        self.model = torch.load(path,
                                map_location=torch.device('cpu')
                                if "Windows" in platform.system() else torch.device('cuda'))
        self.model.train(False)
        with torch.no_grad():
            for _ in range(episode_nums):
                state = self.env.reset()  # Reset environment and record the starting state
                for t in itertools.count():
                    action = self.select_action_probabilities(state)
                    # Step through environment using chosen action
                    state, reward, done, info = self.env.step(action.item())
                    # Save reward
                    self.reward_episode.append(reward)
                    if done:
                        print(info)
                        print(t)
                        break
