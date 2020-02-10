import itertools
import os
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


def plot_grad_flow(named_parameters):
    """
    This function was used to plot gradient flow
    Parameters
    ----------
    named_parameters: Names of nn.Module parameters

    Returns
    -------

    """
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
    """
    Function to plot gradients of networks
    Parameters
    ----------
    model: nn.Module
    writer: SummaryWriter to write to
    epoch: defines the timestep for which this gradient is plotted

    Returns None
    -------

    """
    for f in model.parameters():
        hist_name = 'gradients/' + str(list(f.grad.data.size()))
        writer.add_histogram(hist_name, f, epoch)


class Policy(nn.Module):

    def __init__(self, env, save_path=None, update_freq=10, gamma=0.99, learning_rate=0.001, use_gpu=True,
                 tb_summary=True):
        """

        Parameters
        ----------
        env: Gym environment object
        save_path: Path to save the tensorboard events and weights of the training
        update_freq: defines after how many episodes is the network updated ("bacth")
        gamma: Defines how long a future reward has effect on the previous ones.
        learning_rate: Defines the initial learning rate.
        use_gpu: Defines whether to use GPU for training or not.
        tb_summary: If true, tensorboard event file will be created and saved.
        """
        super(Policy, self).__init__()
        torch.manual_seed(int(time.time()))
        self.env = env
        self.use_gpu = use_gpu if torch.cuda.is_available() else False
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")

        # Saving path creation if needed
        self.save_path = save_path \
            if save_path is not None else 'torchSummary/{}'.format(time.strftime("%Y%m%d_%H%M%S", time.gmtime()))
        if not os.path.exists(self.save_path) and tb_summary:
            os.mkdir(self.save_path)
        self.writer = SummaryWriter(self.save_path) if tb_summary else None

        self.current_episode = 0  # Initial episode counter
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.update_freq = update_freq
        self.timesteps_observed = 3  # Defines how many timesteps to feed for the network

        hidden_size_lstm = 64
        hidden_size_conv = 16

        # Building network modules
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size_conv * 2, kernel_size=11, padding=5,
                      stride=1),
            nn.BatchNorm2d(hidden_size_conv * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size_conv * 2, out_channels=hidden_size_conv * 2,
                      kernel_size=11, padding=5, stride=1),
            nn.BatchNorm2d(hidden_size_conv * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size_conv * 2, out_channels=hidden_size_conv,
                      kernel_size=11, padding=5, stride=1),
            nn.BatchNorm2d(hidden_size_conv),
            nn.Conv2d(in_channels=hidden_size_conv, out_channels=1,
                      kernel_size=11, padding=5, stride=1),
            nn.BatchNorm2d(1),
            nn.AdaptiveMaxPool2d(output_size=(10, 2))
        ).to(device=self.device)
        self.lstm = nn.LSTM(input_size=20, hidden_size=hidden_size_lstm, num_layers=1).to(device=self.device)

        self.linear = nn.Sequential(nn.Linear(in_features=hidden_size_lstm,
                                              out_features=hidden_size_lstm // 2),
                                    nn.ReLU(),
                                    nn.Linear(in_features=hidden_size_lstm // 2, out_features=self.action_space),
                                    nn.Softmax(dim=-1)
                                    ).to(device=self.device)
        self.gamma = gamma
        self.loss = 10

        # Episode policy and reward history
        self.policy_history = torch.tensor([], requires_grad=True, device=self.device)
        self.action_history = torch.tensor([])
        self.reward_episode = []

        self.model = list(self.convolution.parameters()) + list(self.linear.parameters()) \
                     + list(self.lstm.parameters())

        self.optimizer = optim.Adam(list(self.model), lr=learning_rate, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                              patience=int(0.01 / learning_rate), verbose=True,
                                                              threshold=0.01,
                                                              threshold_mode='rel',
                                                              cooldown=10, min_lr=0, eps=1e-08)
        # Putting parameters onto gpu if needed
        if self.use_gpu:
            print("Using CUDA backend.")

    def forward(self, x):
        """
        Function for forwarding the input of the network
        Parameters
        ----------
        x: has the dimensions as [time, x, y, channel]

        Returns
        -------
        the output of the network

        """
        # Changing dimension to be [time, channel, x, y]
        x = x.permute(0, 3, 1, 2)
        # Going through the timesteps and extracting the convolutional output
        lstm_input = self.convolution(x).flatten(-2, -1)  # using that batch size is one.
        # Preparing and propagating through lstm layer(s)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(lstm_input)
        # Removing unnecessary time steps (only using the last one)
        x = x[-1, :, :]
        # Getting the output of the MLP as the probability of actions
        linear_result = self.linear(x)
        return linear_result

    def update(self, episode):
        """
        Function used for updating the network
        Parameters
        ----------
        episode: episode number for the summary writer.

        Returns
        -------
        None
        """
        self.current_episode += 1  # Increasing episodes done

        # Calculating discounted rewards
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in self.reward_episode[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Scale rewards with the length of the path
        rewards = torch.tensor(rewards, dtype=torch.float, requires_grad=True, device=self.device) / len(rewards)

        # Calculating loss
        self.loss = (torch.sum(torch.mul(self.policy_history, rewards).mul(-1), -1))
        # Writing data to tensorboard
        self.writer.add_scalar('loss', self.loss.item(), episode)
        self.writer.add_scalar('episode/reward', sum(rewards), episode)
        self.loss /= self.update_freq

        # Update network weights
        self.loss.backward()
        # Step the network
        self.optimizer.step() if self.current_episode % self.update_freq == 0 else None
        self.optimizer.zero_grad() if self.current_episode % self.update_freq == 0 else None
        torch.cuda.empty_cache() if self.current_episode % self.update_freq == 0 else None

        # Delete histories
        self.policy_history = torch.tensor([], requires_grad=True, device=self.device)
        self.action_history = torch.tensor([])
        self.reward_episode = []

    def select_action_probabilities(self, state_):
        """

        Parameters
        ----------
        state_: State to predict from

        Returns
        -------
        Chosen action based on input
        """
        state_ = torch.tensor(state_, dtype=torch.float, requires_grad=True, device=self.device)
        action_probs = self.forward(state_)
        # Creating categorical distribution based on the output of the network
        c = Categorical(action_probs)
        # Selecting action
        action_ = c.sample()
        # Add probability of our chosen action to our history
        self.policy_history = torch.cat([self.policy_history, c.log_prob(action_).reshape(-1, )], 0)
        self.action_history = torch.cat([self.action_history, action_.float().detach().cpu()], 0)

        return action_.detach().cpu() if action_.is_cuda else action_.detach()

    def train_network(self, episodes=100):
        """
        Function to train the network
        Parameters
        ----------
        episodes: How many episodes to run

        Returns
        -------
        None
        """
        # Initiating global variables
        running_reward = 0
        done_average = 0
        max_reward = -100000
        stopping_counter = 0

        # Running the episodes
        for episode in range(episodes):
            state_list = [self.env.reset()]
            episode_reward = 0
            running_speed = []
            t = 0
            info = 0
            error_running_traci = False
            for t in itertools.count():
                # Selecting action based on current state
                action = self.select_action_probabilities(np.stack(state_list[-self.timesteps_observed:]))

                # Step through environment using chosen action
                try:
                    state, reward, done, info = self.env.step(action.item())
                except RuntimeError:
                    self.env.__init__()
                    self.env.render(mode='none')
                    # Delete histories
                    self.policy_history = torch.tensor([], requires_grad=True, device=self.device)
                    self.action_history = torch.tensor([])
                    self.reward_episode = []
                    error_running_traci = True
                    break
                state_list.append(state)
                running_speed.append(info['velocity'])
                # Save reward
                self.reward_episode.append(reward)

                # Printing to console if episode is terminated
                if done:
                    episode_reward = sum(self.reward_episode) / len(self.reward_episode)
                    print(f"Steps:{t}, distance: {info['distance']:.3f}, "
                          f"average speed: {sum(running_speed) / len(running_speed):.2f} "
                          f"cause: {info['cause']}, reward: {episode_reward:.3f} "
                          f"lane_changes: {info['lane_change']}\n")
                    print(f"Episode {episode + 1}:")
                    break

            if not error_running_traci:
                running_reward += episode_reward

                # Updating the network
                self.update(episode)

                # If needed writing episode details to tensorboard
                if self.writer is not None:
                    self.writer.add_scalar('episode/length', t, episode)
                    self.writer.add_scalar('episode/speed', sum(running_speed) / len(running_speed), episode)
                    self.writer.add_scalar('episode/lane_change', info['lane_change'], episode)
                    self.writer.add_scalar('episode/finished', 1 if info['cause'] is None else 0, episode)
                    done_average += 1 if info['cause'] is None else 0
                    self.writer.add_scalar('episode/distance', info["distance"], episode)

                # Calculating average based on 50 episodes
                if episode % 50 == 0 and episode != 0:
                    running_reward = running_reward / 50
                    done_average = done_average / 50
                    print(
                        'Episode {}\tLast length: {:5d}\tAverage reward: {:.2f}'.format(episode + 1, t, running_reward))
                    self.writer.add_scalar('average/reward', running_reward,
                                           episode + 1) if self.writer is not None else None
                    self.writer.add_scalar('average/done', done_average,
                                           episode + 1) if self.writer is not None else None

                    self.scheduler.step(running_reward)
                    # Checking if reward has improved
                    if running_reward > max_reward:
                        max_reward = running_reward
                        stopping_counter = 0
                        test = os.listdir(self.save_path)

                        for item in test:
                            if item.endswith(".weight"):
                                os.remove(os.path.join(self.save_path, item))
                        # Saving weights with better results
                        torch.save(self.state_dict(),
                                   os.path.join(self.save_path, 'model_{}.weight'.format(episode + 1)))
                    elif stopping_counter > episodes * 0.01:
                        print(f"The rewards did not improve since {50 * (stopping_counter - 1)} episodes")
                        self.env.stop()
                        break
                    else:
                        stopping_counter += 1
                    # Clearing averaging variables
                    running_reward = 0
                    done_average = 0

        # Saving final weights
        torch.save(self.state_dict(),
                   os.path.join(self.save_path, 'model_final.weight'))

    def eval_model(self, path, episode_nums):
        state_dicts = torch.load(path,
                                 map_location=torch.device('cpu')
                                 if "Windows" in platform.system() else torch.device('cuda'))
        self.load_state_dict(state_dicts)
        with torch.no_grad():
            for _ in range(episode_nums):
                state_list = [self.env.reset()]  # Reset environment and record the starting state
                for t in itertools.count():
                    action = self.select_action_probabilities(np.stack(state_list[-self.timesteps_observed:]))
                    # Step through environment using chosen action
                    state, reward, done, info = self.env.step(action.item())
                    state_list.append(state)
                    # Save reward
                    self.reward_episode.append(reward)
                    if done:
                        print(f"Reward: {sum(self.reward_episode):.3f}, and steps done: {t}")
                        print(f"Reason:{info['cause']}")
                        break
