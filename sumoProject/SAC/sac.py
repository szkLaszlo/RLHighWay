import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from sumoProject.SAC.model import GaussianPolicy, QNetwork, DeterministicPolicy, ImageProcessor, \
    ImageAutoencoderPredictor
from sumoProject.SAC.utils import soft_update, hard_update


class SAC(nn.Module):
    def __init__(self, input_image_size, action_space, args):
        super(SAC, self).__init__()
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.loss_factor = 100

        kernel_size = (7, 3)
        padding = (3, 1)

        max_pool = (2, 2)
        hidden_channels = 128
        hidden_lstm = 256
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        self.logdir = args.log_dir
        self.autoencoder = ImageAutoencoderPredictor(input_shape=input_image_size, hidden_channels=hidden_channels,
                                                     hidden_lstm=hidden_lstm, kernel_size=kernel_size,
                                                     padding=padding, stride=max_pool, lr=args.lr)
        self.autoencoder_target = ImageAutoencoderPredictor(input_shape=input_image_size,
                                                            hidden_channels=hidden_channels,
                                                            hidden_lstm=hidden_lstm, kernel_size=kernel_size,
                                                            padding=padding, stride=max_pool, lr=args.lr)
        self.critic = QNetwork(self.autoencoder.encoder_output_features, action_space.shape[0],
                               args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(self.autoencoder.encoder_output_features, action_space.shape[0],
                                      args.hidden_size).to(self.device)

        self.convolution = self.autoencoder.encoder

        hard_update(self.critic_target, self.critic)
        hard_update(self.autoencoder_target, self.autoencoder)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(self.autoencoder.encoder_output_features, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(self.autoencoder.encoder_output_features, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            state = self.convolution(state).flatten(1)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.squeeze(0).detach().cpu().numpy()

    def forward(self, memory, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_batch = self.convolution(next_state_batch).flatten(1)
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            state_batch = self.convolution(state_batch).flatten(1)
        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1,
                              next_q_value) * self.loss_factor  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2,
                              next_q_value) * self.loss_factor  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() * self.loss_factor
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))])

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            print('Updating target network softly')
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.autoencoder_target, self.autoencoder, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, autoencoder_path=None):
        model_path = os.path.join(self.logdir, 'models')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if actor_path is None:
            actor_path = os.path.join(model_path, f"actor_{suffix}.pkl")
        if critic_path is None:
            critic_path = os.path.join(model_path, f"critic_{suffix}.pkl")
        if autoencoder_path is None:
            autoencoder_path = os.path.join(model_path, f"ae_{suffix}.pkl")
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.autoencoder.state_dict(), autoencoder_path)
        print(f'Saving models to model path: {model_path}')

    # Load model parameters
    def load_model(self, actor_path, critic_path, autoencoder_path):
        print('Loading models from {}, {} and {}'.format(actor_path, critic_path, autoencoder_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if autoencoder_path is not None:
            self.autoencoder.load_state_dict(torch.load(autoencoder_path))
