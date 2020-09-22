import platform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ImageEncoder(nn.Module):

    def __init__(self, in_features, hidden_size, kernel_size=(7, 3), padding=(3, 1), stride=(4, 2)):
        super(ImageEncoder, self).__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_features[0], out_channels=hidden_size, kernel_size=kernel_size, padding=padding,
                      stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding,
                      stride=stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=1, kernel_size=kernel_size, padding=padding,
                      stride=stride),
        )

    def forward(self, x_in):
        # batch x channel x height x width
        if x_in.ndim == 3:
            x_in = x_in.unsqueeze(0)
        # batch x channel x height x width
        x_out = self.conv_encoder(x_in)
        return x_out


class ImageDecoder(nn.Module):

    def __init__(self, in_features, out_features, hidden_channels, stride, kernel_size, padding):
        super(ImageDecoder, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size,
                               padding=padding,
                               stride=stride),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               padding=padding,
                               stride=stride),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=out_features[0], kernel_size=kernel_size,
                               padding=padding,
                               stride=stride),
        )
        self.out_features = out_features
        self.input_size = in_features

    def forward(self, x_in):
        # batch x channel x height x width
        x_in = self.deconv(x_in)
        x_out = F.interpolate(x_in, self.out_features[1:], mode='bilinear')
        return x_out


class LSTMPredictor(nn.Module):

    def __init__(self, latent_space, hidden_size, output_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=latent_space, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input_, actions):
        # batch x time x features
        if input_.ndim > 3:
            input_ = input_.flatten(2)
        if actions.ndim < 3:
            actions = actions.unsqueeze(0)
        lstm_input = torch.cat([input_,actions], dim=-1)
        lstm_out, _ = self.lstm(lstm_input)
        output_ = self.linear(lstm_out[:, -1, :].unsqueeze(1))
        return output_


class ImageAutoencoderPredictor(nn.Module):

    def __init__(self, input_shape, hidden_channels, hidden_lstm, kernel_size, padding, stride, device='cuda', lr=0.001,
                 num_of_actions=2):
        super(ImageAutoencoderPredictor, self).__init__()
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.encoder = ImageEncoder(in_features=input_shape,
                                    hidden_size=hidden_channels,
                                    kernel_size=kernel_size, padding=padding, stride=stride).to(self.device)
        with torch.no_grad():
            sample_encoder_output = self.encoder(torch.ones(input_shape, device=self.device))
        self.encoder_output_size = sample_encoder_output.size()
        self.encoder_output_features = sample_encoder_output.flatten(2).size()[-1]
        self.decoder = ImageDecoder(in_features=self.encoder_output_size[-1], out_features=input_shape,
                                    hidden_channels=hidden_channels, stride=stride, kernel_size=kernel_size,
                                    padding=padding).to(self.device)

        self.predictor = LSTMPredictor(latent_space=self.encoder_output_features + num_of_actions,
                                       hidden_size=hidden_lstm, output_size=self.encoder_output_features).to(self.device)
        self.autoencoder_optim = torch.optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=lr)
        self.predictor_optim = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, input_, actions):
        # batch x channel x heights x width
        encoder_output = self.encoder(input_)
        # batch x channel x height x width
        decoder_output = self.decoder(encoder_output)

        # batch x time x channel x height x width
        predictor_input = encoder_output.unsqueeze(0)
        # batch x features
        predictor_output = self.predictor(predictor_input, actions)
        # batch x channel x height x width
        pred_dec_input = predictor_output.reshape(1,1,-1,encoder_output.shape[-1])
        # batch x channel x height x width
        pred_dec_output = self.decoder(pred_dec_input)
        # channel x height x width
        pred_dec_output = pred_dec_output.squeeze(0)

        return encoder_output, decoder_output, pred_dec_output

    def forward_and_loss(self, input_, target, actions):
        # batch x channel x heights x width
        input_ = torch.tensor(input_, dtype=torch.float32, device=self.device)
        target = torch.tensor(target, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        encoder_out, decoder_out, pred_dec_out = self.forward(input_, actions)

        autoencoder_loss = self.loss_fn(decoder_out, input_)
        autoencoder_loss.backward(retain_graph=True)
        self.autoencoder_optim.step()
        self.autoencoder_optim.zero_grad()

        predictor_loss = self.loss_fn(pred_dec_out, target)
        predictor_loss.backward()
        self.predictor_optim.step()
        self.predictor_optim.zero_grad()

        return autoencoder_loss.cpu().detach().item(), predictor_loss.cpu().detach().item()


class ImageProcessor(nn.Module):

    def __init__(self, hidden_size=128, output_size=(100, 2),
                 model_path="/home/st106/workspace/RLHighWay/sumoProject/agents/torchSummary/20200212_191313/model_final.weight"):
        super(ImageProcessor, self).__init__()
        device = torch.device('cpu') if "Windows" in platform.system() else torch.device('cuda')
        if model_path is not None:
            state_dicts = torch.load(model_path, map_location=device)
            hidden_size_conv = state_dicts['convolution.0.weight'].size()[0] // 2
        else:
            hidden_size_conv = hidden_size
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size_conv * 2, kernel_size=11, padding=5,
                      stride=1),
            nn.BatchNorm2d(hidden_size_conv * 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size_conv * 2, out_channels=hidden_size_conv * 2,
                      kernel_size=11, padding=5, stride=1),
            nn.BatchNorm2d(hidden_size_conv * 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size_conv * 2, out_channels=hidden_size_conv,
                      kernel_size=11, padding=5, stride=1),
            nn.BatchNorm2d(hidden_size_conv),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=hidden_size_conv, out_channels=1,
                      kernel_size=11, padding=5, stride=1),
            nn.BatchNorm2d(1),
            nn.AdaptiveMaxPool2d(output_size=output_size)
        ).to(device=device)
        self.device = device
        if model_path is not None:
            for key in list(state_dicts.keys()):
                if key not in self.state_dict().keys():
                    del state_dicts[key]
            self.load_state_dict(state_dicts)

    def forward(self, input_):
        if input_.ndim == 3:
            input_ = input_.unsqueeze(0).permute(0, 3, 1, 2)
        elif input_.ndim == 4:
            input_ = input_.permute(0, 3, 1, 2)
        u = self.convolution(input_).flatten(1)
        return u


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear((num_inputs + num_actions), hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear((num_inputs + num_actions), hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
