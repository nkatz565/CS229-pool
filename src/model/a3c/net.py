import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import set_init


class Net(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim, action_ranges=None, scale=100):
        super().__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        if action_ranges is not None:
            self.action_ranges = [(mn * scale, mx * scale) for mn, mx in action_ranges] # scale up for more efficient learning
        else:
            self.action_ranges = None
        self.scale = scale

        # Actor
        self.a1 = nn.Linear(s_dim, h_dim)
        self.mu = nn.Linear(h_dim, a_dim)
        self.sigma = nn.Linear(h_dim, a_dim)

        # Critic
        self.c1 = nn.Linear(s_dim, h_dim)
        self.v = nn.Linear(h_dim, 1)

        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def set_action_ranges(self, action_ranges):
        self.action_ranges = action_ranges

    def forward(self, x):
        a1 = F.relu(self.a1(x))
        # TODO: not sure if using sigmoid to compress the range is a good idea, since 0 and 1 are unapproachable values
        mu = 2 * torch.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001 # avoid 0

        c1 = F.relu(self.c1(x))
        values = self.v(c1)

        return mu, sigma, values

    def choose_action(self, s):
        self.train(False)
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu, sigma)
        a = m.sample().numpy()

        # Clip value
        if self.action_ranges is not None:
            for i in range(a.size):
                a[i] = a[i].clip(*self.action_ranges[i])
            a = a / self.scale
        return a

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        exp_v = log_prob * td.detach()
        a_loss = -exp_v

        entropy = -m.entropy() # for exploration
        total_loss = (a_loss + c_loss + 0.005 * entropy).mean()
        return total_loss

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
