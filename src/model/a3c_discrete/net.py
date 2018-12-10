import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import set_init, norm, ratio


class Net(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim):
        super().__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        # Actor
        self.a1 = nn.Linear(s_dim, h_dim)
        self.a21 = nn.Linear(h_dim, a_dim[0])
        self.a22 = nn.Linear(h_dim, a_dim[1])

        # Critic
        self.c1 = nn.Linear(s_dim, h_dim)
        self.v = nn.Linear(h_dim, 1)

        set_init([self.a1, self.a21, self.a22, self.c1, self.v])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        a1 = F.relu(self.a1(x))
        logits1 = self.a21(a1)
        logits2 = self.a22(a1)
        c1 = F.relu(self.c1(x))
        values = self.v(c1)

        return logits1, logits2, values

    def choose_action(self, s):
        self.train(False)
        logits1, logits2, _ = self.forward(s)
        prob1 = F.softmax(logits1, dim=1).data
        prob2 = F.softmax(logits2, dim=1).data
        m1 = self.distribution(prob1)
        m2 = self.distribution(prob2)

        return np.array([m1.sample().numpy()[0], m2.sample().numpy()[0]])

    def wrap_action(self, action):
        a = np.copy(action) / self.a_dim
        return a

    def loss_func(self, s, a, v_t):
        self.train()
        logits1, logits2, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)
        m1 = self.distribution(probs1)
        m2 = self.distribution(probs2)
        exp_v1 = m1.log_prob(a[:, 0]) * td.detach().squeeze()
        exp_v2 = m2.log_prob(a[:, 1]) * td.detach().squeeze()
        a_loss = -exp_v1 - exp_v2
        total_loss = (c_loss + a_loss).mean()
        return total_loss

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
