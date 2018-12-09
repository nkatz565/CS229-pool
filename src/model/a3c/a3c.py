import math, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


from .net import Net
from .worker import Worker
from .shared_adam import SharedAdam
from ..env import PoolEnv


HIDDEN_DIM = 100
LR = 0.0002

def train(env_params, model_path, episodes=200, episode_length=50):
    print('Actor-Critic training')

    # Global network
    env = PoolEnv(**env_params)
    gnet = Net(env.state_space.n, env.action_space.n, HIDDEN_DIM)
    gnet.share_memory() # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=LR)  # global optimizer
    global_ep, global_ep_r = mp.Value('i', 0), mp.Value('d', 0.) # 'i': int, 'd': double

    # Parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, i, env_params, HIDDEN_DIM, episodes, episode_length) for i in range(mp.cpu_count())]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
