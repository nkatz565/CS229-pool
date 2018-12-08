import math, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


from .net import Net
from .worker import Worker
from .shared_adam import SharedAdam


HIDDEN_DIM = 100
LR = 0.0002

def train(balls, discrete, visualize, model_path, episodes=200, episode_length=50):
    print('Actor-Critic training')

    # Global network
    env = PoolEnv(balls, is_discrete=discrete, visualize=visualize)
    gnet = Net(env.state_space.n, env.action_space.n, HIDDEN_DIM)
    gnet.share_memory() # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=LR)  # global optimizer
    global_ep, global_ep_r = mp.Value('i', 0), mp.Value('d', 0.) # 'i': int, 'd': double
    max_reward = balls*5 #each ball is worth 5 points, a collision is worth 0
    # Parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, i, balls, discrete, visualize, HIDDEN_DIM, episodes, episode_length, max_reward) for i in range(mp.cpu_count())]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
