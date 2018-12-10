import math, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from .net import Net
from .worker import Worker, norm_state
from .shared_adam import SharedAdam
from ..env import PoolEnv
from .utils import v_wrap


HIDDEN_DIM = 100
LR = 0.002

def choose_action(state, model, action_space, w, h):
    s = norm_state(state, w, h)
    return model.clip_action(model.choose_action(v_wrap(s[None, :])))

def save_model(filepath, model):
    torch.save(model.state_dict(), filepath)

def load_model(filepath, model_params):
    model = Net(**model_params)
    model.load_state_dict(torch.load(filepath))
    return model

def train(env_params, model_path, episodes=200, episode_length=50):
    print('Actor-Critic training')

    # Global network
    env = PoolEnv(**env_params)
    gnet = Net(env.state_space.n, env.action_space.n, HIDDEN_DIM, action_ranges=env.action_space.ranges)
    gnet.share_memory() # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=LR)  # global optimizer
    global_ep, global_ep_r = mp.Value('i', 0), mp.Value('d', 0.) # 'i': int, 'd': double

    # Parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, i, env_params, HIDDEN_DIM, episodes, episode_length, model_path)
               for i in range(mp.cpu_count() // 2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    save_model(model_path, gnet)
