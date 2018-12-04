import torch
import torch.multiprocessing as mp

from .utils import v_wrap, set_init, push_and_pull


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, name, env):
        super().__init__()

    def run(self):
        # Sample usage of gnet:
        #   s = torch.tensor(env.state_space.sample()).view(-1) # flatten
        #   a = gnet.choose_action(s, ranges=env.action_space.ranges)
        pass
