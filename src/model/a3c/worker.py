import torch
import torch.multiprocessing as mp
import datetime

from .utils import v_wrap, set_init, push_and_pull, record, norm
from .net import Net
from ..env import PoolEnv


GLOBAL_UPDATE_RATE = 5 # the network will sync with the global network every X iterations

def norm_state(s, w, h):
    s[::2] /= w
    s[1::2] /= h
    return s

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, name, env_params, HIDDEN_DIM, episodes, episode_length, model_path=None):
        super().__init__()

        self.debug_name = name
        self.env_params = env_params
        self.HIDDEN_DIM = HIDDEN_DIM
        self.gnet = gnet
        self.opt = opt
        
        self.episodes = episodes 
        self.episode_length = episode_length
        self.g_ep = global_ep
        self.g_ep_r = global_ep_r

        self.gamma = 0.8 # reward discount factor

        self.model_path = model_path
        
    def run(self):
        env = PoolEnv(**self.env_params)
        self.lnet = Net(env.state_space.n, env.action_space.n, self.HIDDEN_DIM, action_ranges=env.action_space.ranges)

        total_steps = 1
        while self.g_ep.value < self.episodes:
            state = norm_state(env.reset(), env.state_space.w, env.state_space.h)
            state_buffer, action_buffer, reward_buffer = [], [], []
            rewards = 0 # accumulate rewards for each episode
            done = False
            for t in range(self.episode_length):
                action = self.lnet.choose_action(v_wrap(state[None, :]))
                a = self.lnet.clip_action(action)
                next_state, reward, done = env.step(a)
                next_state = norm_state(next_state, env.state_space.w, env.state_space.h)
                rewards += reward
                done = done or t == self.episode_length - 1

                action_buffer.append(action)
                state_buffer.append(state)
                reward_buffer.append(norm(reward, env.max_reward, env.min_reward))

                # Update global net, assign to local net
                if total_steps % GLOBAL_UPDATE_RATE == 0 or done:
                    push_and_pull(self.opt, self.lnet, self.gnet, done, next_state, state_buffer, action_buffer, reward_buffer, self.gamma)
                    state_buffer, action_buffer, reward_buffer = [], [], []

                # Transition to next state
                state = next_state
                total_steps += 1

                if done:
                    record(self.g_ep, self.g_ep_r, rewards)
                    print('Episode finished after {} timesteps, total rewards {}, global rewards {} (g_ep {}), action: {}, time: {}'.format(t+1, rewards, self.g_ep_r.value, self.g_ep.value, a, datetime.datetime.now().time()))
                    if self.model_path is not None:
                        self.gnet.save(self.model_path)
                    break
