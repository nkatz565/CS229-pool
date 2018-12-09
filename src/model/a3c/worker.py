import torch
import torch.multiprocessing as mp

from .utils import v_wrap, set_init, push_and_pull
from .net import Net
from ..env import PoolEnv


GLOBAL_UPDATE_RATE = 5 # the network will sync with the global network every X iterations

def to_tensor(s):
    """Wraps a state representation into flat tensor"""
    return torch.tensor(s).float().view(-1)

def norm(v, max_v, min_v):
    return (v - (max_v + min_v) / 2) / (max_v - min_v)

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, name, env_params, HIDDEN_DIM, episodes, episode_length, save_model=None):
        super().__init__()

        self.debug_name = name
        self.env_params = env_params
        self.HIDDEN_DIM = HIDDEN_DIM
        self.gnet = gnet
        self.opt = opt
        
        self.episodes = episodes 
        self.episode_length = episode_length
        self.global_ep = global_ep
        self.global_ep_r = global_ep_r

        self.gamma = 0.8 # reward discount factor

        self.save_model = save_model
        
    def run(self):
        env = PoolEnv(**self.env_params)
        self.lnet = Net(env.state_space.n, env.action_space.n, self.HIDDEN_DIM, action_ranges=env.action_space.ranges)

        total_steps = 1
        while self.global_ep.value < self.episodes:
            state = env.reset()
            state_buffer, action_buffer, reward_buffer = [], [], []
            rewards = 0 # accumulate rewards for each episode
            done = False
            for t in range(self.episode_length):
                # Agent takes action using epsilon-greedy algorithm, get reward
                action = self.lnet.choose_action(to_tensor(state))
                next_state, reward, done = env.step(action)
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
                    print('Episode finished after {} timesteps, total rewards {} (worker {})'.format(t+1, rewards, self.debug_name))
                    break
            if self.save_model is not None:
                self.save_model(self.gnet)
