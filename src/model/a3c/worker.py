import torch
import torch.multiprocessing as mp

from .utils import v_wrap, set_init, push_and_pull
from .net import Net
from ..env import PoolEnv

GAMMA = 0.9
GLOBAL_UPDATE_RATE = 5 #the network will sync with the global network ever X iterations
EPISODES = 200

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, name, balls, discrete, visualize, HIDDEN_DIM, episodes, episode_length, max_reward):
        super().__init__()
        self.debug_name = name
        self.balls = balls
        self.discrete = balls
        self.visualize = visualize
        self.HIDDEN_DIM = HIDDEN_DIM
        self.gnet = gnet
        self.opt = opt
        
        self.episodes = episodes 
        self.episode_length = episode_length
        self.global_ep = global_ep
        self.global_ep_r = global_ep_r
        self.max_reward = max_reward
        
    def run(self):

        # Sample usage of gnet:
        #   s = torch.tensor(env.state_space.sample()).view(-1) # flatten
        #   a = gnet.choose_action(s, ranges=env.action_space.ranges)
        env = PoolEnv(self.balls, is_discrete=self.discrete, visualize=self.visualize)
        self.local_net = Net(env.state_space.n, env.action_space.n, self.HIDDEN_DIM)
        total_steps = 1

        while self.global_ep.value < self.episodes:
            current_state = env.reset()
            state_buffer, action_buffer, reward_buffer = [], [], []
            current_reward = 0
            for i in range(0, self.episode_length):
                current_state = torch.tensor(env.state_space.sample()).view(-1) # rachel please verify these lines are working
                a = self.gnet.choose_action(current_state, ranges=env.action_space.ranges)
                next_state, reward_update, done = env.step(a) #get the updated state/rewards, and check if we're done
                if i == self.episode_length-1:
                    done = True #automatically end the episode if we've hit the max length
                current_reward+=reward_update
                action_buffer.append(a)
                state_buffer.append(current_state)
                reward_buffer.append((reward_update+self.max_reward)/self.max_reward)    # why the 8.1?
                
                if total_steps % GLOBAL_UPDATE_RATE == 0 or done: #ever GLOBAL_UPDATE_RATE we sync, or if we're done
                    push_and_pull(self.opt, self.local_net, self.gnet, done, next_state, state_buffer, action_buffer, reward_buffer, GAMMA)
                    state_buffer, action_buffer, reward_buffer = [], [], []
                    if done:
                        print('Worker {} finished. Episode finished after {} timesteps, total rewards {}'.format(self.debug_name, i+1, current_reward))
                        break
                current_state = next_state
                total_steps += 1
        