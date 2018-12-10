import argparse
import sys
import pickle

from .env import PoolEnv
from .q_table import q_table
from .dqn import dqn
from .a3c import a3c
from .a3c_discrete import a3c_discrete


EPISODES = 100
EPISODE_LENGTH = 25

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL model evaluation.')
    parser.add_argument('--model', type=str, default='model.pkl',
            help='Input model path. Default: model.pkl')
    parser.add_argument('--algo', type=str, default='random',
            help='One of q-table, dqn (Deep Q-Network), a3c (Asynchronous Advantage Actor-Critic), or a3c-discrete. Default: random')
    parser.add_argument('--balls', type=int, default=2,
            help='Number of balls on table (including white ball), should be >= 2. Default: 2')
    parser.add_argument('--visualize', dest='visualize', action='store_true',
            help='To see the visualization of the pool game.')
    args = parser.parse_args()

    if args.balls < 2:
        print('Number of balls should be >= 2.')
        sys.exit(1)

    env = PoolEnv(args.balls, visualize=args.visualize)
    model = None
    if args.algo == 'random':
        choose_action = lambda state, model, action_space: action_space.sample()
    elif args.algo == 'q-table':
        choose_action = q_table.choose_action
        env.set_buckets(action=[18, 5], state=[50, 50])
        model = q_table.load_model(args.model)
    elif args.algo == 'dqn':
        choose_action = dqn.choose_action
        env.set_buckets(action=[360, 5])
        model_params = { 's_dim': env.state_space.n,
                         'a_dim': env.action_space.n,
                         'buckets': env.action_space.buckets}
        model = dqn.load_model(args.model, model_params)
    elif args.algo == 'a3c':
        choose_action = lambda s, m, a_s: a3c.choose_action(s, m, a_s, env.state_space.w, env.state_space.h)
        model_params = { 's_dim': env.state_space.n,
                         'a_dim': env.action_space.n,
                         'h_dim': a3c.HIDDEN_DIM,
                         'action_ranges': env.action_space.ranges}
        model = a3c.load_model(args.model, model_params)
    elif args.algo == 'a3c-discrete':
        choose_action = lambda s, m, a_s: a3c_discrete.choose_action(s, m, a_s, env.state_space.w, env.state_space.h)
        env.set_buckets(action=[360, 1])
        model_params = { 's_dim': env.state_space.n,
                         'a_dim': env.action_space.n,
                         'h_dim': a3c_discrete.HIDDEN_DIM}
        model = a3c_discrete.load_model(args.model, model_params)
    else:
        print('Algorithm not supported! Should be one of random, q-table, dqn, or a3c.')
        sys.exit(1)

    total_rewards = 0
    for i_episode in range(EPISODES):
        state = env.reset()
        rewards = 0
        done = False
        for t in range(EPISODE_LENGTH):
            action = choose_action(state, model, env.action_space)
            next_state, reward, done = env.step(action)
            rewards += reward
            state = next_state

            if done:
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                total_rewards += rewards
                break
        if not done:
            print('Episode finished after {} timesteps, total rewards {}'.format(EPISODE_LENGTH, rewards))
            total_rewards += rewards
    print('Average rewards: {}'.format(total_rewards / EPISODES))
