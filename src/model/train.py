import argparse
import sys

from .env import PoolEnv
from .q_table import q_table
from .dqn import dqn
from .a3c import a3c
from .a3c_discrete import a3c_discrete


EPISODES = 1000
EPISODE_LENGTH = 25

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL training.')
    parser.add_argument('output_model', type=str,
            help='Output model path.')
    parser.add_argument('--algo', type=str, default='q-table',
            help='One of q-table, dqn (Deep Q-Network), a3c (Asynchronous Advantage Actor-Critic), a3c-discrete. Default: q-table')
    parser.add_argument('--balls', type=int, default=2,
            help='Number of balls on table (including white ball), should be >= 2. Default: 2')
    parser.add_argument('--visualize', dest='visualize', action='store_true',
            help='To see the visualization of the pool game.')
    args = parser.parse_args()

    if args.balls < 2:
        print('Number of balls should be >= 2.')
        sys.exit(1)

    single_env = True
    
    if args.algo == 'q-table':
        algo = q_table.train
    elif args.algo == 'dqn':
        algo = dqn.train
    elif args.algo == 'a3c':
        algo = a3c.train
        single_env = False
    elif args.algo == 'a3c-discrete':
        algo = a3c_discrete.train
        single_env = False
    else:
        print('Algorithm not supported! Should be one of q-table, dqn, a3c, or a3c-discrete.')
        sys.exit(1)

    if single_env:
        env = PoolEnv(args.balls, visualize=args.visualize)
        algo(env, args.output_model, episodes=EPISODES, episode_length=EPISODE_LENGTH)
    else:
        env_params = { 'num_balls': args.balls, 'visualize': args.visualize }
        algo(env_params, args.output_model, episodes=EPISODES, episode_length=EPISODE_LENGTH)
