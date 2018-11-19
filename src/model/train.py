import argparse
import sys

from .env import PoolEnv
from . import q_table
from . import dqn
from . import ac


EPISODES = 200
EPISODE_LENGTH = 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL training.')
    parser.add_argument('--algo', type=str, default='q-table',
            help='One of q-table, dqn (Deep Q-Network), or ac (Actor-Critic). Default: q-table')
    parser.add_argument('--balls', type=int, default=2,
            help='Number of balls on table (including white ball), should be >= 2.')
    args = parser.parse_args()

    if args.balls < 2:
        print('Number of balls should be >= 2.')
        sys.exit(1)

    if args.algo == 'q-table':
        algo = q_table.train
        is_discrete = True
    elif args.algo == 'dqn':
        algo = dqn.train
        is_discrete = False
    elif args.algo == 'ac':
        algo = ac.train
        is_discrete = False
    else:
        print('Algorithm not supported! Should be one of q-table, dqn, or ac.')
        sys.exit(1)

    env = PoolEnv(args.balls, is_discrete=is_discrete)
    algo(env, episodes=EPISODES, episode_length=EPISODE_LENGTH)
