import argparse
import sys
import pickle

from .env import PoolEnv
from . import q_table
from . import dqn
from . import a3c


EPISODES = 10 
EPISODE_LENGTH = 200

def load_model(filepath):
    with open(filepath, 'rb') as fin:
        model = pickle.load(fin)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL model evaluation.')
    parser.add_argument('--model', type=str, default='model.pkl',
            help='Input model path. Default: model.pkl')
    parser.add_argument('--algo', type=str, default='random',
            help='One of q-table, dqn (Deep Q-Network), or a3c (Asynchronous Advantage Actor-Critic). Default: random')
    parser.add_argument('--balls', type=int, default=2,
            help='Number of balls on table (including white ball), should be >= 2. Default: 2')
    parser.add_argument('--visualize', dest='visualize', action='store_true',
            help='To see the visualization of the pool game.')
    args = parser.parse_args()

    if args.balls < 2:
        print('Number of balls should be >= 2.')
        sys.exit(1)

    model = None
    if args.algo != 'random':
        model = load_model(args.model)

    if args.algo == 'random':
        choose_action = lambda state, model, action_space: action_space.sample()
        is_discrete = False
    elif args.algo == 'q-table':
        choose_action = q_table.choose_action
        is_discrete = True
    elif args.algo == 'dqn':
        choose_action = dqn.choose_action
        is_discrete = False
    elif args.algo == 'a3c':
        choose_action = a3c.choose_action
        is_discrete = False
    else:
        print('Algorithm not supported! Should be one of random, q-table, dqn, or a3c.')
        sys.exit(1)

    env = PoolEnv(args.balls, is_discrete=is_discrete, visualize=args.visualize)
    if is_discrete:
        env.set_buckets(action=[18, 5], state=[50, 50])

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
                total_rewards += rewards / (t + 1)
                break
        if not done:
            print('Episode finished after {} timesteps, total rewards {}'.format(EPISODE_LENGTH, rewards))
            total_rewards += rewards / EPISODE_LENGTH
    print('Average rewards: {}'.format(total_rewards / EPISODES))
