import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


EPS = 1000
A3C_ALGOS = ['a3c', 'a3c_discrete']
DQN_ALGOS = ['dqn']
OTHER_ALGOS = ['random', 'q_table']


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python {} OUTPUT_FILE'.format(sys.argv[0]))
        sys.exit(1)

    output_file = sys.argv[1]

    ys = []
    for algo in OTHER_ALGOS:
        with open('output/results/{}_{}.txt'.format(algo, EPS), 'r') as fin:
            running_rewards = 0
            y = []
            for line in fin:
                split_line = line.strip().split(' ')
                timesteps, rewards = int(split_line[3]), int(split_line[7])
                if running_rewards == 0:
                    running_rewards = rewards
                else:
                    running_rewards = running_rewards * 0.99 + rewards * 0.01
                y += [running_rewards]
        ys += [y[:EPS]]
    for algo in DQN_ALGOS:
        with open('output/results/{}_{}.txt'.format(algo, EPS), 'r') as fin:
            running_rewards = 0
            y = []
            for line in fin:
                split_line = line.strip().split(' ')
                timesteps, rewards = int(split_line[4]), int(split_line[8])
                if running_rewards == 0:
                    running_rewards = rewards
                else:
                    running_rewards = running_rewards * 0.99 + rewards * 0.01
                y += [running_rewards]
        ys += [y[:EPS]]
    for algo in A3C_ALGOS:
        with open('output/results/{}_{}.txt'.format(algo, EPS), 'r') as fin:
            x = []
            y = []
            for line in fin:
                split_line = line.strip().split(' ')
                timesteps, rewards = int(split_line[12][:-2]), float(split_line[10])
                x += [timesteps]
                y += [rewards]
        x, y = zip(*sorted(zip(x, y)))
        ys += [y[:EPS]]

    fig, ax = plt.subplots()
    x = list(range(EPS))
    for y in ys:
        ax.plot(x, y)
    ax.set(xlabel='episode', ylabel='avg rewards', title='Avg. Rewards Trend (exponential moving average)')
    ax.legend(OTHER_ALGOS + DQN_ALGOS + A3C_ALGOS)

    fig.savefig(output_file)
