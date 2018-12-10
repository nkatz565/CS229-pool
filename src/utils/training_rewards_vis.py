import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python {} INPUT_FILE OUTPUT_FILE'.format(sys.argv[0]))
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]

    y = []
    with open(input_file, 'r') as fin:
        running_rewards = 0
        for line in fin:
            split_line = line.strip().split(' ')
            timesteps, rewards = int(split_line[3]), int(split_line[7])
            if running_rewards == 0:
                running_rewards = rewards
            else:
                running_rewards = running_rewards * 0.99 + rewards * 0.01
            y += [running_rewards]
    x = list(range(len(y)))
    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='episode', ylabel='avg rewards', title='Avg. Rewards Trend (exponential moving average)')

    fig.savefig(output_file)
