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
        for line in fin:
            split_line = line.strip().split(' ')
            timesteps, rewards = int(split_line[3]), int(split_line[7])
            y += [rewards]
    x = list(range(len(y)))
    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='episode', ylabel='avg rewards', title='Avg. Rewards Trend')

    fig.savefig(output_file)
