import sys


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python {} INPUT_FILE'.format(sys.argv[0]))
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, 'r') as fin:
        total_rewards = 0
        cnt = 0
        for line in fin:
            split_line = line.strip().split(' ')
            if len(split_line) == 3:
                break
            rewards = int(split_line[7])
            total_rewards += rewards
            cnt += 1
    print(total_rewards / cnt)
