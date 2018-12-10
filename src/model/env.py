import numpy as np

from . import utils
from ..game import gamestate
from ..game import collisions
from ..game import event

class ActionSpace:
    def __init__(self, ranges):
        self.ranges = ranges
        self.buckets = None
        self.is_discrete = False

    @property
    def n(self):
        if self.is_discrete:
            return self.buckets
        else:
            return len(self.ranges)

    def set_buckets(self, buckets):
        self.buckets = buckets
        self.is_discrete = True

    def sample(self):
        if self.is_discrete:
            return [np.random.choice(bucket) for bucket in self.buckets]
        else:
            return [np.random.rand() * (mx - mn) + mn for mn, mx in self.ranges]

    def get_action(self, action):
        if self.is_discrete:
            real_action = []

            # Map discrete buckets to continuous values
            for i, a in enumerate(action):
                bucket = self.buckets[i]
                l, u = self.ranges[i]

                v = (a / bucket) * (u - l) + l
                real_action.append(v)

            return real_action
        else:
            return action

    def encode(self, action):
        """
        Deprecated.
        """
        t = 1
        encoded_action = 0
        for i, a in enumerate(action):
            encoded_action += t * a
            t *= self.buckets[i]
        return encoded_action

    def decode(self, action):
        """
        Deprecated.
        """
        decoded_action = []
        for bucket in self.buckets:
            decoded_action.append(action % bucket)
            action //= bucket
        return decoded_action

class StateSpace:
    def __init__(self, m, size):
        self.m = m
        self.w = size[0]
        self.h = size[1]
        self.buckets = None
        self.is_discrete = False

    @property
    def n(self):
        if self.is_discrete:
            return utils.prod(self.buckets) ** self.m
        else:
            return self.m * 2

    def set_buckets(self, buckets):
        self.buckets = buckets
        self.is_discrete = True

    def sample(self):
        if self.is_discrete:
            return np.random.choice(self.n)
        else:
            return [(np.random.rand() * self.w, np.random.rand() * self.h) for _ in range(self.m)]

    def get_state(self, observation):
        if not self.is_discrete:
            return np.asarray(list(sum(observation, ())), dtype=np.float64)
        else:
            state = [(0, 0)] * len(observation)
            bucket_x, bucket_y = self.buckets
            lx, ux = 0, self.w
            ly, uy = 0, self.h

            # Map continuous values to discrete buckets
            for i, (x, y) in enumerate(observation):
                sx = 0 if x <= lx else (bucket_x - 1 if x >= ux else int(((x - lx) / (ux - lx) * bucket_x)))
                sy = 0 if y <= ly else (bucket_y - 1 if y >= uy else int(((y - ly) / (uy - ly) * bucket_y)))

                state[i] = (sx, sy)

            # Encode state into consecutive state space
            unit_size = bucket_x * bucket_y
            encoded_state = 0
            for i, (sx, sy) in enumerate(state):
                encoded_state += (sy * bucket_y + sx) * (unit_size ** i)

            return encoded_state

class PoolEnv:
    def __init__(self, num_balls=2, visualize=False):
        self.num_balls = num_balls
        self.visualize = visualize

        # Two actions: angle, force
        # In the range of `ranges` in the game
        self.action_space = ActionSpace([(0, 1), (0, 1)])

        # State: a list of `m` ball (x,y) coordinates
        # Representing a table of (w, h) size
        self.state_space = StateSpace(num_balls, [1000, 1000])

        # Reward
        self.ball_in_reward = 5
        self.no_collision_penalty = -1

        # Init
        self.current_obs = None
        self.current_state = None
        self.gamestate = None
        self.reset()

    @property
    def max_reward(self):
        # Small bug here, should be (self.num_balls - 1) * self.ball_in_reward, but doesn't matter
        return self.num_balls * self.ball_in_reward

    @property
    def min_reward(self):
        return self.no_collision_penalty

    def set_buckets(self, action=None, state=None):
        if action is not None:
            self.action_space.set_buckets(action)
        if state is not None:
            self.state_space.set_buckets(state)

    def reset(self):
        self.gamestate = gamestate.GameState(self.num_balls, self.visualize)
        self.current_obs = self.gamestate.return_ball_state()
        self.current_state = self.state_space.get_state(self.current_obs)
        return self.current_state

    def step(self, action):
        real_action = self.action_space.get_action(action) # deal with discretized action
        game = self.gamestate
        ball_pos, holes_in, collision_count, done = game.step(game, real_action[0],real_action[1])

        self.current_obs = ball_pos
        self.current_state = self.state_space.get_state(ball_pos)
        reward = self.ball_in_reward * holes_in + (0 if collision_count > 0 else self.no_collision_penalty)
        return self.current_state, reward, done
