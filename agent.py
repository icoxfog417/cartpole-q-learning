import random
import copy
from collections import defaultdict
from collections import deque
from collections import namedtuple
import numpy as np


class Q():

    def __init__(self, n_actions, observation_space, bin_size, low_bound, high_bound, initial_mean=0.0, initial_std=0.0):
        self.n_actions = n_actions
        self.bin_size = bin_size
        self._low_bound = low_bound
        self._high_bound = high_bound
        self._initial_mean = initial_mean
        self._initial_std = initial_std

        # if we encounter the new observation, we initialize action evaluations
        self.table = defaultdict(lambda: self._initial_std * np.random.randn(self.n_actions) + self._initial_mean)
        self._observation_dimension = np.size(observation_space)
        self._dimension_bins = []

        for low, high in zip(observation_space.low, observation_space.high):
            bins = self.make_bins(low, high)
            self._dimension_bins.append(bins)

    def make_bins(self, low, high):
        _low = low if low != -np.Inf else self._low_bound
        _high = high if high != np.Inf else self._high_bound

        bins = np.arange(_low, _high, (float(_high) - float(_low)) / (self.bin_size - 2))  # exclude both ends
        if 0 not in bins:
            bins = np.sort(np.append(bins, [0]))  # 0 centric bins
        return bins
    
    def observation_to_state(self, observation):
        state = 0
        for d, o in enumerate(observation.flatten()):
            state = state + np.digitize(o, self._dimension_bins[d]) * pow(self.bin_size, d)  # bin_size numeral system
        return state
    
    def values(self, observation):
        state = self.observation_to_state(observation)
        return self.table[state]


class Agent():

    def __init__(self, q, epsilon=0.05):
        self.q = q
        self.epsilon = epsilon
    
    def act(self, observation):
        action = -1
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.q.n_actions)
        else:
            action = np.argmax(self.q.values(observation))
        
        return action


class Trainer():

    def __init__(self, agent, learning_rate=1, gamma=0.95, initial_exploration=0.3, initial_epsilon=1, epsilon_decay=1e-6):
        self.agent = agent
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.initial_exploration = initial_exploration
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.solved = False

    def train(self, env, episode_count, render=False):
        exploration = self.initial_exploration if isinstance(self.initial_exploration, int) else int(episode_count * self.initial_exploration)
        if episode_count < exploration:
            raise Exception("initial exploration count over the total episode count ({} vs {}).".format(exploration, episode_count))

        default_epsilon = self.agent.epsilon
        self.agent.epsilon = self.initial_epsilon
        values = []
        steps = deque(maxlen=100)
        lr = self.learning_rate
        for i in range(episode_count):
            obs = env.reset()
            step = 0
            done = False

            while not done:
                if render:
                    env.render()

                action = self.agent.act(obs)
                next_obs, reward, done, _ = env.step(action)

                state = self.agent.q.observation_to_state(obs)
                future = 0 if done else np.max(self.agent.q.values(next_obs))
                value = self.agent.q.table[state][action]
                self.agent.q.table[state][action] += lr * (reward + self.gamma * future - value)
        
                obs = next_obs
                values.append(value)
                step += 1
            else:
                mean = np.mean(values)
                steps.append(step)
                mean_step = np.mean(steps)
                print("Episode {}: {}steps(avg{}). epsilon={:.3f}, lr={:.3f}, mean q value={:.2f}".format(i, step, mean_step, self.agent.epsilon, lr, mean))
                self.agent.epsilon -= self.epsilon_decay
                if i > exploration or self.agent.epsilon < default_epsilon:
                    self.agent.epsilon = default_epsilon
                if i > exploration:
                    lr = self.learning_rate / ((i - exploration + 1) ** 0.5)
                if mean_step > 200:
                    self.solved = True
