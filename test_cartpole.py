import unittest
import gym
from gym.spaces import Box
import numpy as np
from agent import Q


class TestCartPole(unittest.TestCase):

    def test_make_bins(self):
        env = gym.make("CartPole-v0")
        q = Q(env.action_space.n, env.observation_space, bin_size=7, low_bound=-3, high_bound=3)

        bin_range = (-2, 3)
        correct = np.arange(*bin_range).tolist()  # expected bins: ~-2, ~-1, ~0, ~1, ~2, ~3, (3~) = 7bin, 6 boundary
        bins = q.make_bins(*bin_range)
        self.assertEqual(tuple(correct), tuple(bins))

    def test_observation_to_state(self):
        dummy_observation_space = Box(-2, 3, (2,))
        q = Q(4, dummy_observation_space, bin_size=7, low_bound=-3, high_bound=3)

        state = q.observation_to_state(np.array([-3, 1]))
        self.assertEqual(state, 0 * q.bin_size ** 0 + 4 * q.bin_size ** 1)
                

if __name__ == "__main__":
    unittest.main()

