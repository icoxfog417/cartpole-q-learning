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
        bins = q._make_bins(bin_range[0], bin_range[1], 7)
        self.assertEqual(tuple(correct), tuple(bins))
    
    def test_make_bins_multi_sizes(self):
        dummy_observation_space = Box(0, 6, (2,))
        q = Q(4, dummy_observation_space, bin_size=[3, 5])
        self.assertEqual(3 - 2, len(q._dimension_bins[0]))
        self.assertEqual(5 - 2, len(q._dimension_bins[1]))

    def test_make_bins_multi_bounds(self):
        dummy_observation_space = Box(-3, 3, (2,))
        q = Q(4, dummy_observation_space, bin_size=[3, 5], low_bound=[-2, -1], high_bound=[2, 1])
        self.assertEqual(-2, q._dimension_bins[0][0])
        self.assertEqual(-1, q._dimension_bins[1][0])
        self.assertLess(q._dimension_bins[0][-1], 2)
        self.assertLess(q._dimension_bins[1][-1], 1)

    def test_observation_to_state(self):
        dummy_observation_space = Box(-2, 3, (2,))
        bin_size = 7
        q = Q(4, dummy_observation_space, bin_size=bin_size, low_bound=-3, high_bound=3)

        state = q.observation_to_state(np.array([-3, 1]))
        self.assertEqual(state, 0 * bin_size ** 0 + 4 * bin_size ** 1)
                

if __name__ == "__main__":
    unittest.main()

