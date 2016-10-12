import argparse
import gym
from agent import Q, Agent, Trainer


def create_cartpole_agent():
    env = gym.make("CartPole-v0")
    q = Q(env.action_space.n, env.observation_space, bin_size=10, low_bound=-7, high_bound=7)
    agent = Agent(q, epsilon=0.05)
    return env, agent


def main(episodes, render):
    env, agent = create_cartpole_agent()
    trainer = Trainer(agent, learning_rate=0.1, initial_exploration=600, epsilon_decay=0.001)
    trainer.train(env, episode_count=episodes, render=render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train & run cartpole ")
    parser.add_argument("--render", action="store_true", help="render the screen")
    parser.add_argument("--episode", type=int, default=1000, help="episode to train")

    args = parser.parse_args()

    main(args.episode, args.render)
