import os
import argparse
import gym
from agent import Q, Agent, Trainer


RECORD_PATH = os.path.join(os.path.dirname(__file__), "./upload")


def create_cartpole_agent():
    env = gym.make("CartPole-v0")
    q = Q(env.action_space.n, env.observation_space, bin_size=7, low_bound=-5, high_bound=5)
    agent = Agent(q, epsilon=0.05)
    return env, agent


def main(episodes, render, monitor):
    env, agent = create_cartpole_agent()
    if monitor:
        env.monitor.start(RECORD_PATH)

    trainer = Trainer(agent, learning_rate=1.0, initial_exploration=1000, initial_epsilon=1.0, epsilon_decay=0.001)
    trainer.train(env, episode_count=episodes, render=render)
    print("solved!" if trainer.solved else "not solved...")

    if monitor:
        env.monitor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train & run cartpole ")
    parser.add_argument("--episode", type=int, default=1000, help="episode to train")
    parser.add_argument("--render", action="store_true", help="render the screen")
    parser.add_argument("--monitor", action="store_true", help="monitor")
    parser.add_argument("--upload", type=str, default="", help="upload key to openai gym (training is not executed)")

    args = parser.parse_args()

    if args.upload:
        if os.path.isdir(RECORD_PATH):
            gym.upload(RECORD_PATH, api_key=args.upload)
    else:
        main(args.episode, args.render, args.monitor)
