import gym

from sumoProject.agents.policyGradient import Policy

if __name__ == "__main__":
    env = gym.make('EPHighWay-v1')
    env.set_reward_type('complex')
    env.render(mode='none')
    # Hyperparameters
    learning_rate = 0.0001
    gamma = 0.99
    episodes = 200000
    policy = Policy(env=env, gamma=gamma, learning_rate=learning_rate)
    policy.train_network(episodes=episodes)
