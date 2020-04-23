import gym

from sumoProject.agents.eval_policyGrad import find_latest_weight
from sumoProject.agents.policyGradient import Policy

continues = False

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    path = None
    if continues:
        path = find_latest_weight()
    # Hyperparameters
    learning_rate = 0.001
    gamma = 0.99
    episodes = 20000
    policy = Policy(env=env, gamma=gamma, learning_rate=learning_rate, load_weights_path=path)
    policy.train_network(episodes=episodes)
