import gym

from sumoProject.agents.eval_policyGrad import find_latest_weight
from sumoProject.agents.policyGradient import Policy

continues = True

if __name__ == "__main__":
    env = gym.make('EPHighWay-v1')
    env.render(mode='none')
    path = None
    if continues:
        path = find_latest_weight()
    # Hyperparameters
    learning_rate = 0.00005
    gamma = 0.6
    episodes = 200000
    policy = Policy(env=env, gamma=gamma, learning_rate=learning_rate, load_weights_path=path)
    policy.train_network(episodes=episodes)
