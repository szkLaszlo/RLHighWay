import easygui as easygui
import gym

from sumoProject.agents.policyGradient import Policy

episode_nums = 10
path = easygui.fileopenbox()
env = gym.make('EPHighWay-v1')
env.render()
policy = Policy(env=env)
policy.eval_model(path=path, episode_nums=episode_nums)
