import os

import gym

from sumoProject.agents.policyGradient import Policy


def find_latest_weight(path='torchSummary'):
    """

    :param path:
    :return:
    """
    name_list = os.listdir(path)
    full_list = [os.path.join(path, i) for i in name_list]
    time_sorted_list = sorted(full_list, key=os.path.getmtime)
    if os.path.isdir(time_sorted_list[-1]):
        latest_weight = find_latest_weight(path=time_sorted_list[-1])
        return latest_weight
    else:
        for i in range(1, len(time_sorted_list) + 1):
            if time_sorted_list[-i].endswith('.weight'):
                return time_sorted_list[-i]
    print(time_sorted_list)

episode_nums = 100
# path = easygui.fileopenbox()
path = find_latest_weight()
env = gym.make('EPHighWay-v1')
env.render('human')
policy = Policy(env=env, tb_summary=False)
policy.eval_model(path=path, episode_nums=episode_nums)
