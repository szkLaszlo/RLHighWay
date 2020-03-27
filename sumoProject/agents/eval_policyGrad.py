import os

import gym

from sumoProject.agents.policyGradient import Policy


def find_latest_weight(path='torchSummary', file_end='.weight', exclude_end='.0'):
    """

    :param path:
    :return:
    """
    name_list = os.listdir(path)
    full_list = [os.path.join(path, i) for i in name_list]
    time_sorted_list = sorted(full_list, key=os.path.getmtime)
    for i in range(len(time_sorted_list)):
        if time_sorted_list[i].endswith(exclude_end):
            del time_sorted_list[i]
    if os.path.isdir(time_sorted_list[-1]):
        latest_weight = find_latest_weight(path=time_sorted_list[-1], file_end=file_end, exclude_end=exclude_end)
        return latest_weight
    else:
        for i in range(len(time_sorted_list) - 1, -1, -1):
            if time_sorted_list[-i].endswith(file_end):
                return time_sorted_list[i]
    print(time_sorted_list)


if __name__ == "__main__":
    episode_nums = 100
    # path = easygui.fileopenbox()
    path = find_latest_weight()
    env = gym.make('EPHighWay-v1')
    env.render('human')
    policy = Policy(env=env, tb_summary=False, load_weights_path=path)
    policy.eval_model(episode_nums=episode_nums)
