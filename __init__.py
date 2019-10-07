import os
import sys

from gym.envs.registration import register

# register(
#     id='EPHighWay-v1',
#     entry_point='sumoProject.envs:EPHighWayEnv',
# )

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
        os.system('export SUMO_HOME="/usr/share/sumo"')
print("runned")
