import os
import sys

from gym.envs.registration import register


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    os.system('export SUMO_HOME="/usr/share/sumo"')
#        sys.exit("please declare environment variable 'SUMO_HOME'")
print("runned")
