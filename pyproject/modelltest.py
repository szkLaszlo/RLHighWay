import math

import gym
import matplotlib.pyplot as plt
import numpy as np

from gym_highway.modell import vehicle_modell as v
from gym_highway.modell.environment_vehicle import EnvironmentVehicle

env = gym.make('EPHighWay-v0')
envdict = {'length_forward': 1000, 'length_backward': 500, 'dt': 0.2, 'lane_width': 4, 'lane_count': 3,
                        'density_lane0': 16, 'density_lane1': 8, 'speed_mean_lane0': 110.0 / 3.6,
                        'speed_std_lane0': 10.0 / 3.6, 'speed_mean_lane1': 150.0 / 3.6, 'speed_std_lane1': 10.0 / 3.6,
                        'speed_ego_desired': 130.0 / 3.6}
veh = EnvironmentVehicle(envdict)
# np.array([x,y,th,v])
state = np.array([0.0, 0.0, 0, 10.0])
action = np.array([-1.8, 0])
dt = 0.1
for i in range(100):
    action[0] = 2 * math.sin(i / 10)
    state = v.vehicle_onestep(state, action, dt)
    x0, y0, th0, v0 = state
    print(state)
    plt.plot(x0, y0, '.')
plt.show(True)
