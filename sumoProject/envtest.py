import gym

import numpy as np
import math
import time
import sys
import matplotlib.pyplot as plt

global keyaction
keyaction = '0'
global xdata
xdata = []
global ydata
ydata = []


def press(event):
    global keyaction
    keyaction = event.key
    sys.stdout.flush()

st = [-0.003, -0.0005, 0, 0.0005, 0.003]
ac = [-6.0, -2.0, 0.0, 2.0, 3.5]

t = time.time()
env = gym.make('EPHighWay-v1')
#envs.render()
env.render(mode='jsd')
env.reset()

for _ in range(100):
    terminated = False
    while not terminated:
        action = np.random.randint(0,25)
        state, reward, terminated, info = env.step(action)
        if terminated:
            if info['cause'] is not None:
                print(info['cause'])
            else:
                print(sum(info['rewards'])+reward)
            env.reset()  # some comment wee added

elapsed = time.time() - t
print(elapsed)
print('DONE')
