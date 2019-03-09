import gym

import gym_highway
import numpy as np
import math
import time
import sys
import matplotlib.pyplot as plt
from msvcrt import getch, kbhit

env = gym.make('EPHighWay-v0')
env.reset()

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


def contaction(x):
    return {
        '4': np.array([0.003, 0.]),
        '6': np.array([-0.003, 0.]),
        '8': np.array([0.000, 1.]),
        '2': np.array([0.000, -1.]),
        '0': np.array([0.000, 0.]),
    }.get(x, np.array([0.000, 0.]))


def discaction(x):
    return {
        'up': 22,
        'down': 2,
        'right': 13,
        'left': 11,
        '0': 12,
    }.get(x, 12)


def plotstate(i, st):
    plt.figure(2)
    xstate = [None]*9
    ystate = [None]*9
    xstate[0] = st['FL']['dx']
    xstate[1] = st['FE']['dx']
    xstate[2] = st['FR']['dx']
    xstate[3] = -st['RL']['dx']
    xstate[4] = -st['RE']['dx']
    xstate[5] = -st['RR']['dx']
    xstate[6] = 0
    xstate[7] = 0
    xstate[8] = 0
    ystate[0] = st['pos_y']+4
    ystate[1] = 0
    ystate[2] = st['pos_y']-4
    ystate[3] = st['pos_y']+4
    ystate[4] = 0
    ystate[5] = st['pos_y']-4
    ystate[6] = st['pos_y']+4*st['EL']['dx']/500
    ystate[7] = st['pos_y']-4*st['ER']['dx']/500
    ystate[8] = st['pos_y']
    xdata = xstate
    ydata = ystate
    plt.cla()
    plt.plot(xdata, ydata, 'r*')
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)
    plt.figure(1)


def smaction(x):
    return {
        '4': 0,
        '6': 1
    }.get(x, 2)


st = [-0.003, -0.0005, 0, 0.0005, 0.003]
ac = [-6.0, -2.0, 0.0, 2.0, 3.5]

action = 12
t = time.time()
for i in range(10000):
    env.render(mode='human')
    plt.gcf().canvas.mpl_connect('key_press_event', press)


    action = discaction(keyaction)
    keyaction = '0'

    state, reward, terminated, cause = env.step(action)

    plotstate(i, state)
    # if i % 200 == 0:
    #    plt.close('all')
    action = 12  # np.random.randint(0,25)
    if terminated:
        # print(state)
        # print(reward)
        # print(cause)
        # action = 12
        xdata = []
        ydata = []

        env.reset()  # some comment wee added
elapsed = time.time() - t
print(elapsed)
print('DONE')
