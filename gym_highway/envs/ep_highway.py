import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from gym_highway.modell.modell import Modell
from gym_highway.modell.modell import Egovehicle
from gym_highway.modell.modell import Envvehicle
from gym_highway.modell.environment_vehicle import CollisionExc
import os
import datetime

logger = logging.getLogger(__name__)
"""
   The following section provides global parameters, to modify the environment
   For different types of agent behavior, modify 
   globals['action_space'] ('DISCRETE' 'CONTINUOUS' 'STATEMACHINE')
"""
globals = {}
# Highway geometry parameters
globals['length_forward'] = 1000  # Simulation scope in  front of the ego vehicle in [m]
globals['length_backward'] = 500  # Simulation scope behind the ego vehicle in [m]
globals['dt'] = 0.2  # Steptime in [s]
globals['lane_width'] = 4  # Width of one Lane in [m]
globals['lane_count'] = 3  # Number of lanes

# Agent action parameters
globals['action_space'] = 'DISCRETE'  # agent action type. Choose from: 'DISCRETE' 'CONTINUOUS' 'STATEMACHINE'

# parameters for DISCRETE action. Sets of steering angles (st in[rad]) and accelerations ([ac in m/s^2]) to chose from
globals['discreteparams'] = {'st': [-0.003, -0.0005, 0, 0.0005, 0.003], 'ac': [-6.0, -2.0, 0.0, 2.0, 3.5]}
# parameters for CONTINUOUS action. Lower and upper bounds (alow,ahigh)[steering [rad],acceleration [m/s^2]]
globals['continuousparams'] = {'alow': np.array([-0.003, -6.0]), 'ahigh': np.array([0.003, 3.5])}

# Vehicle Generation Parameters
globals['density_lanes_LB'] = 18  # Lower bound of the random density for one lane [vehicle/km]
globals['density_lanes_UB'] = 26  # Upper bound of the random density for one lane [vehicle/km]

# Vehicle desired speed generation parameters normal distribution N(mean,sigma^2) in [m/s] for each lane
# IMPORTANT! Need to add as many values, as globals['lane_count']
globals['speed_lane0'] = [30.0, 9.0]  # generated vehicle desired speed lane 0 [m/s]
globals['speed_lane1'] = [35.0, 9.0]  # generated vehicle desired speed lane 1 [m/s]
globals['speed_lane2'] = [40.0, 9.0]  # generated vehicle desired speed lane 2 [m/s]
globals['speed_lane3'] = [40.0, 9.0]  # generated vehicle desired speed lane 2 [m/s]
globals['speed_lane4'] = [40.0, 9.0]  # generated vehicle desired speed lane 2 [m/s]

# Agent vehicle desired Speed
globals['speed_ego_desired'] = 130.0 / 3.6  # Agent vehicle desired speed [m/s]

# Subreward Weights
globals['creward'] = 0.3 # Subreward weight for distances to other vehicles
globals['lreward'] = 0.3  # Subreward weight for keeping right behavior
globals['yreward'] = 0.1  # Subreward weight for (not) leaving highway
globals['vreward'] = 0.3  # Subreward weight for keeping desired speed


class EPHighWayEnv(gym.Env):
    def step(self, action):
        self._step(action)

    def reset(self):
        self._reset()

    def render(self, mode='human'):
        self._render(mode=mode)

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):

        self.minChangeDist = 200
        self.minFollowDist = 30
        self.envdict = globals
        self.modell = None

        # Set action space
        if self.envdict['action_space'] == 'CONTINUOUS':
            params = self.envdict['continuousparams']
            self.action_space = spaces.Box(params['alow'], params['ahigh'])
            self.actiontype = 0
        elif self.envdict['action_space'] == 'DISCRETE':
            params = self.envdict['discreteparams']
            self.st = params['st']
            self.stlen = len(self.st)
            self.ac = params['ac']
            self.aclen = len(self.ac)
            self.action_space = spaces.Discrete(self.stlen * self.aclen)
            self.actiontype = 1
        elif self.envdict['action_space'] == 'STATEMACHINE':
            self.action_space = spaces.Discrete(3)
            self.actiontype = 2
        else:
            raise (Exception('Unknown action_space'))
        self.envdict['actiontype'] = self.actiontype

        # Set observation space
        low = np.array([0, -50, 0, -50, 0, -50, 0, -50, 0, -50, 0, -50, 0, 0, -5, -.5, 0])
        high = np.array([500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 1, 1, 25, .5, 50])
        self.observation_space = spaces.Box(low, high)

        self.rewards = [0, 0, 0, 0]
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.envdict['density_lanes'] = np.random.randint(self.envdict['density_lanes_LB'], self.envdict[
            'density_lanes_UB'])  # [vehicle density vehicle/km]

        self.modell = Modell(self.envdict)
        self.modell.warmup(False)
        # Picking the EgoVehicle from modell
        self.modell.searchEgoVehicle()

        self.rewards = [0., 0., 0., 0.]
        # Aquiring state from modell
        self.state = self.modell.generate_state_for_ego()
        return self.state

    def calcaction(self, action):
        if self.actiontype == 0:
            return action
        if self.actiontype == 1:
            steer = self.st[action // self.aclen]
            acc = self.ac[action % self.aclen]
            ctrl = [steer, acc]
            return ctrl
        if self.actiontype == 2:
            return action

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        ctrl = self.calcaction(action)
        if self.actiontype != 2:
            isOk, cause = self.modell.onestep(ctrl)
            self.state = self.modell.generate_state_for_ego()
        else:
            while True:
                isOk, cause = self.modell.onestep(ctrl)
                self.state = self.modell.generate_state_for_ego()
                if (not isOk) or (self.modell.egovehicle.state == 'command_receive'):
                    break
        terminated = not isOk
        if terminated:
            print(cause)
            reward = -80.0
            rewards = np.zeros(4)
        else:
            reward, rewards = self.calcreward()

        return self.state, reward, terminated, {'cause': cause, 'rewards': self.rewards}

    def calcreward(self):

        #lane based
        lreward = 0
        laneindex = self.modell.egovehicle.laneindex
        desSped=self.modell.egovehicle.desired_speed
        if laneindex == 0:
            if abs(self.state[16] - desSped) < (desSped * 0.05):
                lreward = 1
            elif self.state[12] == 1:
                lreward = 1
            else:
                lreward = -1
        else:
            if abs(self.state[16] - desSped) < (desSped * 0.05) and (self.state[13] == 1 or (self.state[5] < desSped and self.state[4]< self.minChangeDist)):
                lreward = 1
            else:
                lreward = -1

        # POSITION BASED REWARD
        # dy=abs(self.modell.egovehicle.y-laneindex*self.envdict['lane_width'])
        dy = abs(self.modell.egovehicle.y - (laneindex * self.envdict['lane_width']))
        ytresholdlow = - 0.3 # [m]
        ytresholdhigh = + 0.3 # [m]
        yrewhigh = 1.0
        yrewlow = -1.0
        if dy < ytresholdhigh:
            yreward = yrewhigh
        else:
            yreward = yrewlow

        # DESIRED SPEED BASED REWARD
        dv = abs(self.modell.egovehicle.desired_speed - self.state[16])

        if dv<desSped*0.05:
            vreward = 1
        else:
            if (self.state[2] <= self.minFollowDist) and self.state[16] == self.state[3]:
                vreward = 1
            elif self.state[12] == 1 or (self.state[7] > desSped):
                vreward = 1
            elif self.state[0] < self.minFollowDist:
                vreward = 1
            else:
                vreward = -1

        # Vehicle Closing Based Rewards
        followingtime = self.state[2] / self.state[16]
        if followingtime < 2:
            if self.state[12] == 1 or self.state[0] < self.minFollowDist or (self.state[7] > desSped):
                creward = 1
            else:
                creward = -1
        else:
            creward = 1

        creward *= self.envdict['creward']
        lreward *= self.envdict['lreward']
        yreward *= self.envdict['yreward']
        vreward *= self.envdict['vreward']

        reward = lreward + yreward + vreward + creward
        rewards = {'y': yreward, 'v': vreward, 'l': lreward, 'c': creward}

        self.rewards[0] += rewards['l']
        self.rewards[1] += rewards['y']
        self.rewards[2] += rewards['v']
        self.rewards[3] += rewards['c']

        return reward, rewards

    def _render(self, mode='human', close=False):
        self.modell.render(True, self.rewards)
