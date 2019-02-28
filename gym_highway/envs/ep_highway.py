import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym_highway.modell.modell import Modell
from gym_highway.modell.environment_vehicle import CollisionExc

logger = logging.getLogger(__name__)


class EPHighWayEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):

        self.envdict = {'length_forward': 1000, 'length_backward': 500, 'dt': 0.2, 'lane_width': 4, 'lane_count': 3,
                        'density_lane0': 16, 'density_lane1': 8, 'speed_mean_lane0': 110.0 / 3.6,
                        'speed_std_lane0': 10.0 / 3.6, 'speed_mean_lane1': 150.0 / 3.6, 'speed_std_lane1': 10.0 / 3.6,
                        'speed_ego_desired': 130.0 / 3.6}
        # Vehicle Generation Parameters

        self.modell = None

        self.resetcounter = 0

        self._reset()

        low = np.array([0, -50, 0, -50, 0, -50, 0, -50, 0, -50, 0, -50, 0, 0, -5, -.5, 0])
        high = np.array([500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 1, 1, 25, .5, 50])

        alow = np.array([-0.003, -6.0])
        ahigh = np.array([0.003, 3.5])
        self.action_space = spaces.Discrete(25)
        # self.action_space = spaces.Box(alow,ahigh)
        self.observation_space = spaces.Box(low, high)
        self.cumulatedreward = 0
        self.rewards = [0, 0, 0]
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        # If resentcounter reached 0 ->New modell is created and warms up
        # if (self.modell is None) or (self.resetcounter<=0):

        self.envdict['density_lane0'] = np.random.randint(12, 16)  # 16 #[vehicle density vehicle/km]
        self.envdict['density_lane1'] = np.random.randint(8, 12)  # 8 #[vehicle density vehicle/km]
        self.envdict['density_lane2'] = np.random.randint(6, 10)  # 8 #[vehicle density vehicle/km]

        seb = np.random.randint(100, 120)

        self.envdict['speed_mean_lane0'] = seb / 3.6  # generated vehicle desired speed mean [m/s]
        self.envdict['speed_std_lane0'] = 10.0 / 3.6  # generated vehicle desired speed deviation [m/s]
        seb = (seb + np.random.randint(0, 20))
        self.envdict['speed_mean_lane1'] = seb / 3.6  # generated vehicle desired speed mean [m/s]
        self.envdict['speed_std_lane1'] = 10.0 / 3.6  # generated vehicle desired speed deviation [m/s]
        seb = (seb + np.random.randint(0, 20))
        self.envdict['speed_mean_lane2'] = seb / 3.6  # generated vehicle desired speed mean [m/s]
        self.envdict['speed_std_lane2'] = 10.0 / 3.6  # generated vehicle desired speed deviation [m/s]

        self.modell = Modell(self.envdict)
        self.modell.warmup(False)
        self.resetcounter = 10
        # Picking the EgoVehicle from modell
        self.modell.searchEgoVehicle()
        self.cumulatedreward = 0
        self.rewards = [0, 0, 0, 0]
        self.resetcounter = self.resetcounter - 1
        # Aquiring state from modell
        self.state = self.modell.generate_state_for_ego()
        return self.state

    def calcaction(self, action):
        st = [-0.003, -0.0005, 0, 0.0005, 0.003]
        ac = [-6.0, -2.0, 0.0, 2.0, 3.5]
        steer = st[action // 5]
        acc = ac[action % 5]
        ctrl = [steer, acc]
        return ctrl

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        try:
            ctrl = self.calcaction(action)
            isOk, cause = self.modell.onestep(ctrl)
        except CollisionExc:
            print('Collision exception')
            return self.state, -20.0, True, {'Collision exception'}
        except:
            isOk = False
            print('Internal exception')
            return self.state, -20.0, True, {'Internal exception'}  # Collision exception

        self.state = self.modell.generate_state_for_ego()
        terminated = not isOk
        if terminated:
            print(cause)
            reward = -20.0
            rewards = np.zeros(4)
        else:
            reward, rewards = self.calcreward()

        self.cumulatedreward = self.cumulatedreward + reward
        return self.state, reward, terminated, {'cause': cause, 'rewards': self.rewards}

    def calcreward(self):
        reward = 0
        # LANE BASED REWARD

        lreward = 0
        laneindex = self.modell.egovehicle.laneindex
        if laneindex > 0:
            if (self.state[13] == 0) and (self.state[4] > 30):
                lreward = -min(1, max(0, (self.state[4] - 50.0) / 20.0))

        # POSITION BASED REWARD
        # dy=abs(self.modell.egovehicle.y-laneindex*self.envdict['lane_width'])
        dy = abs(self.modell.egovehicle.y - (self.envdict['lane_count'] - 1.0) / 2 * self.envdict['lane_width'])
        ytresholdlow = (self.envdict['lane_count'] - 1.0) / 2.0 * self.envdict['lane_width'] + 0.3  # [m]
        ytresholdhigh = (self.envdict['lane_count']) / 2.0 * self.envdict['lane_width']  # [m]
        yrewhigh = 1.0
        yrewlow = 0.0
        if dy < ytresholdlow:
            yreward = yrewhigh
        elif dy > ytresholdhigh:
            yreward = yrewlow
        else:
            yreward = yrewhigh - (yrewhigh - yrewlow) * (dy - ytresholdlow) / (ytresholdhigh - ytresholdlow)
        # DESIRED SPEED BASED REWARD
        dv = abs(self.modell.egovehicle.desired_speed - self.state[16])
        vtresholdlow = 1  # [m/s]
        vtresholdhigh = 10  # self.modell.egovehicle.desired_speed #[m/s]
        vrewhigh = 1.0
        vrewlow = 0.1
        if dv < vtresholdlow:
            vreward = vrewhigh
        elif dv > vtresholdhigh:
            vreward = vrewlow
        else:
            vreward = vrewhigh - (vrewhigh - vrewlow) * (dv - vtresholdlow) / (vtresholdhigh - vtresholdlow)

        # Vehicle Closing Based Rewards
        cright = 0  # right safe zone
        cleft = 0  # left safe zone
        cfront = 0  # followed vehicle
        crear = 0  # following vehicle

        lw = self.envdict['lane_width']
        vehy = self.modell.egovehicle.y - self.modell.egovehicle.laneindex * lw

        # right safe zone
        if self.state[13] == 1:
            if vehy < -lw / 4:
                cright = max(-1, (vehy + lw / 4) / (lw / 4))
        # left safe zone
        if self.state[12] == 1:
            if vehy > lw / 4:
                cleft = max(-1, -(vehy - lw / 4) / (lw / 4))
        # front
        followingtime = self.state[2] / self.state[16]
        if followingtime < 1:
            cfront = followingtime - 1
        # rear
        followingtime = self.state[8] / self.state[16]
        if followingtime < 0.5:
            cfront = (followingtime - 0.5) * 2

        creward = max(-1, cright + cleft + cfront + crear)

        creward *= 1.0
        lreward *= 0.7
        yreward *= 0.1
        vreward *= 0.9

        reward = lreward + yreward + vreward + creward

        rewards = {'y': yreward, 'v': vreward, 'l': lreward, 'c': creward}

        self.rewards[0] += rewards['l']
        self.rewards[1] += rewards['y']
        self.rewards[2] += rewards['v']
        self.rewards[3] += rewards['c']

        return reward, rewards

    def _render(self, mode='human', close=False):
        self.modell.render(True, self.rewards)
