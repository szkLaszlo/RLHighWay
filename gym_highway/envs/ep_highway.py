import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym_highway.modell.model import Model
from gym_highway.modell.environment_vehicle import CollisionExc

logger = logging.getLogger(__name__)


class EPHighWayEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):

        self.env_dict = {'length_forward': 1000, 'length_backward': 500, 'dt': 0.2, 'lane_width': 4, 'lane_count': 3,
                         'density_lane0': 12, 'density_lane1': 8, 'speed_mean_lane0': 110.0 / 3.6,
                         'speed_std_lane0': 10.0 / 3.6, 'speed_mean_lane1': 150.0 / 3.6, 'speed_std_lane1': 10.0 / 3.6,
                         'speed_ego_desired': 130.0 / 3.6, 'car_length': 3, 'safe_zone_length': 2,
                         'max_acceleration': 2,
                         'max_deceleration': -6}
        # Vehicle Generation Parameters

        self.model = None

        self.reset_counter = 0

        self._reset()

        low = np.array([0, -50, 0, -50, 0, -50, 0, -50, 0, -50, 0, -50, 0, 0, -5, -.5, 0])
        high = np.array([500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 1, 1, 25, .5, 50])

        alow = np.array([-0.003, -6.0])
        ahigh = np.array([0.003, 3.5])
        self.action_space = spaces.Discrete(49)
        # self.action_space = spaces.Box(alow,ahigh)
        self.observation_space = spaces.Box(low, high)
        self.cumulated_reward = 0
        self.rewards = [0, 0, 0, 0]
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):

        # TODO: lehetne  lane-hez igazítani nem hardcodeolni csak három sávot
        self.env_dict['density_lane0'] = np.random.randint(12, 16)  # 16 #[vehicle density vehicle/km]
        self.env_dict['density_lane1'] = np.random.randint(8, 12)  # 8 #[vehicle density vehicle/km]
        self.env_dict['density_lane2'] = np.random.randint(6, 10)  # 8 #[vehicle density vehicle/km]

        seb = np.random.randint(100, 120)

        self.env_dict['speed_mean_lane0'] = seb / 3.6  # generated vehicle desired speed mean [m/s]
        self.env_dict['speed_std_lane0'] = 10.0 / 3.6  # generated vehicle desired speed deviation [m/s]
        seb = (seb + np.random.randint(0, 20))
        self.env_dict['speed_mean_lane1'] = seb / 3.6  # generated vehicle desired speed mean [m/s]
        self.env_dict['speed_std_lane1'] = 10.0 / 3.6  # generated vehicle desired speed deviation [m/s]
        seb = (seb + np.random.randint(0, 20))
        self.env_dict['speed_mean_lane2'] = seb / 3.6  # generated vehicle desired speed mean [m/s]
        self.env_dict['speed_std_lane2'] = 10.0 / 3.6  # generated vehicle desired speed deviation [m/s]

        self.model = Model(self.env_dict)
        self.model.generate_new_vehicles(1)
        self.model.warm_up(False)
        self.reset_counter = 10
        # Picking the EgoVehicle from model
        self.model.search_ego_vehicle()
        self.cumulated_reward = 0
        self.rewards = [0, 0, 0, 0]
        self.reset_counter = self.reset_counter - 1
        # Aquiring state from modell
        self.state = self.model.generate_state_for_ego()
        return self.state

    def calculate_action(self, action):
        st = [-0.003, -0.0005, 0, 0.0005, 0.003]
        ac = [-6.0, -2.0, 0.0, 2.0, 3.5]
        steer = st[action // 5]
        acc = ac[action % 5]
        ctrl = [steer, acc]
        return ctrl

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        try:
            ctrl = self.calculate_action(action)
            is_ok, cause = self.model.one_step(ctrl)
        except CollisionExc:
            print('Collision exception')
            return self.state, -20.0, True, {'Collision exception'}

        self.state = self.model.generate_state_for_ego()
        terminated = not is_ok
        if terminated:
            print(cause)
            reward = -20.0
        else:
            reward, rewards = self.calculate_reward()

        self.cumulated_reward = self.cumulated_reward + reward

        return self.state, reward, terminated, {'cause': cause, 'rewards': self.rewards}

    def calculate_reward(self):
        reward = 0

        # LANE BASED REWARD

        lane_reward = 0
        lane_index = self.model.ego_vehicle.lane_index
        if lane_index > 0:
            if (self.state['ER']['dx'] == 500) and (self.state['FR']['dx'] > 30):
                lane_reward = -min(1, max(0, (self.state['FR']['dx'] - 50.0) / 20.0))

        # POSITION BASED REWARD
        # dy=abs(self.modell.egovehicle.y-lane_index*self.envdict['lane_width'])
        dy = abs(self.model.ego_vehicle.y - (self.env_dict['lane_count'] - 1.0) / 2 * self.env_dict['lane_width'])
        y_treshold_low = (self.env_dict['lane_count'] - 1.0) / 2.0 * self.env_dict['lane_width'] + 0.3  # [m]
        y_treshold_high = (self.env_dict['lane_count']) / 2.0 * self.env_dict['lane_width']  # [m]
        y_reward_max = 1.0
        y_reward_low = 0.0
        if dy < y_treshold_low:
            y_reward = y_reward_max
        elif dy > y_treshold_high:
            y_reward = y_reward_low
        else:
            y_reward = y_reward_max - (y_reward_max - y_reward_low) * \
                       (dy - y_treshold_low) / (y_treshold_high - y_treshold_low)

        # DESIRED SPEED BASED REWARD
        dv = abs(self.model.ego_vehicle.desired_speed - self.state['speed'])
        v_treshold_low = 1  # [m/s]
        v_treshold_high = 10  # self.modell.egovehicle.desired_speed #[m/s]
        v_reward_high = 1.0
        v_reward_low = 0.1
        if dv < v_treshold_low:
            v_reward = v_reward_high
        elif dv > v_treshold_high:
            v_reward = v_reward_low
        else:
            v_reward = v_reward_high - (v_reward_high - v_reward_low) * \
                       (dv - v_treshold_low) / (v_treshold_high - v_treshold_low)

        # Vehicle Closing Based Rewards
        closing_right = 0  # right safe zone
        closing_left = 0  # left safe zone
        closing_front = 0  # followed vehicle
        closing_rear = 0  # following vehicle

        lane_width = self.env_dict['lane_width']
        vehicle_y = self.model.ego_vehicle.y - self.model.ego_vehicle.lane_index * lane_width

        # right safe zone
        if self.state['ER']['dx'] != 500:
            if vehicle_y < -lane_width / 4:
                closing_right = max(-1, (vehicle_y + lane_width / 4) / (lane_width / 4))
        # left safe zone
        if self.state['EL']['dx'] != 500:
            if vehicle_y > lane_width / 4:
                closing_left = max(-1, -(vehicle_y - lane_width / 4) / (lane_width / 4))
        # front
        following_time = self.state['FE']['dx'] / self.state['speed']
        if following_time < 1:
            closing_front = following_time - 1
        # rear
        following_time = self.state['RE']['dx'] / self.state['speed']
        if following_time < 0.5:
            closing_front = (following_time - 0.5) * 2

        closing_reward = max(-1, closing_right + closing_left + closing_front + closing_rear)

        closing_reward *= 1.0
        lane_reward *= 0.7
        y_reward *= 0.1
        v_reward *= 0.9

        reward = lane_reward + y_reward + v_reward + closing_reward

        rewards = {'y': y_reward, 'v': v_reward, 'l': lane_reward, 'c': closing_reward}

        self.rewards[0] += rewards['l']
        self.rewards[1] += rewards['y']
        self.rewards[2] += rewards['v']
        self.rewards[3] += rewards['c']

        return reward, rewards

    def _render(self, mode='human', close=False):
        self.model.render(True, self.rewards, zoom=2)

