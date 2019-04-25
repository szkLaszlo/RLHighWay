import copy
import os
import random
import subprocess
import sys
import time
import warnings
from math import sin

import gym
import math
from gym import spaces
import numpy as np
import traci.constants as tc
import traci


class EPHighWayEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):

        self.rendering = None

        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        low = np.array(
            [-500, -50, -500, -50, -500, -50, -500, -50, -500, -50, -500, -50, -500, -50, -500, -50, 0, -1, -90, -10])
        high = np.array([500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 500, 50, 50, 3, 90, 20])
        self.action_space = spaces.Discrete(49)
        self.observation_space = spaces.Box(low, high)
        self.cumulated_reward = 0
        self.rewards = [0, 0, 0]
        self.lane_width = None
        self.lane_offset = None
        self.sumoBinary = None
        self.sumoCmd = None
        self.egoID = None
        self.state = None
        self.desired_speed = None
        self.dt = None

    def reset(self):
        if self.rendering is not None:
            try:
                for vehs in traci.vehicle.getIDList():
                    del vehs
                traci.close(False)
            except KeyError:
                pass
            except AttributeError:
                pass

            print("Starting SUMO")
            traci.start(self.sumoCmd)
            self.lane_width = traci.lane.getWidth('A_0')
            self.lane_offset = traci.junction.getPosition('J1')[1] - 3 * self.lane_width
            self.cumulated_reward = 0
            self.rewards = [0, 0, 0]
            self.egoID = None
            self.dt = traci.simulation.getDeltaT()
            self.desired_speed = random.randint(100, 140)
            self.desired_speed = self.desired_speed / 3.6
            self.state = None
            while self.egoID is None:
                self.one_step()
            self.state = self.get_surroundings()
            return self.state
        else:
            raise RuntimeError('Please run render before reset!')

    def calculate_action(self, action):
        st = [-0.03, -0.05, 0, 0.05, 0.03]
        ac = [-0.2, -0.1, 0.0, 0.1, 0.2]
        steer = st[action // 5]
        acc = ac[action % 5]
        ctrl = [steer, acc]
        return ctrl

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        ctrl = self.calculate_action(action)
        traci.vehicle.setSpeed(self.egoID, max(self.state['speed'] + ctrl[1], 0))
        traci.vehicle.setMaxSpeed(self.egoID, max(self.state['speed'] + ctrl[1], 1))
        self.state['angle'] += ctrl[0]
        if self.egoID is not None:
            self.state = self.get_surroundings()
            lane_new = traci.vehicle.getLaneID(self.egoID)
            if lane_new[-1] != self.state['lane']:
                if self.state['lane'] < 0 or self.state['lane'] > 2:
                    return self.state, -100, True, {'cause': 'Left Highway', 'rewards': [-100, -100, -100]}
                lane_new = lane_new[:-1] + str(self.state['lane'])
                x = traci.vehicle.getLanePosition(self.egoID)
                try:
                    traci.vehicle.moveTo(self.egoID, lane_new, x)
                except traci.exceptions.TraCIException:
                    self.state['lane']=int(traci.vehicle.getLaneID(self.egoID)[-1])
                    pass
            is_ok, cause = self.one_step()

        if self.egoID is not None and self.rendering and is_ok:
            egoPos = traci.vehicle.getPosition(self.egoID)
            traci.gui.setOffset('View #0', egoPos[0], egoPos[1])

        terminated = not is_ok
        if terminated and cause is not None:
            reward = -100.0
            self.cumulated_reward = reward
            self.rewards = [-100, -100, -100]
        else:
            reward, rewards = self.calculate_reward()

            self.cumulated_reward = self.cumulated_reward + reward

        return self.state, reward, terminated, {'cause': cause, 'rewards': self.rewards}

    def calculate_reward(self):
        reward = 0

        # LANE BASED REWARD

        lane_reward = 0
        lane_index = self.state['lane']
        if lane_index > 0:
            if (self.state['ER']['dx'] == 500) and (self.state['FR']['dx'] > 30):
                lane_reward = -min(1, max(0, (self.state['FR']['dx'] - 50.0) / 20.0))

        # POSITION BASED REWARD
        # dy=abs(self.modell.egovehicle.y-lane_index*self.envdict['lane_width'])
        dy = abs(self.state['y_pos'] - self.state['lane'] * self.lane_width)
        y_treshold_low = self.state['lane'] / 2.0 * self.lane_width + 0.3  # [m]
        y_treshold_high = self.state['lane'] / 2.0 * self.lane_width  # [m]
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
        dv = abs(self.desired_speed - self.state['speed'])
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

        lane_reward *= 0.7
        y_reward *= 0.1
        v_reward *= 0.9

        reward = lane_reward + y_reward + v_reward

        rewards = {'y': y_reward, 'v': v_reward, 'l': lane_reward}

        self.rewards[0] += rewards['l']
        self.rewards[1] += rewards['y']
        self.rewards[2] += rewards['v']

        return reward, rewards

    def render(self, mode='human', close=False):
        if mode == 'human':
            self.rendering = True
        else:
            self.rendering = False

        if self.rendering:
            self.sumoBinary = "C:/Sumo/bin/sumo-gui"
            self.sumoCmd = [self.sumoBinary, "-c", "../envs/jatek.sumocfg", "--start", "--quit-on-end",
                            "--collision.mingap-factor", "0", "--collision.action", "remove", "--no-warnings", "1"]
        else:
            self.sumoBinary = "C:/Sumo/bin/sumo"
            self.sumoCmd = [self.sumoBinary, "-c", "../envs/no_gui.sumocfg", "--start", "--quit-on-end",
                            "--collision.mingap-factor", "0", "--collision.action", "remove", "--no-warnings", "1"]

    def get_surroundings(self):
        cars_around = traci.vehicle.getContextSubscriptionResults(self.egoID)
        ego_data = cars_around[self.egoID]
        state = {}
        basic_vals = {'dx': 500, 'dv': 0}
        basic_keys = ['FL', 'FE', 'FR', 'RL', 'RE', 'RR', 'EL', 'ER']
        for keys in basic_keys:
            if keys in ['RL', 'RE', 'RR']:
                state[keys] = copy.copy(basic_vals)
                state[keys]['dv'] = 0
                state[keys]['dx'] = -500
            else:
                state[keys] = copy.copy(basic_vals)
        lane = {0: [], 1: [], 2: []}
        for keys in cars_around.keys():
            if keys is not self.egoID:
                new_car = dict()
                new_car['dx'] = cars_around[keys][tc.VAR_POSITION][0] - ego_data[tc.VAR_POSITION][0]
                new_car['dv'] = cars_around[keys][tc.VAR_SPEED] - ego_data[tc.VAR_SPEED]
                new_car['l'] = cars_around[keys][tc.VAR_LENGTH]
                lane[cars_around[keys][tc.VAR_LANE_INDEX]].append(new_car)
        [lane[i].sort(key=lambda x: x['dx']) for i in lane.keys()]
        for keys in lane.keys():
            if keys == ego_data[tc.VAR_LANE_INDEX]:
                for veh in lane[keys]:
                    if veh['dx'] - veh['l'] > 0:
                        if veh['dx'] - veh['l'] < state['FE']['dx']:
                            state['FE']['dx'] = veh['dx'] - veh['l']
                            state['FE']['dv'] = veh['dv']
                    elif veh['dx'] + ego_data[tc.VAR_LENGTH] < 0:
                        if veh['dx'] + ego_data[tc.VAR_LENGTH] > state['RE']['dx']:
                            state['RE']['dx'] = veh['dx'] + ego_data[tc.VAR_LENGTH]
                            state['RE']['dv'] = veh['dv']
            elif keys > ego_data[tc.VAR_LANE_INDEX]:
                for veh in lane[keys]:
                    if veh['dx'] - veh['l'] > 0:
                        if veh['dx'] - veh['l'] < state['FL']['dx']:
                            state['FL']['dx'] = veh['dx'] - veh['l']
                            state['FL']['dv'] = veh['dv']
                    elif veh['dx'] + ego_data[tc.VAR_LENGTH] < 0:
                        if veh['dx'] + ego_data[tc.VAR_LENGTH] > state['RL']['dx']:
                            state['RL']['dx'] = veh['dx'] + ego_data[tc.VAR_LENGTH]
                            state['RL']['dv'] = veh['dv']
                    else:
                        if veh['dx'] < state['EL']['dx']:
                            state['EL']['dx'] = veh['dx']
                            state['EL']['dv'] = veh['dv']
            elif keys < ego_data[tc.VAR_LANE_INDEX]:
                for veh in lane[keys]:
                    if veh['dx'] - veh['l'] > 0:
                        if veh['dx'] - veh['l'] < state['FR']['dx']:
                            state['FR']['dx'] = veh['dx'] - veh['l']
                            state['FR']['dv'] = veh['dv']
                    elif veh['dx'] + ego_data[tc.VAR_LENGTH] < 0:
                        if veh['dx'] + ego_data[tc.VAR_LENGTH] > state['RR']['dx']:
                            state['RR']['dx'] = veh['dx'] + ego_data[tc.VAR_LENGTH]
                            state['RR']['dv'] = veh['dv']
                    else:
                        if veh['dx'] < state['ER']['dx']:
                            state['ER']['dx'] = veh['dx']
                            state['ER']['dv'] = veh['dv']
        state['speed'] = ego_data[tc.VAR_SPEED]
        state['lane'] = ego_data[tc.VAR_LANE_INDEX]
        if self.state is not None:
            state['angle'] = self.state['angle']
            state['y_pos'] = self.state['y_pos'] + (state['speed']) * self.dt * sin(
                math.radians(state['angle']))
            if state['y_pos'] > (self.state['lane'] + 1) * self.lane_width:
                state['lane'] += 1
            elif state['y_pos'] < self.state['lane'] * self.lane_width:
                state['lane'] -= 1
        else:
            state['angle'] = 0
            state['y_pos'] = ego_data[tc.VAR_POSITION][1] - self.lane_offset + (state['speed']) * self.dt * sin(
                math.radians(state['angle']))

        return state

    def one_step(self):
        terminated = False
        w = traci.simulationStep()

        IDsOfVehicles = traci.vehicle.getIDList()
        if "ego" in IDsOfVehicles and self.egoID is None:
            self.egoID = "ego"
            lanes = [-1, 0, 1]
            traci.vehicle.setLaneChangeMode(self.egoID, 0x0)
            traci.vehicle.setSpeedMode(self.egoID, 0x0)
            traci.vehicle.subscribeContext(self.egoID, tc.CMD_GET_VEHICLE_VARIABLE, 0.0,
                                           [tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_ANGLE, tc.VAR_POSITION,
                                            tc.VAR_LENGTH])
            traci.vehicle.addSubscriptionFilterLanes(lanes, noOpposite=True, downstreamDist=100.0, upstreamDist=100.0)
        cause = None
        if self.egoID is not None:
            if self.egoID in traci.simulation.getCollidingVehiclesIDList():
                cause = "Collision"
            elif traci.vehicle.getSpeed(self.egoID) < (70 / 3.6):
                cause = 'Too Slow'
            else:
                cause = None
            if cause is not None:
                terminated = True
                self.egoID = None
        return (not terminated), cause

    @staticmethod
    def state_to_tuple(state):
        new_state = []
        for keys in ['FL', 'FE', 'FR', 'RL', 'RE', 'RR', 'EL', 'ER', 'speed', 'angle', 'y_pos', 'lane']:
            if keys not in state.keys():
                raise RuntimeError('Not valid state!')
            else:
                if dict is type(state[keys]):
                    for key in ['dx', 'dv']:
                        new_state.append(state[keys][key])
                else:
                    new_state.append(state[keys])
        return new_state
