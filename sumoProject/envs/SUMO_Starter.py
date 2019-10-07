import copy
import math
import platform
import random
from math import sin

import gym
import numpy as np
import traci
import traci.constants as tc
from gym import spaces


class EPHighWayEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):

        self.max_punishment = -20
        self.steps_done = 0
        self.rendering = None

        low = np.array(
            [-200, -50, -200, -50, -200, -50, -200, -50, -200, -50, -200, -50, -200, -50, 0, -1, -90, -10])
        high = np.array([200, 50, 200, 50, 200, 50, 200, 50, 200, 50, 200, 50, 200, 50, 50, 3, 90, 20])
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.cumulated_reward = 0
        self.rewards = [0, 0, 0, 0]
        self.lane_width = None
        self.lane_offset = None
        self.sumoBinary = None
        self.sumoCmd = None
        self.egoID = None
        self.state = None
        self.desired_speed = None
        self.dt = None
        self.middle_counter = 0

    def stop(self):
        traci.close()

    def reset(self):
        if self.rendering is not None:
            try:
                for vehs in traci.vehicle.getIDList():
                    del vehs
            except KeyError:
                pass
            except AttributeError:
                pass
            except TypeError:
                pass

            traci.load(self.sumoCmd[1:])
            self.lane_width = traci.lane.getWidth('A_0')
            self.lane_offset = traci.junction.getPosition('J1')[1] - 2 * self.lane_width - self.lane_width / 2
            self.cumulated_reward = 0
            self.rewards = [0, 0, 0, 0]
            self.egoID = None
            self.steps_done = 0
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
        st = [-0.5, 0, 0.5]
        ac = [-0.7, 0.0, 0.3]
        steer = st[action // len(st)]
        acc = ac[action % len(st)]
        ctrl = [steer, acc]
        return ctrl

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        new_x, last_x = 0, 0
        IDsOfVehicles = traci.vehicle.getIDList()
        if "ego" in IDsOfVehicles:
            ctrl = self.calculate_action(action)
            traci.vehicle.setSpeed(self.egoID,
                                   min(max(self.state['speed'] + ctrl[1], 0), 50))  # todo hardcoded max speed
            traci.vehicle.setMaxSpeed(self.egoID, min(max(self.state['speed'] + ctrl[1], 1), 50))
            self.state['angle'] += ctrl[0]
            if self.egoID is not None:
                last_x = traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
                is_ok, cause = self.one_step()
                if is_ok:
                    self.state = self.get_surroundings()
                    lane_new = traci.vehicle.getLaneID(self.egoID)
                    if int(lane_new[-1]) != self.state['lane']:
                        if self.state['lane'] < 0 or self.state['lane'] > 2:
                            reward = self.max_punishment
                            self.cumulated_reward += reward
                            new_x = \
                                traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
                            return self.state, reward, True, {'cause': 'Left Highway',
                                                              'rewards': self.cumulated_reward,
                                                              'distance': 500 + new_x}
                        lane_new = lane_new[:-1] + str(self.state['lane'])
                        x = traci.vehicle.getLanePosition(self.egoID)
                        try:
                            traci.vehicle.moveTo(self.egoID, lane_new, x)
                        except traci.exceptions.TraCIException:
                            self.state['lane'] = int(traci.vehicle.getLaneID(self.egoID)[-1])
                            pass
                        self.state = self.get_surroundings(only_env_recheck=True)
                    new_x = traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
                else:
                    new_x = last_x + max(self.state['speed'] + ctrl[1], 0) * self.dt
        else:
            is_ok = False
            cause = None

        if self.egoID is not None and self.rendering and is_ok:
            egoPos = traci.vehicle.getPosition(self.egoID)
            traci.gui.setOffset('View #0', egoPos[0], egoPos[1])

        terminated = not is_ok
        reward = 0
        if terminated and cause is not None:
            reward = self.max_punishment
        elif not terminated:
            reward = new_x - last_x  # 1
            reward *= self.state['speed'] * 0.001
            if abs(self.state['y_pos']) > 0.3:
                self.middle_counter += 1
            else:
                self.middle_counter = 0
            if self.middle_counter > 100:
                reward = -1
            self.steps_done += 1
        else:
            reward = -self.max_punishment
        reward = reward
        self.cumulated_reward = self.cumulated_reward + reward
        return self.state, reward, terminated, {'cause': cause, 'rewards': self.cumulated_reward,
                                                'distance': 500 + new_x}

    def render(self, mode='human', close=False):
        if mode == 'human':
            self.rendering = True
        else:
            self.rendering = False

        if "Windows" in platform.system():
            if self.rendering:
                self.sumoBinary = "C:/Sumo/bin/sumo-gui"
                self.sumoCmd = [self.sumoBinary, "-c", "../envs/jatek.sumocfg", "--start", "--quit-on-end",
                                "--collision.mingap-factor", "0", "--collision.action", "remove", "--no-warnings", "1",
                                "--random"]
            else:
                self.sumoBinary = "C:/Sumo/bin/sumo"
                self.sumoCmd = [self.sumoBinary, "-c", "../envs/no_gui.sumocfg", "--start", "--quit-on-end",
                                "--collision.mingap-factor", "0", "--collision.action", "remove", "--no-warnings", "1",
                                "--random"]
        else:
            if self.rendering:
                self.sumoBinary = "/usr/share/sumo/bin/sumo-gui"
                self.sumoCmd = [self.sumoBinary, "-c", "../envs/jatek.sumocfg", "--start", "--quit-on-end",
                                "--collision.mingap-factor", "0", "--collision.action", "remove", "--no-warnings", "1",
                                "--random"]
            else:
                self.sumoBinary = "/usr/share/sumo/bin/sumo"
                self.sumoCmd = [self.sumoBinary, "-c", "../envs/no_gui.sumocfg", "--start", "--quit-on-end",
                                "--collision.mingap-factor", "0", "--collision.action", "remove", "--no-warnings", "1",
                                "--random"]

        traci.start(self.sumoCmd)

    def get_surroundings(self, only_env_recheck=False):
        cars_around = traci.vehicle.getContextSubscriptionResults(self.egoID)
        # traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
        ego_data = cars_around[self.egoID]
        state = {}
        basic_vals = {'dx': 200, 'dv': 0}
        basic_keys = ['FL', 'FE', 'FR', 'RL', 'RE', 'RR', 'EL', 'ER']
        for state_key in basic_keys:
            if state_key in ['RL', 'RE', 'RR']:
                state[state_key] = copy.copy(basic_vals)
                state[state_key]['dv'] = 0
                state[state_key]['dx'] = -200
            elif state_key in ['EL', 'ER']:
                # state[keys] = copy.copy(basic_vals)
                state[state_key] = 0
                # state[keys]['dx'] = 3 * self.lane_width + self.lane_offset - ego_data[tc.VAR_POSITION][1]
            # elif keys in ['ER']:
            #     state[keys] = copy.copy(basic_vals)
            #     state[keys]['dv'] = 0
            #     state[keys]['dx'] = ego_data[tc.VAR_POSITION][1] - self.lane_offset
            else:
                state[state_key] = copy.copy(basic_vals)
        lane = {0: [], 1: [], 2: []}
        for car_id in cars_around.keys():
            if car_id is not self.egoID:
                new_car = dict()
                new_car['dx'] = cars_around[car_id][tc.VAR_POSITION][0] - ego_data[tc.VAR_POSITION][0]
                new_car['dy'] = abs(cars_around[car_id][tc.VAR_POSITION][1] - ego_data[tc.VAR_POSITION][1])
                new_car['dv'] = cars_around[car_id][tc.VAR_SPEED] - ego_data[tc.VAR_SPEED]
                new_car['l'] = cars_around[car_id][tc.VAR_LENGTH]
                lane[cars_around[car_id][tc.VAR_LANE_INDEX]].append(new_car)
        [lane[i].sort(key=lambda x: x['dx']) for i in lane.keys()]
        for lane_id in lane.keys():
            if lane_id == ego_data[tc.VAR_LANE_INDEX]:
                for veh in lane[lane_id]:
                    if veh['dx'] - veh['l'] > 0:
                        if veh['dx'] - veh['l'] < state['FE']['dx']:
                            state['FE']['dx'] = veh['dx'] - veh['l']
                            state['FE']['dv'] = veh['dv']
                    elif veh['dx'] + ego_data[tc.VAR_LENGTH] < 0:
                        if veh['dx'] + ego_data[tc.VAR_LENGTH] > state['RE']['dx']:
                            state['RE']['dx'] = veh['dx'] + ego_data[tc.VAR_LENGTH]
                            state['RE']['dv'] = veh['dv']
            elif lane_id > ego_data[tc.VAR_LANE_INDEX]:
                for veh in lane[lane_id]:
                    if veh['dx'] - veh['l'] > 0:
                        if veh['dx'] - veh['l'] < state['FL']['dx']:
                            state['FL']['dx'] = veh['dx'] - veh['l']
                            state['FL']['dv'] = veh['dv']
                    elif veh['dx'] + ego_data[tc.VAR_LENGTH] < 0:
                        if veh['dx'] + ego_data[tc.VAR_LENGTH] > state['RL']['dx']:
                            state['RL']['dx'] = veh['dx'] + ego_data[tc.VAR_LENGTH]
                            state['RL']['dv'] = veh['dv']
                    else:
                        state['EL'] = 1
                        # if veh['dy'] < state['EL']['dx']:
                        #     state['EL']['dx'] = veh['dy']
                        #     state['EL']['dv'] = veh['dv']
            elif lane_id < ego_data[tc.VAR_LANE_INDEX]:
                for veh in lane[lane_id]:
                    if veh['dx'] - veh['l'] > 0:
                        if veh['dx'] - veh['l'] < state['FR']['dx']:
                            state['FR']['dx'] = veh['dx'] - veh['l']
                            state['FR']['dv'] = veh['dv']
                    elif veh['dx'] + ego_data[tc.VAR_LENGTH] < 0:
                        if veh['dx'] + ego_data[tc.VAR_LENGTH] > state['RR']['dx']:
                            state['RR']['dx'] = veh['dx'] + ego_data[tc.VAR_LENGTH]
                            state['RR']['dv'] = veh['dv']
                    else:
                        state['ER'] = 1
                        # if veh['dy'] < state['ER']['dx']:
                        #     state['ER']['dx'] = veh['dy']
                        #     state['ER']['dv'] = veh['dv']
        state['speed'] = ego_data[tc.VAR_SPEED]
        state['lane'] = ego_data[tc.VAR_LANE_INDEX]  # todo: onehot vector
        if self.state is not None:
            state['angle'] = self.state['angle']
            state['y_pos'] = (self.state['y_pos'] + (state['speed']) * self.dt * sin(
                math.radians(state['angle']))) if not only_env_recheck else self.state['y_pos']
            if state['y_pos'] > self.lane_width / 2:
                state['lane'] -= 1
                state['y_pos'] = -1 * (self.lane_width - state['y_pos'])
            elif state['y_pos'] < -self.lane_width / 2:
                state['lane'] += 1
                state['y_pos'] += self.lane_width

        else:
            state['angle'] = 0
            state['y_pos'] = ego_data[tc.VAR_POSITION][1] - self.lane_offset - \
                             (state['lane']) * self.lane_width \
                             + (state['speed']) * self.dt * sin(math.radians(state['angle']))
        if math.isclose(abs(state['y_pos']), 0, rel_tol=1e-4, abs_tol=1e-4):
            state['y_pos'] = 0.0
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
            traci.vehicle.setSpeed(self.egoID, self.desired_speed)
            traci.vehicle.setMaxSpeed(self.egoID, self.desired_speed)
            traci.vehicle.subscribeContext(self.egoID, tc.CMD_GET_VEHICLE_VARIABLE, 0.0,
                                           [tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_ANGLE, tc.VAR_POSITION,
                                            tc.VAR_LENGTH])
            traci.vehicle.addSubscriptionFilterLanes(lanes, noOpposite=True, downstreamDist=100.0, upstreamDist=100.0)
        cause = None
        if self.egoID is not None:
            if self.egoID in traci.simulation.getCollidingVehiclesIDList():
                cause = "Collision"
            elif self.egoID in traci.vehicle.getIDList() and traci.vehicle.getSpeed(self.egoID) < (50 / 3.6):
                cause = 'Too Slow'
            elif self.egoID in traci.simulation.getArrivedIDList():
                cause = None
                self.egoID = None
                terminated = True
            else:
                cause = None
            if cause is not None:
                terminated = True
                self.egoID = None
        return (not terminated), cause

    def state_to_tuple(self, state):
        new_state = []
        for keys in ['FL', 'FE', 'FR', 'RL', 'RE', 'RR', 'EL', 'ER', 'speed', 'angle', 'y_pos', 'lane']:
            if keys not in state.keys():
                raise RuntimeError('Not valid state!')
            else:
                if dict is type(state[keys]):
                    for key in ['dx', 'dv']:
                        if key == 'dx':
                            new_state.append(state[keys][key] / 200)
                        else:
                            new_state.append((state[keys][key] / 50))
                elif keys is "speed":
                    new_state.append(state[keys] / 50)
                elif keys is "angle":
                    new_state.append(state[keys] / 90)
                elif keys is "y_pos":
                    new_state.append(state[keys] / self.lane_width / 2)
                else:
                    new_state.append(state[keys])
        return new_state

    def calculate_reward(self):
        reward = 0

        # LANE BASED REWARD

        lane_reward = 0
        lane_index = self.state['lane']
        if lane_index > 0:
            if (self.state['ER']['dx'] == 200) and (self.state['FR']['dx'] > 30):
                lane_reward = -min(1, max(0, (self.state['FR']['dx'] - 50.0) / 20.0))

        # POSITION BASED REWARD
        # dy=abs(self.modell.egovehicle.y-lane_index*self.envdict['lane_width'])
        dy = abs(self.state['y_pos'] - self.state['lane'] * self.lane_width)
        y_treshold_low = self.state['lane'] / 2.0 * self.lane_width + 0.3  # [m]
        y_treshold_high = self.state['lane'] / 2.0 * self.lane_width - 0.3  # [m]
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

        # Vehicle Closing Based Rewards
        closing_right = 0  # right safe zone
        closing_left = 0  # left safe zone
        closing_front = 0  # followed vehicle
        closing_rear = 0  # following vehicle

        lane_width = self.lane_width
        vehicle_y = self.state['y_pos'] - self.state['lane'] * lane_width

        # right safe zone
        if self.state['ER']['dx'] != 200:
            if vehicle_y < -lane_width / 4:
                closing_right = max(-1, (vehicle_y + lane_width / 4) / (lane_width / 4))
        # left safe zone
        if self.state['EL']['dx'] != 200:
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
