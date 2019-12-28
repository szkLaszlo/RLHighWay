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

        self.max_punishment = -2
        self.steps_done = 0
        self.rendering = None

        low = np.array(
            [-200, -50, -200, -50, -200, -50, -200, -50, -200, -50, -200, -50, -200, -50, 0, -1, -90, -10, 0])
        high = np.array([200, 50, 200, 50, 200, 50, 200, 50, 200, 50, 200, 50, 200, 50, 50, 3, 90, 20, 50])
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
        self.ego_start_position = 100000
        self.middle_counter = 0
        self.reward_type = "simple"
        self.environment_state = 0
        self.lanechange_counter = 0
        self.wants_to_change = []
        self.change_after = 5
        self.min_departed_vehicles = np.random.randint(40, 60, 1).item()
        self.environment_state_list = []

    def set_reward_type(self, reward_type):
        self.reward_type = reward_type

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
            self.middle_counter = 0
            self.ego_start_position = 100000
            self.lanechange_counter = 0
            self.wants_to_change = []
            self.change_after = 2
            self.min_departed_vehicles = np.random.randint(40, 60, 1).item()
            while self.egoID is None:
                self.one_step()
            self.environment_state_list = []
            self.environment_state = self.get_surroundings_env()
            self.state['velocity'] = self.desired_speed - 5
            return self.environment_state
        else:
            raise RuntimeError('Please run render before reset!')

    def calculate_action(self, action):
        st = [-1, 0, 1]
        ac = [-0.7, 0.0, 0.3]
        steer = st[action // len(st)]
        acc = ac[action % len(st)]
        ctrl = [steer, acc]
        return ctrl

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        new_x, last_x = 0, 0
        IDsOfVehicles = traci.vehicle.getIDList()
        reward = 0
        if self.egoID in IDsOfVehicles:
            ctrl = self.calculate_action(action)
            # traci.vehicle.setMaxSpeed(self.egoID, min(max(self.state['speed'] + ctrl[1], 1), 50))
            traci.vehicle.setSpeed(self.egoID,
                                   min(max(self.state['velocity'] + ctrl[1], 0), 50))  # todo hardcoded max speed
            self.wants_to_change.append(ctrl[0])
            if sum(self.wants_to_change) > self.change_after or sum(self.wants_to_change) < -self.change_after:
                self.lanechange_counter += 1
                last_lane = traci.vehicle.getLaneID(self.egoID)[:-1]
                lane_new = int(traci.vehicle.getLaneID(self.egoID)[-1]) + ctrl[0]
                if lane_new not in [0, 1, 2]:
                    reward = self.max_punishment
                    self.cumulated_reward += reward
                    new_x = \
                        traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
                    return self.environment_state, reward, True, {'cause': 'Left Highway',
                                                                  'rewards': self.cumulated_reward,
                                                                  'velocity': self.state['velocity'],
                                                                  'distance': new_x - self.ego_start_position,
                                                                  'lane_change': self.lanechange_counter}
                else:
                    reward = 1
                    self.wants_to_change = []
                    lane_new = last_lane + str(lane_new)
                    x = traci.vehicle.getLanePosition(self.egoID)
                    # traci.vehicle.setRoute(self.egoID, [lane_new[:-2]])
                    done = False
                    while not done:
                        try:
                            traci.vehicle.moveTo(self.egoID, lane_new, x)
                        except traci.exceptions.TraCIException:
                            x += 0.1
                        else:
                            done = True
            if len(self.wants_to_change) > self.change_after:
                self.wants_to_change.pop(0)
            if self.egoID is not None:
                last_x = traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
                is_ok, cause = self.one_step()
                if is_ok:
                    environment_state = self.calculate_environment()
                    new_x = traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
                else:
                    new_x = last_x + max(self.state['velocity'] + ctrl[1], 0) * self.dt
        else:
            is_ok = False
            cause = None

        if self.egoID is not None and self.rendering and is_ok:
            egoPos = traci.vehicle.getPosition(self.egoID)
            traci.gui.setOffset('View #0', egoPos[0], egoPos[1])

        terminated = not is_ok
        if terminated and cause is not None:
            reward = self.max_punishment
        elif not terminated:
            if self.reward_type == 'complex':
                reward, _ = self.calculate_reward()
                reward += 1
            else:
                reward = reward - (abs(self.state['velocity'] - self.desired_speed)) / self.desired_speed
            self.steps_done += 1
        else:
            reward = -self.max_punishment
        reward = reward
        self.cumulated_reward = self.cumulated_reward + reward
        return self.environment_state, reward, terminated, {'cause': cause, 'rewards': self.cumulated_reward,
                                                            'velocity': self.state['velocity'],
                                                            'distance': new_x - self.ego_start_position,
                                                            'lane_change': self.lanechange_counter}

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

        traci.start(self.sumoCmd[:4])

    def get_surroundings(self, only_env_recheck=False):
        cars_around = traci.vehicle.getContextSubscriptionResults(self.egoID)
        self.calculate_environment()
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
                state[state_key] = 0
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
        state['des_speed'] = self.desired_speed
        return state

    def get_surroundings_env(self, only_env_recheck=False):
        environment, ego_state = self.calculate_environment()
        self.environment_state_list.append(environment)
        if len(self.environment_state_list) > 3:
            self.environment_state_list.pop(0)
        elif len(self.environment_state_list) == 1:
            self.environment_state_list.append(environment)
            self.environment_state_list.append(environment)
        self.environment_state = np.concatenate(self.environment_state_list, -1)
        self.state = ego_state
        return self.environment_state

    def one_step(self):
        terminated = False
        w = traci.simulationStep()

        IDsOfVehicles = traci.vehicle.getIDList()
        if len(IDsOfVehicles) > self.min_departed_vehicles and self.egoID is None:
            for carID in IDsOfVehicles:
                if traci.vehicle.getPosition(carID)[0] < self.ego_start_position and \
                        traci.vehicle.getSpeed(carID) > (60 / 3.6):
                    self.egoID = carID
                    self.ego_start_position = traci.vehicle.getPosition(self.egoID)[0]
            lanes = [-2, -1, 0, 1, 2]
            traci.vehicle.setLaneChangeMode(self.egoID, 0x0)
            traci.vehicle.setSpeedMode(self.egoID, 0x0)
            traci.vehicle.setColor(self.egoID, (255, 0, 0))
            traci.vehicle.setType(self.egoID, 'ego')
            traci.vehicle.setMinGap(self.egoID, 0)
            traci.vehicle.setMinGapLat(self.egoID, 0)

            traci.vehicle.setSpeedFactor(self.egoID, 2)
            traci.vehicle.setSpeed(self.egoID, self.desired_speed)
            traci.vehicle.setMaxSpeed(self.egoID, 50)
            traci.vehicle.subscribeContext(self.egoID, tc.CMD_GET_VEHICLE_VARIABLE, 0.0,
                                           [tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_ANGLE, tc.VAR_POSITION,
                                            tc.VAR_LENGTH, tc.VAR_WIDTH])
            traci.vehicle.addSubscriptionFilterLanes(lanes, noOpposite=True, downstreamDist=100.0, upstreamDist=100.0)
            return True, None
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

    def calculate_reward(self):
        reward = 0
        # LANE BASED REWARD
        lane_reward = 0
        lane_index = self.state['lane']
        if lane_index > 0:
            if (self.state['ER'] == 0) and (self.state['FR']['dx'] > 30):
                lane_reward = -min(1, max(0, (self.state['FR']['dx'] - 50.0) / 20.0))

        # POSITION BASED REWARD
        dy = abs(self.state['y_pos'])
        y_treshold_low = 0.3  # [m]
        y_treshold_high = 0.3  # [m]
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
        vehicle_y = self.state['y_pos']

        # right safe zone
        if self.state['ER'] != 0:
            if vehicle_y > lane_width / 4:
                closing_right = max(-1, (-vehicle_y + lane_width / 4) / (lane_width / 4))
        # left safe zone
        if self.state['EL'] != 0:
            if vehicle_y > - lane_width / 4:
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

    def calculate_environment(self):

        cars_around = traci.vehicle.getContextSubscriptionResults(self.egoID)
        # traci.vehicle.getContextSubscriptionResults(self.egoID)[self.egoID][tc.VAR_POSITION][0]
        ego_state = {}

        environment_collection = []
        for car_id, car in cars_around.items():
            car_state = {'x_position': car[tc.VAR_POSITION][0] - car[tc.VAR_LENGTH] / 2,
                         'y_position': car[tc.VAR_POSITION][1],
                         'length': car[tc.VAR_LENGTH],
                         'width': car[tc.VAR_WIDTH],
                         'velocity': car[tc.VAR_SPEED],
                         'lane_id': car[tc.VAR_LANE_INDEX],
                         'heading': car[tc.VAR_ANGLE]}
            if car_id == self.egoID:
                ego_state = copy.copy(car_state)
            environment_collection.append(copy.copy(car_state))
        grid_per_meter = 1
        x_range = 50  # symmetrically for front and back
        x_range_grid = x_range * grid_per_meter  # symmetrically for front and back
        y_range = 9  # symmetrucally for left and right
        y_range_grid = y_range * grid_per_meter  # symmetrucally for left and right

        state_matrix = np.zeros((2 * x_range_grid, 2 * y_range_grid, 3))
        for element in environment_collection:
            indexes_to_fill = []
            dx = int(np.rint((element['x_position'] - ego_state["x_position"]) * grid_per_meter))
            dy = int(np.rint((ego_state["y_position"] - element['y_position']) * grid_per_meter))
            l = int(np.ceil(element['length'] / 2 * grid_per_meter))
            w = int(np.ceil(element['width'] / 2 * grid_per_meter))
            if (abs(dx) < (x_range_grid - element['length'] / 2 * grid_per_meter)) and \
                    abs(dy) < (y_range_grid - element['width'] / 2 * grid_per_meter):
                state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                y_range_grid + dy - w:y_range_grid + dy + w, 0] += np.ones_like(
                    state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                    y_range_grid + dy - w:y_range_grid + dy + w, 0]) * element['velocity'] / 50
                # state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                # y_range_grid + dy - w:y_range_grid + dy + w, 2] += np.ones_like(
                #     state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                #     y_range_grid + dy - w:y_range_grid + dy + w, 2]) * element['heading'] / 180
                state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                y_range_grid + dy - w:y_range_grid + dy + w, 1] += np.ones_like(
                    state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                    y_range_grid + dy - w:y_range_grid + dy + w, 1]) * element['lane_id'] / 2
                state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                y_range_grid + dy - w:y_range_grid + dy + w, 2] += np.ones_like(
                    state_matrix[x_range_grid + dx - l:x_range_grid + dx + l,
                    y_range_grid + dy - w:y_range_grid + dy + w, 2]) * self.desired_speed / 50


        # lane = traci.vehicle.getLaneID(self.egoID).split('_')[0]
        # for i in range(3):
        #     laneID = f"{lane}_{i}"
        #     lane_pos = traci.lane.getShape(laneID)[0][1]
        #     lane_width = traci.lane.getWidth(laneID)
        #     dy = int((ego_state["y_position"] - lane_pos - ego_state['width']) * grid_per_meter)
        #     w = int(np.ceil(lane_width * grid_per_meter))
        #     state_matrix[:, y_range_grid + dy:y_range_grid + dy + w, 3] = np.ones_like(
        #         state_matrix[:, y_range_grid + dy:y_range_grid + dy + w, 3]) * 0.3 * (i + 0.1)

        # import matplotlib.pyplot as plt
        # plt.gcf()
        # plt.imshow(state_matrix)
        # plt.show()
        return state_matrix, ego_state
