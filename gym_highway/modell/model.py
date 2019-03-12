import numpy as np
import matplotlib.pyplot as plt
import math
from gym_highway.modell.egovehicle import EgoVehicle
from gym_highway.modell.environment_vehicle import EnvironmentVehicle


class Model:
    def __init__(self, env_dict=None):
        if env_dict is None:
            self.env_dict = {'length_forward': 1000, 'length_backward': 500, 'dt': 0.2, 'lane_width': 4,
                             'lane_count': 3,
                             'density_lane0': 12, 'density_lane1': 8, 'speed_mean_lane0': 110.0 / 3.6,
                             'speed_std_lane0': 10.0 / 3.6, 'speed_mean_lane1': 150.0 / 3.6,
                             'speed_std_lane1': 10.0 / 3.6,
                             'speed_ego_desired': 130.0 / 3.6, 'car_length': 3, 'safe_zone_length': 2,
                             'max_acceleration': 2,
                             'max_deceleration': -6}
            # Vehicle Generation Parameters

        else:
            self.env_dict = env_dict
        self.ego_vehicle = None

        self.highway_length = self.env_dict['length_forward'] + self.env_dict['length_backward']
        self.lanes = []
        self.next_vehicle = []
        for i in range(self.env_dict['lane_count']):
            self.lanes.append([])

    def one_step(self, action):
        """
        Function steps the ego vehicle based on @param: action.
        :param action: action to make
        :return: success, cause
        """
        # 1. Stepping the ego vehicle
        self.ego_vehicle.step(action)

        if self.ego_vehicle.vx < 20:
            return False, 'Low speed'
        # perform lane change and collision check
        fine, cause = self.check_position()
        if not fine:
            # self.lanes[self.ego_vehicle.lane_index].remove(self.ego_vehicle)
            # self.search_ego_vehicle()  # TODO: ez minek ide?
            return False, cause

        # 2. Transpose everyone to set x of egovehicle to 0
        offs = -self.ego_vehicle.x
        for j in range(self.env_dict['lane_count']):
            lane = self.lanes[j]
            for i in range(len(lane)):
                lane[i].x = lane[i].x + offs

        # 3. Stepping every other vehicle
        for i in range(self.env_dict['lane_count']):
            lane = self.lanes[i]
            for j in reversed(range(len(lane))):
                veh = lane[j]
                if isinstance(veh, EnvironmentVehicle):
                    state = self.generate_state_for_ego(current_ego=veh)
                    veh.step(state=state, lanes=self.lanes)

        # 4. Deleting vehicles out of range
        for j in range(self.env_dict['lane_count']):
            lane = self.lanes[j]
            for veh in lane:
                if veh.x < -self.env_dict['length_backward'] or veh.x > self.env_dict['length_forward']:
                    lane.remove(veh)

        # 5. NewBorn vehicles If lane density is smaller than desired
        self.generate_new_vehicles(200)

        self.random_new_des_speed()

        return True, 'Fine'

    def check_position(self):
        lane_index = int(round(self.ego_vehicle.y / self.env_dict['lane_width']))
        if (lane_index < 0) or (lane_index + 1 > self.env_dict['lane_count']):
            return False, 'Left Highway'

        if lane_index != self.ego_vehicle.lane_index:
            # Lane change in process
            old_lane = self.lanes[self.ego_vehicle.lane_index]
            new_lane = self.lanes[lane_index]
            old_lane.remove(self.ego_vehicle)
            new_lane.append(self.ego_vehicle)
            new_lane.sort(key=lambda car: car.x)
            self.ego_vehicle.lane_index = lane_index

        lane = self.lanes[self.ego_vehicle.lane_index]
        i = lane.index(self.ego_vehicle)
        front = i + 1
        if len(lane) > front:
            if self.ego_vehicle.x > (lane[front].x - lane[front].length / 2 - self.ego_vehicle.length / 2):
                return False, 'Front Collision'

        rear = i - 1
        if i:
            if lane[rear].x > self.ego_vehicle.x - self.ego_vehicle.length / 2 - lane[rear].length / 2:
                return False, 'Rear Collision'

        return True, 'Everything is fine'

    def generate_new_vehicles(self, epsilon):
        for i in range(self.env_dict['lane_count']):
            lane = self.lanes[i]
            density = 1000 * len(lane) / (self.env_dict['length_backward'] + self.env_dict['length_forward'])

            while density < self.env_dict['density_lane' + str(i)]:
                if len(lane) < 2:
                    new_x = np.random.randint(-self.env_dict['length_backward'] + 10,
                                              self.env_dict['length_forward'] - 10)
                    dist = 41
                    j = -1
                    v_front = 10000
                else:
                    j = len(lane)
                    while j > len(lane) - 1:
                        j = np.random.randint(0, len(lane))
                    veh = lane[j]
                    if j == len(lane) - 1:
                        dist = self.env_dict['length_forward'] - veh.x
                        if dist < 40:
                            continue
                        new_x = np.random.randint(veh.x + 20, self.env_dict['length_forward'] - 10)
                        v_front = 1000
                    elif j == 0:
                        dist = veh.x + self.env_dict['length_backward']
                        if abs(dist) < 40:
                            continue
                        new_x = np.random.randint(-self.env_dict['length_backward'] + 10, veh.x - 20)
                        v_front = 1000
                    else:
                        veh_next = lane[j + 1]
                        v_front = veh_next.vx

                        dist = veh_next.x - veh.x
                        if abs(dist) < 40:
                            continue
                        new_x = np.random.randint(veh.x + 15, veh_next.x - 15)

                if abs(dist) < 40 or abs(new_x) < epsilon:
                    continue
                else:

                    ev = EnvironmentVehicle(self.env_dict)
                    ev.desired_speed = \
                        self.env_dict['speed_mean_lane' + str(i)] + np.random.randn() * \
                        self.env_dict['speed_std_lane' + str(i)]

                    ev.vx = min(ev.desired_speed, v_front)
                    ev.x = new_x
                    ev.y = i * self.env_dict['lane_width']
                    ev.lane_index = i
                    if i == 0:
                        ev.color = 'b'
                    elif i == 1:
                        ev.color = 'k'
                    else:
                        ev.color = 'y'
                    lane.insert(j + 1, ev)
                    lane.sort(key=lambda car: car.x)
                density = 1000 * len(lane) / (self.env_dict['length_backward'] + self.env_dict['length_forward'])

    def random_new_des_speed(self):
        factor = 0.05
        for i in range(self.env_dict['lane_count']):
            lane = self.lanes[i]
            for ev in lane:
                if (ev is EnvironmentVehicle) and (np.random.rand() < factor):
                    ev.desired_speed = self.env_dict['speed_mean_lane' + str(i)] + np.random.randn() * self.env_dict[
                        'speed_std_lane' + str(i)]

    def generate_state_for_ego(self, current_ego=None):
        """
        Calculating the environment for the current_ego vehicle


        :return: np.array()
            idx   |  Meaning          |  Elements   |  Default
            ------+-------------------+-------------+----------
            0,1   | Front Left  Lane  |  dx,dv      |  500,0
            2,3   | Front Ego   Lane  |  dx,dv      |  500,0
            4,5   | Front Right Lane  |  dx,dv      |  500,0
            6,7   | Rear  Left  Lane  |  dx,dv      |  500,0
            8,9   | Rear  Ego   Lane  |  dx,dv      |  500,0
            10,11 | Rear  Right Lane  |  dx,dv      |  500,0
            12    | Left  Safe Zone   |  Occup [0,1]|  -
            13    | Right Safe Zone   |  Occup [0,1]|  -
            14    | Vehicle y pos     |  pos [m]    |  -
            15    | Vehicle heading   |  th[rad]    |  -
            16    | Vehicle speed     |  v[m/s]     |  -
        """

        basic_keys = ['FL', 'FE', 'FR', 'RL', 'RE', 'RR', 'EL', 'ER']
        state = {}

        if current_ego is None:
            current_ego = self.ego_vehicle
            state['pos_y'] = 0
            state['heading'] = 0
            state['speed'] = 0
            basic_vals = {'dx': 500, 'dv': 0}
        else:
            basic_vals = None
        lane_count = self.env_dict['lane_count']
        for keys in basic_keys:
            state[keys] = basic_vals
        # Left
        if not (current_ego.lane_index == lane_count - 1):
            state['FL'], state['RL'], state['EL'] = self.search_lane_for_state(
                self.lanes[current_ego.lane_index + 1], current_ego=current_ego)
        # right
        if current_ego.lane_index != 0:
            state['FR'], state['RR'], state['ER'] = self.search_lane_for_state(
                self.lanes[current_ego.lane_index - 1], current_ego=current_ego)
        # Ego lane
        state['FE'], state['RE'], _ = \
            self.search_lane_for_state(self.lanes[current_ego.lane_index], current_ego=current_ego)

        state['pos_y'] = round(current_ego.y, 3)
        state['heading'] = round(math.atan2(current_ego.vy, current_ego.vx), 6)
        state['speed'] = round(math.sqrt(current_ego.vx ** 2 + current_ego.vy ** 2), 3)
        return state

    def search_lane_for_state(self, lane, current_ego=None):
        if current_ego is None:
            current_ego = self.ego_vehicle
        safe_zone_length = self.env_dict['safe_zone_length']
        rear = None
        side = None
        front = None
        ret_state = np.array([500, 0, 500, 0, 500, 0])
        ego_rear = current_ego.x - current_ego.length / 2 - safe_zone_length
        ego_front = current_ego.x + current_ego.length / 2 + safe_zone_length
        lane.sort(key=lambda car: car.x)
        # TODO: if lane is  sorted by x, and then this is good.
        for i in range(len(lane)):
            if (lane[i].x + lane[i].length / 2) < ego_rear:
                rear = lane[i]
            elif (lane[i].x - lane[i].length / 2) < ego_front:
                side = lane[i]
            else:
                front = lane[i]
                break  # just if lane is ordered otherwise this causes problems.
        if front is not None:
            ret_state[0] = front.x - front.length / 2 - current_ego.length / 2 - current_ego.x
            ret_state[1] = front.vx - current_ego.vx
        if rear is not None:
            ret_state[2] = current_ego.x - current_ego.length / 2 - rear.length / 2 - rear.x
            ret_state[3] = rear.vx - current_ego.vx
        if side is not None:
            ret_state[4] = current_ego.x - side.x
            ret_state[5] = side.vx - current_ego.vx

        if current_ego == self.ego_vehicle:
            return {'dx': ret_state[0], 'dv': ret_state[1]}, {'dx': ret_state[2], 'dv': ret_state[3]}, \
                   {'dx': ret_state[4], 'dv': ret_state[5]}
        else:
            return front, rear, side

    def search_ego_vehicle(self, preferred_lane_id=-1):
        # TODO: nem értem ez mit szeretne csinálni.
        if preferred_lane_id == -1:
            lane_ind = np.random.randint(0, self.env_dict['lane_count'])
        else:
            lane_ind = preferred_lane_id

        lane = self.lanes[lane_ind]
        try:
            ind = self.lanes[lane_ind].index(self.ego_vehicle)
        except ValueError:
            ind = -1

        if ind == -1:
            e = EgoVehicle(self.env_dict)
            e.x = 0
            e.y = 0
            e.vy = 0
            e.vx = e.desired_speed
            e.lane_index = lane_ind
            e.desired_speed = self.env_dict['speed_ego_desired']
            self.ego_vehicle = e
            self.lanes[lane_ind].append(e)
            self.lanes[lane_ind].sort(key=lambda car: car.x)
            return

        self.ego_vehicle = self.lanes[lane_ind][ind]

        ind = self.lanes[lane_ind].index(self.ego_vehicle)
        old = self.ego_vehicle

        e = EgoVehicle(self.env_dict)
        e.x = old.x
        e.y = old.y
        e.vy = 0
        e.vx = old.vx
        e.lane_index = lane_ind
        self.ego_vehicle = e
        self.lanes[lane_ind][ind] = e

    def warm_up(self, render=True):
        warm_up_time = 30  # [secs]

        for i in range(self.env_dict['lane_count']):
            self.next_vehicle.append(self.calc_next_vehicle_following(i))

        for _ in range(int(warm_up_time / self.env_dict['dt'])):
            for i in range(self.env_dict['lane_count']):
                lane = self.lanes[i]
                vehicle_cnt = len(lane)
                if vehicle_cnt == 0:
                    first_length = 100000.0
                    v_next = 100000.0
                else:
                    first_length = lane[0].x - lane[0].length + self.env_dict['length_backward']
                    v_next = lane[0].vx

                if first_length > self.next_vehicle[i]:
                    ev = EnvironmentVehicle(self.env_dict)
                    ev.desired_speed = self.env_dict['speed_mean_lane' + str(i)] + np.random.randn() * self.env_dict[
                        'speed_std_lane' + str(i)]
                    ev.vx = min(ev.desired_speed, v_next)

                    ev.x = -self.env_dict['length_backward']
                    ev.y = i * self.env_dict['lane_width']
                    ev.lane_index = i
                    if i == 0:
                        ev.color = 'b'
                    elif i == 1:
                        ev.color = 'k'
                    else:
                        ev.color = 'y'
                    lane.insert(0, ev)
                    self.next_vehicle[i] = self.calc_next_vehicle_following(i)

                vehicle_cnt = len(lane)
                for j in reversed(range(vehicle_cnt)):
                    veh = lane[j]
                    if isinstance(veh, EnvironmentVehicle):
                        state = self.generate_state_for_ego(current_ego=veh)
                        veh.step(state=state, lanes=self.lanes)
            if render:
                self.render()

    def render(self, close=False, rewards=None, zoom=1):
        a = plt.axes()
        a.clear()

        for i in range(self.env_dict['lane_count']):
            for j in range(len(self.lanes[i])):
                self.lanes[i][j].render(a, zoom=zoom)

        lf = self.env_dict['length_forward']
        lb = -self.env_dict['length_backward']
        lw = self.env_dict['lane_width'] * zoom
        lc = self.env_dict['lane_count']

        lines = a.plot([lb, lf], [(lc - .5) * lw, (lc - .5) * lw], 'k')
        plt.setp(lines, linewidth=.5)
        lines = plt.plot([lb, lf], [-lw / 2, -lw / 2], 'k')
        plt.setp(lines, linewidth=.5)
        for i in range(lc - 1):
            lines = plt.plot([lb, lf], [(i + .5) * lw, (i + .5) * lw], 'k--')
            plt.setp(lines, linewidth=.5)
        plt.axis('equal')
        if close:
            plt.xlim([-200, 200])
        else:
            plt.xlim([-self.env_dict['length_backward'] - 100, self.env_dict['length_forward'] + 100])

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        th = math.atan2(self.ego_vehicle.vy, self.ego_vehicle.vx) * 180 / math.pi
        v = math.sqrt(self.ego_vehicle.vx ** 2 + self.ego_vehicle.vy ** 2) * 3.6

        tstr = 'Speed: %4.2f [km/h]\nTheta:  %4.2f [deg]\nPos:  %4.2f [m]\nLane:  %d [-]' % \
               (v, th, self.ego_vehicle.y, self.ego_vehicle.lane_index)

        plt.text(0.05, 0.95, tstr, transform=plt.axes().transAxes, verticalalignment='top', bbox=props, fontsize=14,
                 family='monospace')

        if not (rewards is None):
            tstr = 'Lane reward: %5.3f\n   y reward:  %5.3f\n   v reward:  %5.3f\n   c reward:  %5.3f\n ' % (
                rewards[0], rewards[1], rewards[2], rewards[3])
            plt.text(0.05, 0.35, tstr, transform=plt.axes().transAxes, verticalalignment='top', bbox=props, fontsize=14,
                     family='monospace')

        # plt.show()
        plt.pause(0.003)

    def calc_next_vehicle_following(self, lane):
        mean = 1000 / self.env_dict['density_lane' + str(lane)]
        return max(10, mean + np.random.randn() * 20)
