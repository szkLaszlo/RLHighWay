from gym_highway.modell.egovehicle import EgoVehicle
from gym_highway.modell.vehicle_base import BaseVehicle
import numpy as np

log_cnt = 0
logs_in_file = 40
log_list = []


class EnvironmentVehicle(BaseVehicle):

    def __init__(self, dict_env):
        super().__init__(dict_env)
        self.desired_speed = 0.0
        self.state_action = 'in_lane'
        self.change_needed = 0
        self.change_finished = 0  # Indicates the vehicle is leaving its lane
        self.lane_index = 0
        self.lane_width = dict_env['lane_width']
        self.old_lane = 0
        self.behave = np.random.randint(0, 2)
        self.skip = 0
        self.state = {}

    def _get_random(self):
        sigma = 5
        self.desired_speed = 130 / 3.6 + sigma * np.random.randn()
        self.vx = self.desired_speed

    def step(self, state, lanes):
        """
        Steps with vehicle. Updates state

        decision:  1- Reach target speed
                   2- Follow next vehicle

        :param lanes:
        :param state:
        :return: Vehicle reached highway limit (front or rear)
        """
        self.state = state
        front = state['FE']
        front_right = state['FR']
        front_left = state['FL']
        rear = state['RE']
        rear_right = state['RR']
        rear_left = state['RL']
        right = state['ER']
        left = state['EL']

        if self.state_action == 'in_lane':
            acc = 0
            # Desired acceleration
            if self.vx < self.desired_speed:
                acc = self.max_acc
            else:
                acc = self.max_dec

            if front is not None:
                # following car ahead
                dv = front.vx - self.vx
                dx = abs(front.x - self.x) - front.length / 2 - self.length / 2
                if dx <= 8 * self.length:
                # desired following dist
                    dist = front.vx * 1.4
                    d_dist = dist - dx
                    acc_ghr = -1 * d_dist + 10 * dv

                    acc_ghr = min(max(self.max_dec, acc_ghr), self.max_acc)

                    if self.vx > self.desired_speed:
                        acc = min(self.max_dec, acc_ghr)
                    else:
                        acc = acc_ghr

            self.vx = self.vx + self.dt * acc
            self.x = self.x + self.dt * self.vx

            #   Keeping right
            if self.lane_index != 0 and self.behave:
                if ((front_right is None) or (
                        ((abs(front_right.x - self.x) - front_right.length / 2 - self.length / 2 ) / 11) > self.length)) \
                        and right is None:
                    if (rear_right is None) or (
                            ((abs(self.x - rear_right.x) - self.length / 2 - rear_right.length / 2 ) / 9) > self.length):
                        if front_right is not None:
                            if (self.vx * 0.9) < front_right.vx:
                                self.state_action = 'switch_lane_right'
                                self.change_needed = 1
                                self.old_lane = self.lane_index
                        else:
                            self.state_action = 'switch_lane_right'
                            self.change_needed = 1
                            self.old_lane = self.lane_index

            #  Feltartja a mögötte haladót, lehúzódás jobbra
            """
            if not (vbehind is None):
                if (
                        self.x - self.length - vbehind.x) / 5 < self.length:  # Mögötte haladó megközelítette 5 autónyi távolságra
                    if (vbehind.desired_speed > self.desired_speed):
                        if (vright_a is not None) and (vright_b is not None):
                            if ((
                                        vright_a.x - vright_a.length - self.x) / 5) > self.length:  # Előtte 5 autónyi hely van a jobb oldali sávban
                                if ((
                                            self.x - vright_b.x) / 8) > self.length:  # Mögötte 8 autónyi hely van a jobb oldali sávban
                                    print("Under overtake")
                                    #self.state = 'switch_lane_right'
                                    #self.switch_lane = 1
            """
            #  Gyorsabban menne, előzés

            if self.lane_index != (self.env_dict['lane_count'] - 1) and self.behave:
                if front is not None and left is None:
                    diff = (abs(front.x - self.x) - front.length / 2 - self.length / 2)
                    if (diff / 10) < self.length:
                        if self.desired_speed > front.desired_speed:
                            if (front_left is None) or \
                                    (((abs(front_left.x - self.x) - front_left.length / 2 - self.length / 2) / 4) > self.length):
                                if (rear_left is None) or \
                                        (((abs(self.x - rear_left.x) - self.length / 2 - rear_left.length / 2) / 4) > self.length):
                                    if (rear is None) \
                                            or (isinstance(rear, EnvironmentVehicle)
                                                and (rear.state_action != 'acceleration')):
                                        self.state_action = 'acceleration'
                                        s = abs(front.x - self.x) - front.length / 2 - self.length / 2
                                        v_rel = abs(front.vx - self.vx)
                                        t = 3 / self.env_dict['dt']
                                        a = abs((2 * (s - (v_rel * t))) / (t * t))
                                        self.max_acc = a

        elif self.state_action == 'switch_lane_right':
            acc = 0

            if front_right is not None:
                # following GHR model
                dv = front_right.vx - self.vx
                dx = abs(front_right.x - self.x) - front_right.length / 2 - self.length / 2
                # desired following dist
                dist = front_right.vx * 1.4
                d_dist = dist - dx
                acc_ghr = -1 * d_dist + 10 * dv
                acc_ghr = min(max(self.max_dec, acc_ghr), self.max_acc)
                if self.vx > self.desired_speed:
                    acc = min(self.max_dec, acc_ghr)
                else:
                    acc = acc_ghr

            self.vx = self.vx + self.dt * acc
            self.x = self.x + self.dt * self.vx

            self.y = self.y - np.sin(0.08)*self.vx
            if self.y <= ((self.lane_index - 1) * self.lane_width):
                self.y = ((self.lane_index - 1) * self.lane_width)
                self.lane_index = self.lane_index - 1
                self.change_finished = 1
                self.change_needed = 0
                self.state_action = 'in_lane'
                lanes[self.lane_index + 1].remove(self)
                lanes[self.lane_index].append(self)
                if self.lane_index == 0:
                    self.color = 'b'
                elif self.lane_index == 1:
                    self.color = 'k'
                else:
                    self.color = 'y'
                lanes[self.lane_index].sort(key=lambda car: car.x)

        elif self.state_action == 'switch_lane_left':
            acc = 0
            if front is not None:
                # following GHR model
                dv = front.vx - self.vx
                dx = abs(front.x - self.x) - front.length / 2 - self.length / 2
                if dx <= 5*self.length:
                # desired following dist
                    dist = front.vx * 1.4
                    d_dist = dist - dx
                    acc_ghr = -1 * d_dist + 5 * dv
                    acc_ghr = min(max(self.max_dec, acc_ghr), self.max_acc)
                    if self.vx > self.desired_speed:
                        acc = min(self.max_dec, acc_ghr)
                    else:
                        acc = acc_ghr

            self.vx = self.vx + self.dt * acc
            self.x = self.x + self.dt * self.vx

            self.y = self.y + np.sin(0.08)*self.vx
            if self.y >= ((self.lane_index + 1) * self.lane_width):
                self.y = ((self.lane_index + 1) * self.lane_width)
                self.change_finished = 1
                self.change_needed = 0
                self.lane_index = self.lane_index + 1
                lanes[self.lane_index - 1].remove(self)
                lanes[self.lane_index].append(self)
                if self.lane_index == 0:
                    self.color = 'b'
                elif self.lane_index == 1:
                    self.color = 'k'
                else:
                    self.color = 'y'
                lanes[self.lane_index].sort(key=lambda car: car.x)
                self.state_action = 'in_lane'

        elif self.state_action == 'acceleration':
            acc = self.max_acc

            self.vx = self.vx + self.dt * acc
            self.x = self.x + self.dt * self.vx

            if front is not None:
                s = (abs(front.x - self.x) - front.length / 2 - self.length / 2)
                if (s / 8) < self.length:
                    if rear_left is not None:
                        if (self.vx > (0.95 * rear_left.vx)) and \
                                (((abs(self.x - rear_left.x) - self.length / 2 - rear_left.length / 2) / 3) > self.length):
                            if front_left is not None:
                                if (((abs(front_left.x - self.x) - front_left.length / 2 - self.length / 2) / 3) > self.length) \
                                        and (front_left.vx > (self.vx * 0.95)):
                                    self.state_action = 'switch_lane_left'
                                    self.change_needed = 1
                                    self.max_acc = 2
                                    self.old_lane = self.lane_index
                                    # print('Overtake at: ', self.x)
                                else:
                                    self.state_action = 'in_lane'
                                    # self.vx = front.vx
                            else:
                                self.state_action = 'switch_lane_left'
                                self.change_needed = 1
                                self.max_acc = 2
                                self.old_lane = self.lane_index
                                # print('Overtake at: ', self.x)
                        else:
                            self.state_action = 'in_lane'
                            # self.vx = front.vx
                    else:
                        if front_left is not None:
                            if ((abs(front_left.x - self.x) - front_left.length) / 3) > self.length and \
                                    (front_left.vx > (self.vx * 0.95)):
                                self.state_action = 'switch_lane_left'
                                self.change_needed = 1
                                self.max_acc = 2
                                self.old_lane = self.lane_index
                                # print('Overtake at: ', self.x)
                            else:
                                self.state_action = 'in_lane'
                                # self.vx = front.vx
                        else:
                            self.state_action = 'switch_lane_left'
                            self.change_needed = 1
                            self.max_acc = 2
                            self.old_lane = self.lane_index
                            # print('Overtake at: ', self.x)
            # else:
            # self.state_action = 'in_lane'

        reachedend = False

        if (self.x > self.env_dict['length_forward']) or (self.x < self.env_dict['length_backward']):
            reachedend = True

        return reachedend

    def warmup_step(self, vnext):
        """
        Steps with vehicle. Updates state

        decision:  1- Reach target speed
                   2- Follow next vehicle

        :param vnext: vehicle in front
        :param vright: vehicle to the right
        :param vleft: vehicle to the left
        :return: Vehicle reached highway limit (front or rear)
        """
        acc = 0
        # Desired acceleration
        if self.vx < self.desired_speed:
            acc = self.max_acc
        else:
            acc = -self.max_acc

        if not (vnext is None):
            # following GHR model
            dv = vnext.vx - self.vx
            dx = vnext.x - vnext.length - self.x
            if dx < 0:
                env_save_log()
                raise CollisionExc('Collision')
                print('Collision')
            # desired following dist
            # dist = vnext.vx * 1.4
            dist = vnext.vx * 1.2
            ddist = dist - dx
            accghr = -1 * ddist + 5 * dv

            # alpha=0.6
            # m=0.4
            # l=1.9
            # accghr=alpha*(self.vx**m)*dv/(dx**l)

            accghr = min(max(self.max_dec, accghr), self.max_acc)
            if self.vx > self.desired_speed:
                acc = min(-self.max_acc, accghr)
            else:
                acc = accghr

        self.vx = self.vx + self.dt * acc
        self.x = self.x + self.dt * self.vx


class CollisionExc(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def env_add_entry(text):
    global log_cnt

    log_cnt += 1
    write = 'Step ' + str(log_cnt) + '\n'
    write += text
    log_list.append(write)

    if log_cnt > logs_in_file:
        log_list.pop(0)


def env_save_log():
    if log_cnt > logs_in_file:
        log_file = open(r'C:\log_file.txt', 'w+')
        for i in range(logs_in_file):
            entry = log_list[i]
            log_file.write(entry)
        log_file.close()
