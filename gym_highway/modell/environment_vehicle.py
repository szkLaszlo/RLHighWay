from gym_highway.modell.vehicle_base import BaseVehicle
import numpy as np

log_cnt = 0
logs_in_file = 40
log_list = []


class EnvironmentVehicle(BaseVehicle):

    def __init__(self, dict_env):
        super().__init__(dict_env)
        self.desired_speed = 0.0
        self.state = 'in_lane'
        self.change_needed = 0
        self.change_finished = 0  # Indicates the vehicle is leaving its lane
        self.lane_index = 0
        self.lane_width = dict_env['lane_width']
        self.old_lane = 0
        self.skip = 0

    def _get_random(self):
        sigma = 5
        self.desired_speed = 130 / 3.6 + sigma * np.random.randn()
        self.vx = self.desired_speed

    def step(self, next_vehicle, behind=None, right_a=None, right_b=None, left_a=None, left_b=None):
        """
        Steps with vehicle. Updates state

        decision:  1- Reach target speed
                   2- Follow next vehicle

        :param left_b:
        :param left_a:
        :param right_b:
        :param right_a:
        :param behind:
        :param next_vehicle: vehicle in front
        :return: Vehicle reached highway limit (front or rear)
        """

        if self.state == 'in_lane':
            acc = 0
            # Desired acceleration
            if self.vx < self.desired_speed:
                acc = self.max_acc
            else:
                acc = -self.max_acc

            if next_vehicle is not None:
                # following GHR model
                dv = next_vehicle.vx - self.vx
                dx = next_vehicle.x - next_vehicle.length/2 - self.length/2 - self.x
                if dx < 0:
                    print('Collision, ID: ', self.ID, ' v_next ID: ', next_vehicle.ID, ' in lane: ', self.lane_index)
                    print(next_vehicle.x, ' - ', next_vehicle.length, ' - ', self.x)
                    env_save_log()
                    raise CollisionExc('Collision')

                # desired following dist
                # dist = vnext.vx * 1.4
                dist = next_vehicle.vx * 1.2
                d_dist = dist - dx
                acc_ghr = -1 * d_dist + 10 * dv

                # alpha=0.6
                # m=0.4
                # l=1.9
                # acc_ghr=alpha*(self.vx**m)*dv/(dx**l)

                acc_ghr = min(max(self.max_dec, acc_ghr), self.max_acc)
                if self.vx > self.desired_speed:
                    acc = min(-self.max_acc, acc_ghr)
                else:
                    acc = acc_ghr

            self.vx = self.vx + self.dt * acc
            self.x = self.x + self.dt * self.vx

            #   Keeping right
            if self.lane_index != 0:
                if (right_a is None) or (((right_a.x - right_a.length - self.x) / 11) > self.length):
                    if (right_b is None) or (((self.x - self.length - right_b.x) / 9) > self.length):
                        if not (right_a is None):
                            if (self.vx * 0.7) < right_a.vx:
                                self.state = 'switch_lane_right'
                                self.change_needed = 1
                                self.old_lane = self.lane_index
                        else:
                            self.state = 'switch_lane_right'
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

            if self.lane_index != (self.env_dict['lane_count'] - 1):
                if not (next_vehicle is None):
                    diff = (next_vehicle.x - next_vehicle.length - self.x)
                    if ((diff / 9) < self.length):
                        if self.desired_speed > next_vehicle.desired_speed:
                            if (left_a is None) or (((left_a.x - left_a.length - self.x) / 4) > self.length):
                                if (left_b is None) or (((self.x - self.length - left_b.x) / 4) > self.length):
                                    if (behind is None) or (
                                            isinstance(behind, EnvironmentVehicle) and (behind.state != 'acceleration')):
                                        self.state = 'acceleration'
                                        s = next_vehicle.x - next_vehicle.length - self.x
                                        vrel = abs(next_vehicle.vx - self.vx)
                                        t = 3 / self.env_dict['dt']
                                        a = abs((2 * (s - (vrel * t))) / (t * t))
                                        self.max_acc = a

        elif self.state == 'switch_lane_right':
            acc = self.max_acc

            if not (next_vehicle is None):
                # following GHR model
                dv = next_vehicle.vx - self.vx
                dx = next_vehicle.x - next_vehicle.length - self.x
                if dx < 0:
                    print('Collision, ID: ', self.ID, ' vnext ID: ', next_vehicle.ID, ' in lane: ', self.lane_index)
                    env_save_log()
                    raise CollisionExc('Collision')
                    print('Collision')
                # desired following dist
                # dist = vnext.vx * 1.4
                dist = next_vehicle.vx * 1.2
                d_dist = dist - dx
                acc_ghr = -1 * d_dist + 10 * dv

                acc_ghr = min(max(self.max_dec, acc_ghr), self.max_acc)
                if self.vx > self.desired_speed:
                    acc = min(-self.max_acc, acc_ghr)
                else:
                    acc = acc_ghr

            self.vx = self.vx + self.dt * acc
            self.x = self.x + self.dt * self.vx

            self.y = self.y - 0.4
            if self.y <= ((self.lane_index - 1) * self.lane_width):
                self.y = ((self.lane_index - 1) * self.lane_width)
                self.lane_index = self.lane_index - 1
                self.change_finished = 1
                self.state = 'in_lane'

        elif self.state == 'switch_lane_left':
            acc = max(self.max_acc, 2)
            if not (next_vehicle is None):
                # following GHR model
                dv = next_vehicle.vx - self.vx
                dx = next_vehicle.x - next_vehicle.length - self.x
                if dx < 0:
                    print('Collision, ID: ', self.ID, ' vnext ID: ', next_vehicle.ID, ' in lane: ', self.lane_index)
                    env_save_log()
                    raise CollisionExc('Collision')
                    print('Collision')

                # desired following dist
                # dist = vnext.vx * 1.4
                dist = next_vehicle.vx * 1.2
                d_dist = dist - dx
                acc_ghr = -1 * d_dist + 10 * dv

                acc_ghr = min(max(self.max_dec, acc_ghr), self.max_acc)
                if self.vx > self.desired_speed:
                    acc = min(-self.max_acc, acc_ghr)
                else:
                    acc = acc_ghr

            self.vx = self.vx + self.dt * acc
            self.x = self.x + self.dt * self.vx

            self.y = self.y + 0.4
            if self.y >= ((self.lane_index + 1) * self.lane_width):
                self.y = ((self.lane_index + 1) * self.lane_width)
                self.lane_index = self.lane_index + 1
                self.state = 'in_lane'

        elif self.state == 'acceleration':
            acc = self.max_acc

            self.vx = self.vx + self.dt * acc
            self.x = self.x + self.dt * self.vx

            if not (next_vehicle is None):
                s = (next_vehicle.x - next_vehicle.length - self.x)
                if (s / 3) < self.length:
                    if not (left_b is None):
                        if (self.vx > (0.8 * left_b.vx)) and (((self.x - self.length - left_b.x) / 3) > self.length):
                            if not (left_a is None):
                                if (((left_a.x - left_a.length - self.x) / 3) > self.length) and \
                                        (left_a.vx > (self.vx * 0.8)):
                                    self.state = 'switch_lane_left'
                                    self.change_needed = 1
                                    self.max_acc = 2
                                    self.old_lane = self.lane_index
                                    # print('Overtake at: ', self.x)
                                else:
                                    self.state = 'in_lane'
                                    self.vx = next_vehicle.vx
                            else:
                                self.state = 'switch_lane_left'
                                self.change_needed = 1
                                self.max_acc = 2
                                self.old_lane = self.lane_index
                                # print('Overtake at: ', self.x)
                        else:
                            self.state = 'in_lane'
                            self.vx = next_vehicle.vx
                    else:
                        if not (left_a is None):
                            if (((left_a.x - left_a.length - self.x) / 3) > self.length):
                                self.state = 'switch_lane_left'
                                self.change_needed = 1
                                self.max_acc = 2
                                self.old_lane = self.lane_index
                                # print('Overtake at: ', self.x)
                            else:
                                self.state = 'in_lane'
                                self.vx = next_vehicle.vx
                        else:
                            self.state = 'switch_lane_left'
                            self.change_needed = 1
                            self.max_acc = 2
                            self.old_lane = self.lane_index
                            # print('Overtake at: ', self.x)
            else:
                self.state = 'in_lane'
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
            accghr = -1 * ddist + 10 * dv

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
