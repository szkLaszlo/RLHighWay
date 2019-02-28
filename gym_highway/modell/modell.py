import numpy as np
import matplotlib.pyplot as plt
import math
from gym_highway.modell.ego_vehicle import Egovehicle
from gym_highway.modell.environment_vehicle import Envvehicle


class Modell:
    def __init__(self, envdict=None):
        if envdict is None:
            self.envdict = {'length_forward': 1000, 'length_backward': 500, 'dt': 0.2, 'lane_width': 4, 'lane_count': 2,
                            'density_lane0': 16, 'density_lane1': 8, 'speed_mean_lane0': 110.0 / 3.6,
                            'speed_std_lane0': 10.0 / 3.6, 'speed_mean_lane1': 180.0 / 3.6,
                            'speed_std_lane1': 10.0 / 3.6, 'speed_ego_desired': 130.0 / 3.6}

            # Vehicle Generation Parameters

        else:
            self.envdict = envdict
        self.egovehicle = None

        self.highwaylength = self.envdict['length_forward'] + self.envdict['length_backward']
        self.lanes = []
        self.nextvehicle = []
        for i in range(self.envdict['lane_count']):
            self.lanes.append([])

    def onestep(self, action):
        """

        :param action: takes action for egovehicle
        :return: success, cause
        """
        # 1. Stepping the ego vehicle
        self.egovehicle.step(action)
        if self.egovehicle.vx < 20:
            return False, 'Low speed'
        # perform lane change and collision check
        fine, cause = self.check_position()
        if not fine:
            #print(cause)
            self.lanes[self.egovehicle.laneindex].remove(self.egovehicle)
            self.searchEgoVehicle()
            return False, cause

        # 2. Transpose everyone to set x of egovehicle to 0
        offs = -self.egovehicle.x
        for j in range(self.envdict['lane_count']):
            lane = self.lanes[j]
            for i in range(len(lane)):
                lane[i].x = lane[i].x + offs
        # 3. Stepping every other vehicle
        for i in range(self.envdict['lane_count']):
            lane = self.lanes[i]
            vehiclecnt = len(lane)
            for j in range(vehiclecnt):
                veh = lane[j]
                if isinstance(veh, Envvehicle):
                    if j + 1 < vehiclecnt:
                        vnext = lane[j + 1]
                    else:
                        vnext = None
                    veh.step(vnext, None, None)
        # 4. Deleting vehicles out of range
        for j in range(self.envdict['lane_count']):
            lane = self.lanes[j]
            for veh in lane:
                if veh.x < -self.envdict['length_backward'] or veh.x > self.envdict['length_forward']:
                    lane.remove(veh)
        # 5. NewBorn vehicles If lane density is smaller than desired
        self.generate_new_vehicles()

        self.random_new_des_speed()

        return True, 'Fine'

    def check_position(self):
        laneindex = int(round(self.egovehicle.y / self.envdict['lane_width']))
        if (laneindex < 0) or (laneindex + 1 > self.envdict['lane_count']):
            return False, 'Left Highway'

        if laneindex != self.egovehicle.laneindex:
            # sávváltás van
            oldlane = self.lanes[self.egovehicle.laneindex]
            newlane = self.lanes[laneindex]
            oldlane.remove(self.egovehicle)
            i = 0
            for i in range(len(newlane)):
                if newlane[i].x > self.egovehicle.x:
                    break
            newlane.insert(i, self.egovehicle)
            self.egovehicle.laneindex = laneindex

        lane = self.lanes[self.egovehicle.laneindex]
        i = lane.index(self.egovehicle)
        front = i + 1
        if len(lane) > front:
            if self.egovehicle.x > lane[front].x - lane[front].length:
                return False, 'Front Collision'

        rear = i - 1
        if 1 <= i:
            if lane[rear].x > self.egovehicle.x - self.egovehicle.length:
                #print(i, lane[rear].x, self.egovehicle.x, self.egovehicle.length)
                return False, 'Rear Collision'

        return True, 'Everything is fine'

    def generate_new_vehicles(self):
        for i in range(self.envdict['lane_count']):
            lane = self.lanes[i]
            density = 1000 * len(lane) / (self.envdict['length_backward'] + self.envdict['length_forward'])
            if density == 0:
                desdist = 0
            else:
                desdist = 1000.0 / self.envdict['density_lane' + str(i)]
            if density < self.envdict['density_lane' + str(i)]:
                # Need to create
                # random front-back
                r = np.random.rand() < 0.5
                if r == 0:
                    # back generation
                    vehiclecnt = len(lane)
                    if vehiclecnt == 0:
                        firstlength = 100000.0
                        vnext = 100000.0
                    else:
                        firstlength = lane[0].x - lane[0].length + self.envdict['length_backward']
                        vnext = lane[0].vx
                    if firstlength > desdist:
                        ev = Envvehicle(self.envdict)
                        ev.desired_speed = self.envdict['speed_mean_lane' + str(i)] + np.random.randn() * \
                                                                                      self.envdict[
                                                                                          'speed_std_lane' + str(i)]
                        ev.vx = min(ev.desired_speed, vnext)

                        ev.x = -self.envdict['length_backward'] + 10
                        ev.y = i * self.envdict['lane_width']
                        if i == 0:
                            ev.color = 'b'
                        else:
                            ev.color = 'k'
                        lane.insert(0, ev)
                    else:
                        vehiclecnt = len(lane)
                        if vehiclecnt == 0:
                            firstlength = 100000.0
                            vnext = 100000.0
                        else:
                            last = lane[-1]
                            firstlength = abs(self.envdict['length_forward'] - last.x)
                            vnext = last.vx
                        if firstlength > desdist:
                            ev = Envvehicle(self.envdict)
                            ev.desired_speed = self.envdict['speed_mean_lane' + str(i)] + np.random.randn() * \
                                                                                          self.envdict[
                                                                                              'speed_std_lane' + str(i)]
                            ev.vx = min(ev.desired_speed, vnext)

                            ev.x = self.envdict['length_forward'] - 10
                            ev.y = i * self.envdict['lane_width']
                            if i == 0:
                                ev.color = 'b'
                            else:
                                ev.color = 'k'
                            lane.insert(len(lane), ev)

    def random_new_des_speed(self):
        factor = 0.05
        for i in range(self.envdict['lane_count']):
            lane = self.lanes[i]
            for ev in lane:
                if (ev is Envvehicle) and (np.random.rand() < factor):
                    ev.desired_speed = self.envdict['speed_mean_lane' + str(i)] + np.random.randn() * self.envdict[
                        'speed_std_lane' + str(i)]

    def generate_state_for_ego(self):
        """
        Calculating the environment for the ego vehicle


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
        state = np.zeros(17)
        state[0] = 500
        state[2] = 500
        state[4] = 500
        state[6] = 500
        state[8] = 500
        state[10] = 500

        lc = self.envdict['lane_count']
        # Left
        if not (self.egovehicle.laneindex == lc - 1):
            state[0], state[1], state[6], state[7], state[12] = self.searchlane_forstate(
                self.lanes[self.egovehicle.laneindex + 1], self.egovehicle.x)
        # right
        if self.egovehicle.laneindex != 0:
            state[4], state[5], state[10], state[11], state[13] = self.searchlane_forstate(
                self.lanes[self.egovehicle.laneindex - 1], self.egovehicle.x)
        # Ego lane
        state[2], state[3], state[8], state[9], _ = self.searchlane_forstate(self.lanes[self.egovehicle.laneindex],
                                                                             self.egovehicle.x)
        state[14] = round(self.egovehicle.y, 3)
        state[15] = round(math.atan2(self.egovehicle.vy, self.egovehicle.vx), 6)
        state[16] = round(math.sqrt(self.egovehicle.vx ** 2 + self.egovehicle.vy ** 2), 3)
        return state

    def searchlane_forstate(self, lane, x):
        szlen = 2  # safe zone length
        rear = None
        safe = None
        front = None
        res = np.array([0, 500, 0, 500, 0])
        egorear = self.egovehicle.x - self.egovehicle.length - szlen
        egofront = self.egovehicle.x + szlen
        egox = self.egovehicle.x
        egov = self.egovehicle.vx
        for i in range(len(lane)):
            if lane[i].x < egorear:
                rear = lane[i]
            elif lane[i].x - lane[i].length < egofront:
                safe = lane[i]
            else:
                front = lane[i]
                break
        if not (front is None):
            res[0] = front.x - front.length - egox
            res[1] = front.vx - egov
        if not (rear is None):
            res[2] = egox - self.egovehicle.length - rear.x
            res[3] = egov - rear.vx
        if not (safe is None):
            res[4] = 1
        return res

    def searchEgoVehicle(self, preferredlaneid=-1):
        if preferredlaneid==-1:
            laneind= np.random.randint(0,self.envdict['lane_count'])
        else:
            laneind=preferredlaneid

        lane = self.lanes[laneind]
        ind = -1
        dist = 100000.0
        for i in range(len(lane)):
            if abs(lane[i].x) < dist:
                ind = i
                dist = abs(lane[i].x)
            else:
                break

        if ind == -1:
            e = Egovehicle(self.envdict)
            self.egovehicle = e
            self.lanes[laneind].append(e)
            e.x = 0
            e.y = 0
            e.vy = 0
            e.vx = e.desired_speed
            e.laneindex = laneind
            e.desired_speed=self.envdict['speed_ego_desired']
            return

        offs = -lane[ind].x
        for j in range(self.envdict['lane_count']):
            lane = self.lanes[j]
            for i in range(len(lane)):
                lane[i].x = lane[i].x + offs
        self.egovehicle = self.lanes[laneind][ind]
        for j in range(self.envdict['lane_count']):
            lane = self.lanes[j]
            for a in lane:
                if a.x <= -self.envdict['length_backward'] or a.x > self.envdict['length_forward']:
                    lane.remove(a)
        ind = self.lanes[laneind].index(self.egovehicle)
        old = self.egovehicle

        e = Egovehicle(self.envdict)
        self.egovehicle = e
        self.lanes[laneind][ind] = e
        e.x = old.x
        e.y = old.y
        e.vy = 0
        e.vx = old.vx
        e.laneindex = laneind

    def warmup(self, render=True):
        warmuptime = 30  # [secs]

        for i in range(self.envdict['lane_count']):
            self.nextvehicle.append(self.calcnextvehiclefollowing(i))

        for _ in range(int(warmuptime / self.envdict['dt'])):
            for i in range(self.envdict['lane_count']):
                lane = self.lanes[i]
                vehiclecnt = len(lane)
                if vehiclecnt == 0:
                    firstlength = 100000.0
                    vnext = 100000.0
                else:
                    firstlength = lane[0].x - lane[0].length + self.envdict['length_backward']
                    vnext = lane[0].vx

                if firstlength > self.nextvehicle[i]:
                    ev = Envvehicle(self.envdict)
                    ev.desired_speed = self.envdict['speed_mean_lane' + str(i)] + np.random.randn() * self.envdict[
                        'speed_std_lane' + str(i)]
                    ev.vx = min(ev.desired_speed, vnext)

                    ev.x = -self.envdict['length_backward']
                    ev.y = i * self.envdict['lane_width']
                    if i == 0:
                        ev.color = 'b'
                    else:
                        ev.color = 'k'
                        ev.color = np.random.rand(3, )
                    lane.insert(0, ev)
                    self.nextvehicle[i] = self.calcnextvehiclefollowing(i)

                vehiclecnt = len(lane)
                for j in range(vehiclecnt):
                    veh = lane[j]
                    if isinstance(veh, Envvehicle):
                        if j + 1 < vehiclecnt:
                            vnext = lane[j + 1]
                        else:
                            vnext = None
                        veh.step(vnext, None, None)
                        if veh.x > self.envdict['length_forward']:
                            lane.remove(veh)
            if render:
                self.render()

    def render(self, close=False, rewards=None):
        plt.axes().clear()
        for i in range(self.envdict['lane_count']):
            for j in range(len(self.lanes[i])):
                self.lanes[i][j].render()

        lf = self.envdict['length_forward']
        lb = -self.envdict['length_backward']
        lw = self.envdict['lane_width']
        lc = self.envdict['lane_count']

        lines = plt.plot([lb, lf], [(lc - .5) * lw, (lc - .5) * lw], 'k')
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
            plt.xlim([-self.envdict['length_backward'] - 100, self.envdict['length_forward'] + 100])

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        th=math.atan2(self.egovehicle.vy, self.egovehicle.vx)*180/math.pi
        v = math.sqrt(self.egovehicle.vx ** 2 + self.egovehicle.vy ** 2)*3.6

        tstr='Speed: %4.2f [km/h]\nTheta:  %4.2f [deg]\nPos:  %4.2f [m]\n ' % (v,th,self.egovehicle.y)

        plt.text(0.05,0.95,tstr,transform=plt.axes().transAxes,verticalalignment='top', bbox=props, fontsize=14,family='monospace')

        if not (rewards is None):
            tstr = 'Lane reward: %5.3f\n   y reward:  %5.3f\n   v reward:  %5.3f\n   c reward:  %5.3f\n ' % (rewards[0],rewards[1],rewards[2],rewards[3])
            plt.text(0.05, 0.35, tstr, transform=plt.axes().transAxes, verticalalignment='top', bbox=props, fontsize=14,
                     family='monospace')

        plt.show(False)
        plt.pause(0.003)




    def calcnextvehiclefollowing(self, lane):
        mean = 1000 / self.envdict['density_lane' + str(lane)]
        return max(10, mean + np.random.randn() * 20)
