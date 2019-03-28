from gym_highway.modell.vehicle_base import BaseVehicle
import numpy as np
import math


class EgoVehicle(BaseVehicle):

    def __init__(self, dict_base):
        super().__init__(dict_base)
        self.desired_speed = dict_base['speed_ego_desired']
        self.color = 'r'
        self.lane_index = 0

    def vehicle_onestep(self, vehicle_state, action, dt):
        """
        :param vehicle_state: np.array([x,y,th,v])
                            x,y - position ([m,m])
                            th  - angle ([rad] zero at x direction,CCW)
                            v   - velocity ([m/s])
        :param action: np.array([steering, acceleration])
                            steering     - angle CCW [rad]
                            acceleration - m/s^2
        :param dt: sample time [s]
        :return:the new vehicle state in same structure as the vehicle_state param
        """
        # Fixed Vehicle axle length
        axle_length = self.length-1

        state = vehicle_state
        # The new speed v'=v+dt*a
        state[3] = max(0, vehicle_state[3] + dt * action[1])
        # The travelled distance s=(v+v')/2*dt
        s = (state[3] + vehicle_state[3]) / 2 * dt

        if action[0] == 0:  # Not steering
            # unit vector
            dx = math.cos(state[2])
            dy = math.sin(state[2])
            state[0] = vehicle_state[0] + dx * s
            state[1] = vehicle_state[1] + dy * s
        else:  # Steering
            # turning_radius=axle_length/tanh(steering)
            turning_radius = axle_length / math.tanh(action[0])
            # The new theta heading th'=th+s/turning_radius
            turn = s / turning_radius
            state[2] = vehicle_state[2] + turn
            # TODO: ezt nem értem miért kell, vagy miért nem turn van csekkolva
            if math.pi < state[2]:
                state[2] = state[2] - 2 * math.pi
            if -math.pi > state[2]:
                state[2] = state[2] + 2 * math.pi
            # new position
            # transpose distance dist=|2*turning_radius*sin(turn/2)|
            dist = abs(2 * turning_radius * math.sin(turn / 2))
            # transpose angle ang=th+turn/2
            ang = vehicle_state[2] + turn / 2
            # unit vector
            dx = math.cos(ang)
            dy = math.sin(ang)
            # new position
            state[0] = vehicle_state[0] + dx * dist
            state[1] = vehicle_state[1] + dy * dist
        return state

    def step(self, action):
        th = math.atan2(self.vy, self.vx)
        v = math.sqrt(self.vx ** 2 + self.vy ** 2)

        state = np.array([self.x, self.y, th, v])
        new_state = self.vehicle_onestep(state, action, self.env_dict['dt'])
        self.x = new_state[0]
        self.y = new_state[1]
        self.vx = new_state[3] * math.cos(new_state[2])
        self.vy = new_state[3] * math.sin(new_state[2])
        self.desired_speed = self.vx

