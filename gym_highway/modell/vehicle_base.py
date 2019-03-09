"""
    Base Vehicle Class
    Vehicle State is vector
        0 - Position longitudnal [m]
        1 - Position lateral [m]
        2 - Heading (dir->x =0, CCW) [rad]
        3 - Speed x direction [m/s]
        4 - Speed y direction [m/s]
"""
import numpy as np
import matplotlib.pyplot as plt


class BaseVehicle:
    def __init__(self, dict_base):
        self.env_dict = dict_base
        self.dt = self.env_dict['dt']
        self.length = self.env_dict['car_length']  # vehicle length in [m]
        self.max_acc = dict_base['max_acceleration']  # Max acceleration m/s^2
        self.max_dec = dict_base['max_deceleration']  # Max deceleration m/s^2
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.color = 'b'

    def render(self):
        x = self.x
        y = self.y
        l = self.length
        plt.plot([x, x, x + self.length, x + self.length, x], [y - 1, y + 1, y + 1, y - 1, y - 1], self.color)
