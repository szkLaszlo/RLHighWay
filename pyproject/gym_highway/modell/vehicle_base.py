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

    def render(self, axes, zoom=None):

        if zoom is None:
            zoom = 1
        x = self.x
        y = self.y * zoom
        l = self.length * zoom / 2
        axes.plot([x-l, x-l, x + l, x + l, x-l], [y - l/2, y + l/2, y + l/2, y - l/2, y - l/2], self.color)
