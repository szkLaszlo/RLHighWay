"""
   Simple kinematics vehicle modell
"""
import numpy as np
import math


def vehicle_onestep(vehiclestate, action, dt):
    """

    :param vehiclestate: np.array([x,y,th,v])
                        x,y - position ([m,m])
                        th  - angle ([rad] zero at x direction,CCW)
                        v   - velocity ([m/s])
    :param action: np.array([steering, acceleration])
                        steering     - angle CCW [rad]
                        acceleration - m/s^2
    :param dt: sample time [s]
    :return:the new vehicle state in same structure as the vehiclestate param
    """
    # Fixed Vehicle axle length
    L = 3

    state = vehiclestate
    # The new speed v'=v+dt*a
    state[3] = max(0, vehiclestate[3] + dt * action[1])
    # The travelled distance s=(v+v')/2*dt
    s = (state[3] + vehiclestate[3]) / 2 * dt

    if action[0] == 0:  # Not steering
        # unit vector
        dx = math.cos(state[2])
        dy = math.sin(state[2])
        state[0] = vehiclestate[0] + dx * s
        state[1] = vehiclestate[1] + dy * s
    else:  # Steering
        # Turning Radius R=axlelength/tanh(steering)
        R = L / math.tanh(action[0])
        # The new theta heading th'=th+s/R
        turn = s / R
        state[2] = vehiclestate[2] + turn
        if math.pi < state[2]:
            state[2] = state[2] - 2 * math.pi
        if -math.pi > state[2]:
            state[2] = state[2] + 2 * math.pi
        # new position
        # transpose distance dist=2*R*sin(|turn/2|)
        dist = abs(2 * R * math.sin(turn / 2))
        # transpose angle ang=th+turn/2
        ang = vehiclestate[2] + turn / 2
        # unit vector
        dx = math.cos(ang)
        dy = math.sin(ang)
        # new position
        state[0] = vehiclestate[0] + dx * dist
        state[1] = vehiclestate[1] + dy * dist
    return state
