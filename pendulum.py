from math import sin, cos
import numpy as np
from numpy import ndarray
from numpy.linalg import inv
import pandas as pd


class Pendulum:
    def __init__(self, identifier: int, m2: float):
        # Constants
        self.m1, self.m2 = 2.0, m2
        self.l1, self.l2 = 1, 1
        self.g = 9.81
        self.scale = 100

        # Data settings
        self.data = []
        self.identifier = identifier

    def reset(self):
        self.data = []

    def next_state(self, state: ndarray, t: float) -> ndarray:
        """
        Calculate the values of the next state matrix, following lagrangian formulas of motion.
        :param state: current state
        :param t: current time
        :return:
        """
        a1d, a2d = state[0], state[1]
        a1, a2 = state[2], state[3]

        m11, m12 = (self.m1 + self.m2) * self.l1, self.m2 * self.l2 * cos(a1 - a2)
        m21, m22 = self.l1 * cos(a1 - a2), self.l2
        m = np.array([[m11, m12], [m21, m22]])

        f1 = -self.m2 * self.l2 * a2d * a2d * sin(a1 - a2) - (self.m1 + self.m2) * self.g * sin(a1)
        f2 = self.l1 * a1d * a1d * sin(a1 - a2) - self.g * sin(a2)
        f = np.array([f1, f2])

        accel = inv(m).dot(f)

        return np.array([accel[0], accel[1], a1d, a2d])

    def rk4_step(self, state: ndarray, t: float, dt: float) -> ndarray:
        """
        Approximate next state matrix by applying Runge-Kutta method for derivatives.
        :param state: current state
        :param t: current time
        :param dt: time discretion
        :return: next state
        """
        k1 = self.next_state(state, t)
        k2 = self.next_state(state + 0.5 * k1 * dt, t + 0.5 * dt)
        k3 = self.next_state(state + 0.5 * k2 * dt, t + 0.5 * dt)
        k4 = self.next_state(state + k3 * dt, t + dt)

        return dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def get_points(self, a1: float, a2: float) -> ndarray:
        """
        Calculate positions of point-masses based on angles.
        :param a1: angle 1
        :param a2: angle 2
        :return:
        """
        x1 = self.l1 * self.scale * sin(a1)
        y1 = self.l1 * self.scale * cos(a1)
        x2 = x1 + self.l2 * self.scale * sin(a2)
        y2 = y1 + self.l2 * self.scale * cos(a2)

        return np.array([x1, y1, x2, y2])

    def simulate(self, a1: float, a2: float, duration: float):
        # Initial config
        t = 0.0
        delta_t = 0.02
        state = np.array([0.0, 0.0, a1, a2])

        # Algorithm
        c = 0
        while t < duration:
            # Get point mass positions
            coordinates = self.get_points(state[2], state[3])

            # Save data
            self.data.append(np.concatenate((coordinates, state), axis=0))

            # Get next state
            t += delta_t
            state = state + self.rk4_step(state, t, delta_t)

            # Increment iteration
            c += 1

    def save(self) -> pd.DataFrame:
        df = pd.DataFrame(data=self.data, columns=['x1', 'y1', 'x2', 'y2', 'v1', 'v2', 'a1', 'a2'])
        df['id'] = self.identifier
        df['m2'] = self.m2
        df = df.round(2)
        return df

    def run(self, a1: float, a2: float, duration: float) -> pd.DataFrame:
        self.reset()
        self.simulate(a1, a2, duration)
        df = self.save()
        return df
