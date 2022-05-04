# %% [markdown]
# # Polynomial
#
# $p(t) = c_0 + c_1 t + c_2 t^2 \dots = \sum\limits_{k=0}^{n-1} c_k t^k$
#
# where $c_k$ are constant coefficients
#
# ## Time Scaling
#
# We can consider time scaling by a factor of $T$.
#
# $p(t/T) = \sum\limits_{k=0}^{n-1} (c_k/T^k) t^k$
#
# ## Derivative
#
# $p' = \sum\limits_{k=0}^{n-1} (c_k k) t^{k-1} = \sum\limits_{k=1}^{n-1} (c_k k) t^{k-1} $
#
# $p'' = \sum\limits_{k=2}^{n-1} (c_k k (k-1)) t^{k-2} $
#
# $p^{(3)} = \sum\limits_{k=3}^{n-1} (c_k k (k-1) (k-2)) t^{k-3} $
#
# $p^{(m)} = \sum\limits_{k=m}^{n-1} \dfrac{c_k k!}{(k-m)!} t^{k-m} $

# %%
import numpy as np
import math
import dataclasses
import json
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from typing import List


class Trajectory1D:

    def __init__(self, T: List[float], P: List[Polynomial]):
        assert len(T) == len(P)
        self.T: List[float] = T
        self.P: List[Polynomial] = P

    def derivative(self, m):
        return Trajectory1D(self.T, [Pi.deriv(m) for Pi in self.P])

    def time_scale(self, scale: float):
        n = len(self.P[0].coef)
        coef_div = np.array([scale**k for k in range(n)])
        P_scaled = [Polynomial(Pi.coef / coef_div) for Pi in self.P]
        T_scaled = (np.array(self.T) * scale).tolist()
        return Trajectory1D(T_scaled, P_scaled)

    def coef_array(self) -> np.array:
        data = []
        for Ti, xi in zip(self.T, self.P):
            row = np.hstack([Ti, xi.coef])
            data.append(row)
        return np.array(data)

    def eval(self):
        t_list = []
        y_list = []
        S = np.hstack([0, np.cumsum(self.T)])
        legs = len(self.T)
        for leg, Pi in enumerate(self.P):
            gamma = np.arange(0, self.T[leg], 0.1)
            t_list.append(gamma + S[leg])
            y_list.append(Pi(gamma))
        return np.hstack(t_list), np.hstack(y_list)


@dataclasses.dataclass
class TrajectoryOutput:
    t: np.array     # time [s]
    pos: np.array  # position [m]
    vel: np.array   # velocity [m/s]
    acc: np.array   # acceleration [m/s^2]
    omega: np.array  # angular velocity [rad/s]
    yaw: np.array   # yaw angle [rad]
    roll: np.array  # required roll angle [rad]
    pitch: np.array  # required pitch angle [rad]

    def max_data(self):
        max_speed = np.max(np.linalg.norm(self.vel, axis=1))
        max_acc = np.max(np.linalg.norm(self.acc, axis=1))
        max_omega = np.max(np.linalg.norm(self.omega, axis=1))
        max_roll = np.rad2deg(np.max(self.roll))
        max_pitch = np.rad2deg(np.max(self.pitch))
        return \
            "max speed (m/s)\t\t\t: {max_speed:10g}\n" \
            "max acceleration (m/s^2)\t: {max_acc:10g}\n" \
            "max omega (rad/s)\t\t: {max_omega:10g}\n" \
            "max roll (deg)\t\t\t: {max_roll:10g}\n" \
            "max pitch (deg)\t\t\t: {max_pitch:10g}\n".format(**locals())


class Trajectory4D:

    def __init__(self, x: Trajectory1D, y: Trajectory1D, z: Trajectory1D, yaw: Trajectory1D):
        self.x: Trajectory1D = x
        self.y: Trajectory1D = y
        self.z: Trajectory1D = z
        self.yaw: Trajectory1D = yaw
        self.T: List[float] = x.T
        assert np.all(x.T == y.T)
        assert np.all(x.T == z.T)
        assert np.all(x.T == yaw.T)

    def time_scale(self, scale: float):
        return Trajectory4D(
            self.x.time_scale(scale),
            self.y.time_scale(scale),
            self.z.time_scale(scale),
            self.yaw.time_scale(scale))

    def derivative(self, m):
        return Trajectory4D(
            self.x.derivative(m),
            self.y.derivative(m),
            self.z.derivative(m),
            self.yaw.derivative(m))

    def eval(self):
        t, x = self.x.eval()
        y = self.y.eval()[1]
        z = self.z.eval()[1]
        yaw = self.yaw.eval()[1]
        return t, np.vstack([x, y, z, yaw]).T

    def compute_inputs(self) -> TrajectoryOutput:
        # flat variables
        t, p = self.eval()
        v = self.derivative(1).eval()[1]
        a = self.derivative(2).eval()[1]
        j = self.derivative(3).eval()[1]

        pos = p[:, :3]
        yaw = p[:, 3]
        vel = v[:, :3]
        dyaw = v[:, 3]
        acc = a[:, :3]
        jerk = j[:, :3]

        thrust = acc + np.array([0, 0, 9.81])  # add gravity

        def normalize(v):
            n = np.linalg.norm(v, axis=1)
            assert np.min(np.abs(n)) > 0
            return (v.T / n).T

        z_body = normalize(thrust)
        x_world = np.array([np.cos(yaw), np.sin(yaw), np.zeros(len(yaw))]).T
        y_body = normalize(np.cross(z_body, x_world))
        x_body = np.cross(y_body, z_body)

        def vector_dot_product(a, b):
            d = np.multiply(a, b).sum(1)
            return np.vstack([d, d, d]).T

        dprod = vector_dot_product(jerk, z_body)
        jerk_orth_zbody = jerk - np.multiply(dprod, z_body)
        h_w = (jerk_orth_zbody.T / np.linalg.norm(thrust, axis=1)).T

        omega = np.array([
            -np.multiply(h_w, y_body).sum(1),
            np.multiply(h_w, x_body).sum(1),
            np.multiply(z_body[:, 2], dyaw)]).T

        # compute required roll/pitch angles
        pitch = np.arcsin(-x_body[:, 2])
        roll = np.arctan2(y_body[:, 2], z_body[:, 2])

        return TrajectoryOutput(t, pos, vel, acc, omega, yaw, roll, pitch)

    def coef_array(self) -> np.array:
        data = []
        for Ti, xi, yi, zi in zip(self.T, self.x.P, self.y.P, self.z.P):
            row = np.hstack([Ti, xi.coef, yi.coef, zi.coef, np.zeros(8)])
            data.append(row)
        return np.array(data)

    def write_csv(self, filename: str) -> None:
        header = "Duration,x^0,x^1,x^2,x^3,x^4,x^5,x^6,x^7,y^0,y^1,y^2,y^3,y^4,y^5,y^6," \
            "y^7,z^0,z^1,z^2,z^3,z^4,z^5,z^6,z^7,yaw^0,yaw^1,yaw^2,yaw^3,yaw^4,yaw^5,yaw^6,yaw^7"
        np.savetxt(filename, self.coef_array(),
                   delimiter=',', fmt='%15g', header=header)


def min_deriv_1d(deriv: int, waypoints: List[List[float]], T: List[float], stop: bool) -> Trajectory1D:
    n = deriv * 2  # number of poly coeff (order + 1)
    S = np.hstack([0, np.cumsum(T)])
    legs = len(T)
    assert len(waypoints) == len(T) + 1

    def coef_weights(t: float, m: int, t0: float):
        """
        Polynomial coefficient weights
        :param t: time
        :param m: derivative order
        """
        w = np.zeros(n)
        for k in range(m, n):
            w[k] = (t - t0)**(k - m) * math.factorial(k) / \
                math.factorial(k - m)
        return w

    b = np.zeros(n * legs)
    A = np.zeros((n * legs, n * legs))
    eq = 0
    for leg in range(legs):
        # first waypoint
        if leg == 0:
            for m in range(n // 2):
                A[eq, n * leg:n * (leg + 1)
                  ] = coef_weights(t=S[leg], m=m, t0=S[leg])
                if m == 0:
                    b[eq] = waypoints[leg]
                else:
                    b[eq] = 0
                eq += 1
        # any waypoint except for first
        else:
            for m in range(n // 2 - 1):
                if m == 0:
                    A[eq, n * leg:n *
                        (leg + 1)] = coef_weights(t=S[leg], m=m, t0=S[leg])
                    b[eq] = waypoints[leg]
                    eq += 1
                elif stop:
                    A[eq, n * leg:n *
                        (leg + 1)] = coef_weights(t=S[leg], m=m, t0=S[leg])
                    b[eq] = 0
                    eq += 1

        # last waypoint only
        if leg == legs - 1:
            for m in range(n // 2):
                A[eq, n * leg:n *
                    (leg + 1)] = coef_weights(t=S[leg + 1], m=m, t0=S[leg])
                if m == 0:
                    b[eq] = waypoints[leg + 1]
                else:
                    b[eq] = 0
                eq += 1

        # continuity
        if leg > 0:
            for m in range(n // 2 + 1):
                A[eq, n * (leg - 1):n * leg] = coef_weights(t=S[leg],
                                                            m=m, t0=S[leg - 1])
                A[eq, n * leg:n * (leg + 1)] = - \
                    coef_weights(t=S[leg], m=m, t0=S[leg])
                b[eq] = 0
                eq += 1

    if eq != n * legs:
        print('warning: equations: {:d}, coefficients: {:d}'.format(
            eq, n * legs))
    c = np.linalg.pinv(A).dot(b)
    P_list = []
    for leg in range(legs):
        Pi = Polynomial(c[n * leg:n * (leg + 1)])
        P_list.append(Pi)
    return Trajectory1D(T, P_list)


def min_deriv_4d(deriv: int, waypoints: List[List[float]], T: List[float], stop: bool) -> Trajectory4D:
    traj_x = min_deriv_1d(deriv, waypoints[:, 0], T, stop)
    traj_y = min_deriv_1d(deriv, waypoints[:, 1], T, stop)
    traj_z = min_deriv_1d(deriv, waypoints[:, 2], T, stop)
    traj_yaw = min_deriv_1d(deriv, waypoints[:, 3], T, stop)
    return Trajectory4D(traj_x, traj_y, traj_z, traj_yaw)


def min_snap_1d(waypoints: List[List[float]], T: List[float], stop: bool) -> Trajectory1D:
    return min_deriv_1d(4, waypoints, T, stop)


def min_snap_4d(waypoints: List[List[float]], T: List[float], stop: bool) -> Trajectory4D:
    return min_deriv_4d(4, waypoints, T, stop)


def min_accel_1d(waypoints: List[List[float]], T: List[float], stop: bool) -> Trajectory1D:
    return min_deriv_1d(2, waypoints, T, stop)


def min_accel_4d(waypoints: List[List[float]], T: List[float], stop: bool) -> Trajectory4D:
    return min_deriv_4d(2, waypoints, T, stop)


def plot_trajectories_time_history(trajectories: List[Trajectory4D]) -> None:
    names = ['pos', 'vel', 'acc', 'jerk', 'snap']
    for i, name in enumerate(names):
        plt.subplot(len(names), 1, i + 1)
        for traj in trajectories:
            plt.plot(*traj.derivative(i + 1).eval())
        plt.xlabel('t, sec')
        plt.grid()
        plt.ylabel(name)


def plot_trajectories_magnitudes(trajectories: List[Trajectory4D]) -> None:
    names = ['pos', 'vel', 'acc', 'jerk', 'snap']
    for i, name in enumerate(names):
        plt.subplot(len(names), 1, i + 1)
        for traj in trajectories:
            t, a = traj.derivative(i).eval()
            plt.plot(t, np.linalg.norm(a, axis=1))
        plt.xlabel('t, sec')
        plt.ylabel(name)


def plot_trajectories(trajectories: List[Trajectory4D]) -> None:
    dataLines = []
    for traj in trajectories:
        x = traj.eval()[1]
        dataLines.append(x.T)

    fig = plt.gcf()
    ax = p3.Axes3D(fig)
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_zlabel('z, m')
    border = 0.1
    all_x = np.array([dat[0, :] for dat in dataLines])
    ax.set_xlim3d(np.floor(np.min(all_x) - border),
                  np.ceil(np.max(all_x) + border))
    all_y = np.array([dat[1, :] for dat in dataLines])
    ax.set_ylim3d(np.floor(np.min(all_y) - border),
                  np.ceil(np.max(all_y) + border))
    all_z = np.array([dat[2, :] for dat in dataLines])
    ax.set_zlim3d(np.floor(np.min(all_z) - border),
                  np.ceil(np.max(all_z) + border))
    lines = [ax.plot(dat[0, :], dat[1, :], dat[2, :])[0] for dat in dataLines]

def animate_trajectories(filename, trajectories, fps):
    # Attaching 3D axis to the figure
    fig = plt.figure(figsize=(10, 10))
    ax = p3.Axes3D(fig)
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_zlabel('z, m')

    # create trajectory data
    dataLines = []
    for traj in trajectories:
        x = traj.eval()[1]
        dataLines.append(x.T)
    lines = [ax.plot([dat[0, 0]], [dat[1, 0]], [dat[2, 0]],
                     'ro', markersize=10)[0] for dat in dataLines]

    fps = 5  # frames per second
    data_period = 0.1  # data period, seconds
    data_length = dataLines[0].shape[1]
    duration = data_period * data_length
    frames = int(np.floor(duration * fps))
    step = data_length // frames

    def update_lines(num, dataLines, lines):
        for line, data in zip(lines, dataLines):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data([
                [data[0, num * step]],
                [data[1, num * step]],
            ])
            line.set_3d_properties([data[2, num * step]])
        return lines

    border = 0.1
    all_x = np.array([dat[0, :] for dat in dataLines])
    ax.set_xlim3d(np.floor(np.min(all_x) - border),
                  np.ceil(np.max(all_x) + border))
    all_y = np.array([dat[1, :] for dat in dataLines])
    ax.set_ylim3d(np.floor(np.min(all_y) - border),
                  np.ceil(np.max(all_y) + border))
    all_z = np.array([dat[2, :] for dat in dataLines])
    ax.set_zlim3d(np.floor(np.min(all_z) - border),
                  np.ceil(np.max(all_z) + border))

    ani = animation.FuncAnimation(
        fig, update_lines, frames, fargs=(dataLines, lines),
        interval=int(1000 / fps), blit=False)
    ani.save(filename)
    plt.close()


def trajectories_to_json(trajectories: List[Trajectory4D]):
    formation = {}
    for drone, traj in enumerate(trajectories):
        formation[drone] = traj.coef_array().tolist()
    return formation
