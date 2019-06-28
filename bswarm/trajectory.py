#%% [markdown]
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

#%%
import numpy as np
import math
import dataclasses
import json
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        P_scaled = [Polynomial(Pi.coef/coef_div) for Pi in self.P]
        T_scaled = (np.array(self.T)*scale).tolist()
        print('T_scaled', T_scaled)
        return Trajectory1D(T_scaled, P_scaled)

    def eval(self):
        t_list = []
        y_list = []
        S = np.hstack([0, np.cumsum(self.T)])
        legs = len(self.T)
        for leg, Pi in enumerate(self.P):
            ti = np.linspace(0, self.T[leg], 1000)
            t_list.append(ti + S[leg])
            y_list.append(Pi(ti))
        return np.hstack(t_list), np.hstack(y_list)


@dataclasses.dataclass
class TrajectoryOutput:
    t: np.array     # time [s]
    pos : np.array  # position [m]
    vel: np.array   # velocity [m/s]
    acc: np.array   # acceleration [m/s^2]
    omega: np.array # angular velocity [rad/s]
    yaw: np.array   # yaw angle [rad]
    roll: np.array  # required roll angle [rad]
    pitch: np.array # required pitch angle [rad]

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

        thrust = acc + np.array([0, 0, 9.81]) # add gravity

        def normalize(v):
            n = np.linalg.norm(v, axis=1)
            assert np.min(np.abs(n)) > 0
            return (v.T/n).T

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
        np.savetxt(filename, self.coef_array(), delimiter=',', fmt='%15g',header=header)


def min_snap_1d(waypoints: List[List[float]], T: List[float]) -> Trajectory1D:
    n = 8
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
            w[k] = (t - t0)**(k-m)*math.factorial(k)/math.factorial(k-m)
        return w
        
    b = np.zeros(n*legs)
    A = np.zeros((n*legs, n*legs))
    eq = 0
    for leg in range(legs):
        # every waypoint
        for m in range(3):
            A[eq, n*leg:n*(leg + 1)] = coef_weights(t=S[leg], m=m, t0=S[leg])
            if m == 0:
                b[eq] = waypoints[leg]
            else:
                b[eq] = 0
            eq += 1
        
        # first waypoint
        if leg == 0:
            for m in [3]:
                A[eq, n*leg:n*(leg + 1)] = coef_weights(t=S[leg], m=m, t0=S[leg])
                b[eq] = 0
                eq += 1
    
        # last waypoint
        if leg == legs - 1:
            for m in range(4):
                A[eq, n*leg:n*(leg + 1)] = coef_weights(t=S[leg+1], m=m, t0=S[leg])
                if m == 0:
                    b[eq] = waypoints[leg + 1]
                else:
                    b[eq] = 0
                eq += 1
                
        # continuity
        if leg > 0:
            for m in range(5):
                A[eq, n*(leg-1):n*leg] = coef_weights(t=S[leg], m=m, t0=S[leg-1])
                A[eq, n*leg:n*(leg + 1)] = -coef_weights(t=S[leg], m=m, t0=S[leg])
                b[eq] = 0
                eq += 1

    if eq != n*legs:
        print('warning: equations: {:d}, coefficients: {:d}'.format(eq, n*legs))
    c = np.linalg.pinv(A).dot(b)
    P_list = []
    for leg in range(legs):
        Pi = Polynomial(c[n*leg:n*(leg + 1)])
        P_list.append(Pi)
    return Trajectory1D(T, P_list)


def min_snap_4d(waypoints: List[List[float]], T: List[float]) -> Trajectory4D:
    traj_x = min_snap_1d(waypoints[:, 0], T)
    traj_y = min_snap_1d(waypoints[:, 1], T)
    traj_z = min_snap_1d(waypoints[:, 2], T)
    traj_yaw = min_snap_1d(waypoints[:, 3], T)
    return Trajectory4D(traj_x, traj_y, traj_z, traj_yaw)


def plot_trajectory_derivatives(traj) -> None:
    names = ['pos', 'vel',
            'acc', 'jerk', 'snap']
    for i, name in enumerate(names):
        plt.subplot(len(names), 1, i + 1)
        plt.plot(*traj.derivative(i).eval())
        plt.xlabel('t, sec')
        plt.grid()
        plt.ylabel(name)


def plot_trajectory_3d(traj) -> None:
    ax = plt.gca(projection='3d')
    t, p = traj.eval()
    ax.plot(p[:, 0], p[:, 1], p[:, 2])
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_zlabel('z, m')
    box_size = np.max(np.max(p[:, :3], axis=0) - np.min(p[:, :3], axis=0))/2
    decimals = 1
    print('box_size', box_size)
    ax.set_xlim(
        np.round(np.mean(p[:, 0]) - box_size, decimals),
        np.round(np.mean(p[:, 0]) + box_size, decimals))
    ax.set_ylim(
        np.round(np.mean(p[:, 1]) - box_size, decimals),
        np.round(np.mean(p[:, 1]) + box_size, decimals))
    ax.set_zlim(
        np.round(np.mean(p[:, 2]) - box_size, decimals),
        np.round(np.mean(p[:, 2]) + box_size, decimals))


def plot_trajectories_3d(trajectories) -> None:
    t, p = trajectories[0].eval()
    p_list = []
    for traj in trajectories:
        p_list.append(traj.eval()[1])
    p_list = np.array(p_list)

    ax = plt.gca(projection='3d')
    for p in p_list:   
        ax.plot3D(p[:, 0], p[:, 1], p[:, 2])
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_zlabel('z, m')
    box_size = np.max(
        np.max(p_list, axis=1) - np.min(p_list, axis=1))/2
    center = (np.max(np.max(p_list, axis=1), axis=0) +
        np.min(np.min(p_list, axis=1), axis=0))/2
    decimals = 1

    ax.set_xlim(
        np.round(center[0] - box_size, decimals),
        np.round(center[0] + box_size, decimals))
    ax.set_ylim(
        np.round(center[1] - box_size, decimals),
        np.round(center[1] + box_size, decimals))
    ax.set_zlim(
        np.round(center[2] - box_size, decimals),
        np.round(center[2] + box_size, decimals))

def trajectories_to_json(trajectories: List[Trajectory4D], filename: str):
    formation = {}
    for drone, traj in enumerate(trajectories):
        formation[drone] = traj.coef_array().tolist()
    with open(filename, 'w') as f:
        json.dump(formation, f)