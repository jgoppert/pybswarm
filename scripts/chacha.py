# %%
import json
import bswarm
import bswarm.formation as formation
import bswarm.trajectory as tgen
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
import os
sys.path.insert(0, os.getcwd())


def plot_formation(F, name):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(F[0, :], F[1, :], F[2, :], 'ro')
    plt.title(name)
    plt.show()


def scale_formation(form, scale):
    formNew = np.copy(form)
    for i in range(3):
        formNew[i, :] *= scale[i]
    return formNew


# %% takeoff
formTakeoff = np.array([
    [-0.5, -1, 0],
    [-0.5, 0, 0],
    [-0.5, 1, 0],
    [0.5, -1, 0],
]).T
plot_formation(formTakeoff, 'takeoff')
# %% Square
form_scale = np.array([1.5, 1.5, 1.5])
form = scale_formation(np.array([
    [0.5, -0.5, 0],
    [-0.5, -0.5, 0],
    [-0.5, 0.5, 0],
    [0.5, 0.5, 0]
]).T, form_scale)
plot_formation(form, 'formation')

waypoints = []


def left(T):
    waypoints.append()


class ChaCha():

    def __init__(self):
        self.waypoints = [form]
        self.T = []

    def wait(self, duration):
        self.waypoints.append(form)
        self.T.append(duration)

    def move(self, duration, vector):
        self.waypoints.append(form + vector)
        self.T.append(duration / 2)
        self.waypoints.append(form)
        self.T.append(duration / 2)

    def move_twice(self, duration, vector):
        self.waypoints.append(form + vector)
        self.T.append(duration / 4)
        self.waypoints.append(form)
        self.T.append(duration / 4)
        self.waypoints.append(form + vector)
        self.T.append(duration / 4)
        self.waypoints.append(form)
        self.T.append(duration / 4)

    def rotate(self, duration, angle):
        self.waypoints.append(formation.rotate_points_z(form, np.deg2rad(angle)))
        self.T.append(duration / 2)
        self.waypoints.append(form)
        self.T.append(duration / 2)

    def chacha(self, duration):
        self.rotate(duration, 20)

    def left(self, duration):
        self.move(duration, np.array([[0, -0.5, 0]]).T)

    def right(self, duration):
        self.move(duration, np.array([[0, 0.5, 0]]).T)

    def back(self, duration):
        self.move(duration, np.array([[-0.5, 0, 0]]).T)

    def hop(self, duration):
        self.move(duration, np.array([[0, 0, 0.2]]).T)

    def two_hops(self, duration):
        self.move_twice(duration, np.array([[0, 0, 0.1]]).T)

    def right_stomp(self, duration):
        self.move(duration, np.array([[0, 0.2, -0.2]]).T)

    def left_stomp(self, duration):
        self.move(duration, np.array([[0, 0.2, -0.2]]).T)

    def right_two_stomps(self, duration):
        self.move_twice(duration, np.array([[0, 0.2, -0.2]]).T)

    def left_two_stomps(self, duration):
        self.move_twice(duration, np.array([[0, -0.2, -0.2]]).T)

    def slide_left(self, duration):
        self.left(duration)

    def slide_right(self, duration):
        self.right(duration)

    def crisscross(self, duration):
        self.rotate(duration, 45)

    def turn_it_out(self, duration):
        self.rotate(duration, -45)

    def funky(self, duration):
        self.move(duration, np.array([[0.1, 0.1, 0]]).T)

c = ChaCha()
for i in range(17):
    c.wait(2)
for i in range(2):
    c.left(2)
    c.back(2)
    c.hop(2)
    c.right_stomp(3)
    c.left_stomp(3)
    c.chacha(5)
    c.turn_it_out(2)
    c.left(2)
    c.back(3)
    c.two_hops(2)
    c.right_stomp(2)
    c.left_stomp(3)
    c.chacha(4)
    c.funky(2)
    c.right(2)
    c.left(2)
    c.back(2)
    c.hop(2)
    c.hop(3)
    c.right_two_stomps(3)
    c.left_two_stomps(2)
    c.slide_left(2)
    c.slide_right(3)
    c.crisscross(2)
    c.crisscross(2)
    c.chacha(5)

origin = np.array([1.5, 2, 2])
waypoints = np.array(c.waypoints)
T = c.T

trajectories = []
for drone in range(waypoints.shape[2]):
    pos_wp = waypoints[:, :, drone] + origin
    yaw_wp = np.zeros((pos_wp.shape[0], 1))
    traj = tgen.min_deriv_4d(4,
                             np.hstack([pos_wp, yaw_wp]), T, stop=False)
    trajectories.append(traj)

tgen.plot_trajectories(trajectories)
plt.show()

# %%
tgen.plot_trajectories_time_history(trajectories)

# %%
tgen.animate_trajectories('chacha.mp4', trajectories, fps=5)

#%%
