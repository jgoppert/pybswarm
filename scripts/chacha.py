#%%
import sys
import os
sys.path.insert(0, os.getcwd())

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import bswarm.trajectory as tgen
import bswarm.formation as form
import bswarm
import json

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

#%% takeoff
formTakeoff = np.array([
    [-0.5, -1, 0],
    [-0.5, 0, 0],
    [-0.5, 1, 0],
    [0.5, -1, 0],
]).T
plot_formation(formTakeoff, 'takeoff')
#%% Square
letter_scale = np.array([1.5, 1.5, 1.5])
form = scale_formation(np.array([
    [0.5, -0.5, 0],
    [-0.5, -0.5, 0],
    [-0.5, 0.5, 0],
    [0.5, 0.5, 0]
]).T, letter_scale)
plot_formation(form, 'formation')

left = np.array([[0, -0.5, 0]]).T
right = np.array([[0, 0.5, 0]]).T
front = np.array([[-0.5, 0, 0]]).T
back = np.array([[0.5, 0, 0]]).T
hop = np.array([[0, 0, 1]]).T
down = np.array([[0, 0, 1]]).T

class ChaCha():

    def __init__(self, origin):
        self.origin = origin
        self.pos = np.array([[0, 0, 0]]).T
        self.traj_list[]

    def chacha(self, T):
        self.pos = np.array([[0, 0, 0]]).T
        self.plan()

    def left(self, T):
        self.pos = self.pos + np.array([[0, -0.5, 0]]).T

    def right(self, T):
        self.pos = self.pos + np.array([[0, 0.5, 0]]).T

    def back(self):
        self.pos = self.pos + np.array([[0.5, 0, 0]]).T


    def plan(self, waypoints, T):
        pos_wp = waypoints[:, :, drone] + origin
        yaw_wp = np.zeros((pos_wp.shape[0], 1))
        traj = tgen.min_snap_4d(
            np.hstack([pos_wp, yaw_wp]), T, stop=True)
        self.trajlist.extend(Traj)


chacha = ChaCha()
chacha.wait(34)
for i in range(2):
    chacha.left(2)
    chacha.back(2)
    chacha.hop(2)
    chacha.right_stomp(3)
    chacha.left_stomp(3)
    chacha.chacha(5)
    chacha.turn_it_out(2)
    chacha.left(2)
    chacha.back(3)
    chacha.hop2(2)
    chacha.right_stomp(2)
    chacha.left_stomp(3)
    chacha.chacha(4)
    chacha.funk(2)
    chacha.right(2)
    chacha.left(2)
    chacha.back(2)
    chacha.hop(2)
    chacha.hop(3)
    chacha.right_two_stomps(3)
    chacha.left_two_stomps(2)
    chacha.slide_left(2)
    chacha.slide_right(3)
    chacha.crisscross(2)
    chacha.crosscross(2)
    chacha.chacha(5)

#%% Create waypoints for flat P -> slanted P -> rotating slanted P -> flat P
waypoints = np.array([
    form, form,
    form + hop, form,
    form + hop, form,
    form + hop, form,
    form + hop, form,
    form + left, form + left,
    form + left + back, form + left + back,
    form + left + back + hop, form + left + back,
    form + left + back + hop, form + left + back,
    form + left + back + hop, form + left + back,
    formTakeoff, formTakeoff])

plt.figure()
ax = plt.axes(projection='3d')
for point in range(waypoints.shape[2]):
    ax.plot3D(waypoints[:, 0, point], waypoints[:, 1, point], waypoints[:, 2, point], '-')
    ax.view_init(azim=45, elev=40)
plt.title('waypoints')
plt.show()

#%% plan trajectories
dist = np.linalg.norm(waypoints[1:, :, :] - waypoints[:-1, :, :], axis=1)
dist_max = np.max(dist, axis=1)
dist_max

trajectories = []

#T = 3*np.ones(len(dist_max))
T = 1*np.ones(len(dist_max))

origin = np.array([1.5, 2, 2])

for drone in range(waypoints.shape[2]):
    pos_wp = waypoints[:, :, drone] + origin
    yaw_wp = np.zeros((pos_wp.shape[0], 1))
    traj = tgen.min_snap_4d(
        np.hstack([pos_wp, yaw_wp]), T, stop=False)
    trajectories.append(traj)

tgen.plot_trajectories_3d(trajectories)
tgen.trajectories_to_json(trajectories, 'scripts/data/chacha.json')
plt.show()


#%%
for traj in trajectories:
    tgen.plot_trajectory_derivatives(traj)
    print('T', T)

#%%


#%%
