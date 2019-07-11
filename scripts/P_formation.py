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
    [0.5, 0, 0],
    [0.5, 1, 0],
]).T
plot_formation(formTakeoff, 'takeoff')
#%% P
letter_scale = np.array([1, 1.5, 1.5])
formP = scale_formation(np.array([
    [-0.5, -0.5, 1],
    [-0.5, 0, 1],
    [-0.25, 0.5, 0.75],
    [0, -0.5, 0.5],
    [0.5, -0.5, 0],
    [0, 0, 0.5],
]).T, letter_scale)
plot_formation(formP, 'P')

#%% U
formU = scale_formation(np.array([
    [-0.5, -0.5, 1],
    [-0.5, 0.5, 1],
    [0, 0.5, 0.5],
    [0, -0.5, 0.5],
    [0.5, -0.25, 0],
    [0.5, 0.25, 0],
]).T, letter_scale)
plot_formation(formU, 'U')
#%% 5
form5 = scale_formation(np.array([
    [-0.5, -0.5, 1],
    [-0.5, 0.5, 1],
    [0, 0.25, 0.5],
    [0, -0.5, 0.5],
    [0.5, -0.5, 0],
    [0.5, 0.25, 0],
]).T, letter_scale)
plot_formation(form5, '5')

#%% 0
form0 = scale_formation(np.array([
    [-0.5, -0.25, 1],
    [-0.5, 0.25, 1],
    [0, 0.5, 0.5],
    [0, -0.5, 0.5],
    [0.5, -0.25, 0],
    [0.5, 0.25, 0],
]).T, letter_scale)
plot_formation(form0, '0')


#%% Create waypoints for flat P -> slanted P -> rotating slanted P -> flat P
waypoints = np.array([
    formTakeoff, formTakeoff,
    formP, formP,
    formU, formU,
    form5, form5,
    form0, form0,
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
T = 20*np.ones(len(dist_max))

origin = np.array([1.5, 2, 1.5])

for drone in range(waypoints.shape[2]):
    pos_wp = waypoints[:, :, drone] + origin
    yaw_wp = np.zeros((pos_wp.shape[0], 1))
    traj = tgen.min_snap_4d(
        np.hstack([pos_wp, yaw_wp]), T, stop=True)
    trajectories.append(traj)

tgen.plot_trajectories_3d(trajectories)
tgen.trajectories_to_json(trajectories, 'scripts/data/p_form.json')
plt.show()


#%%
for traj in trajectories:
    tgen.plot_trajectory_derivatives(traj)
    print('T', T)

#%%


#%%
