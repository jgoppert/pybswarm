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

#%% A flat P formation.

P = 2*np.array([
    [-1, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [-1, 0, 1],
    [0, 0, 1],
    [1, 0, 1],
    [-1, -1, 1]
]).T
plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(P[0, :], P[1, :], P[2, :], 'ro')
plt.title('flat P')
plt.show()

#%% A slanted P formation.

P2 = 2*np.array([
    [-1, 1, 2],
    [0, 1, 2],
    [1, 1, 2],
    [-1, 0, 1.5],
    [0, 0, 1.5],
    [1, 0, 1.5],
    [-1, -1, 1]
]).T
plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(P[0, :], P[1, :], P[2, :], 'ro')
ax.plot3D(P2[0, :], P2[1, :], P2[2, :], 'bo')
plt.title('slanted P')
plt.show()

#%% Create waypoints for flat P -> slanted P -> rotating slanted P -> flat P
waypoints = [P]
for theta in np.linspace(0, 2 * np.pi, 8):
    waypoints.append(form.rotate_points_z(P2, theta))
waypoints.append(P)
waypoints = np.array(waypoints)

plt.figure()
ax = plt.axes(projection='3d')
for point in range(waypoints.shape[2]):
    ax.plot3D(waypoints[:, 0, point], waypoints[:, 1, point], waypoints[:, 2, point], '-')
    ax.view_init(azim=0, elev=40)
plt.title('waypoints')
plt.show()

#%% plan trajectories
dist = np.linalg.norm(waypoints[1:, :, :] - waypoints[:-1, :, :], axis=1)
dist_max = np.max(dist, axis=1)
dist_max

trajectories = []

T = dist_max

for drone in range(waypoints.shape[2]):
    pos_wp = waypoints[:, :, drone]
    yaw_wp = np.zeros((pos_wp.shape[0], 1))
    traj = tgen.min_snap_4d(
        np.hstack([pos_wp, yaw_wp]), T)
    trajectories.append(traj)

tgen.plot_trajectories_3d(trajectories)
tgen.trajectories_to_json(trajectories, 'scripts/data/p_form.json')
plt.show()


#%%
