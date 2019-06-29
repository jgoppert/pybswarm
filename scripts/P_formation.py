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

scale = 0.7
P = scale*np.array([
    [-1, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
    [-1, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [-1, -1, 0]
]).T
P[:, ]
plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(P[0, :], P[1, :], P[2, :], 'ro')
plt.title('flat P')
plt.show()

#%% A slanted P formation.

P2 = np.array(P)
P2[2, :3] = 3
P2[2, 3:6] = 1.5
P2[2, 6] = 0

plt.figure()
ax = plt.axes(projection='3d')
#ax.plot3D(P[0, :], P[1, :], P[2, :], 'ro')
ax.plot3D(P2[0, :], P2[1, :], P2[2, :], 'bo')
plt.title('slanted P')
plt.show()

#%% Create waypoints for flat P -> slanted P -> rotating slanted P -> flat P
shift = np.array([[1, 0, 0]]).T
waypoints = [P, P + shift]
for theta in np.linspace(0, -2 * np.pi, 8):
    waypoints.append(form.rotate_points_z(P2, theta) + shift)
waypoints.append(P2)
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

T = 3*np.ones(len(dist_max))
T[2] = 6

origin = np.array([2, 3, 2])

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
tgen.plot_trajectory_derivatives(trajectories[0])
print('T', T)

#%%
