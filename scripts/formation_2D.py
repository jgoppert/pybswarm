# %%

# make sure the root path is added
import inspect
import sys
import os
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
root_dir = os.path.join(path, os.path.pardir)
sys.path.insert(0, root_dir)

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import bswarm.trajectory_generation as tgen
import bswarm.formation as form
import bswarm

# %% A flat P formation.

P = np.array([
    [-1, 1],
    [0, 1],
    [1, 1],
    [-1, 0],
    [0, 0],
    [1, 0],
    [-1, -1]
]).T

def rotate_2D(P, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return R.dot(P)

# %% Create waypoints
waypoints = [P]
for theta in np.linspace(0, 2*np.pi, 5):
    waypoints.append(rotate_2D(P, theta))
waypoints = np.array(waypoints)

plt.figure()
for point in range(waypoints.shape[2]):
    plt.plot(waypoints[:, 0, point], waypoints[:, 1, point], '-')
plt.title('waypoints')

# %% plan trajectories
dist = np.linalg.norm(waypoints[1:, :, :] - waypoints[:-1, :, :], axis=1)
dist_max = np.max(dist, axis=1)
dist_max

drone_list = []

T = 10 * np.ones(dist.shape[0])

for drone in range(waypoints.shape[2]):
    planner = tgen.plan_min_accel
    trajx = planner(waypoints[:, 0, drone], T)
    trajy = planner(waypoints[:, 1, drone], T)
    drone_list.append([trajx, trajy])

# %%
plt.figure()
for trajx, trajy in drone_list:
    plt.plot(
        trajx.compute_trajectory()['x'],
        trajy.compute_trajectory()['x'])
plt.title('planned trajectories')

# %%
plt.figure()
for trajx, trajy in drone_list:
    trajx.plot()

# #%%
plt.figure()
plt.title('y trajectory')
for trajx, trajy in drone_list:
    trajy.plot()

# add plot command if runing in non-interactive mode
plt.show()

#%%
