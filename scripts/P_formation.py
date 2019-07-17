#%%
import sys
import os
sys.path.insert(0, os.getcwd())

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import bswarm.trajectory as tgen
import bswarm.formation as formation
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
letter_scale = np.array([1.5, 1.5, 1.5])
formLetter = {}
formLetter['P'] = scale_formation(np.array([
    [-0.5, -0.5, 1],
    [-0.5, 0, 1],
    [-0.25, 0.5, 0.75],
    [0, -0.5, 0.5],
    [0.5, -0.5, 0],
    [0, 0, 0.5],
]).T, letter_scale)
plot_formation(formLetter['P'], 'P')

#%% U
formLetter['U'] = scale_formation(np.array([
    [-0.5, -0.5, 1],
    [-0.5, 0.5, 1],
    [0, 0.5, 0.5],
    [0, -0.5, 0.5],
    [0.5, -0.25, 0],
    [0.5, 0.25, 0],
]).T, letter_scale)
plot_formation(formLetter['U'], 'U')
#%% 5
formLetter['5'] = scale_formation(np.array([
    [-0.5, -0.5, 1],
    [-0.5, 0.5, 1],
    [0, 0.25, 0.5],
    [0, -0.5, 0.5],
    [0.5, -0.5, 0],
    [0.5, 0.25, 0],
]).T, letter_scale)
plot_formation(formLetter['5'], '5')

#%% O
formLetter['O'] = scale_formation(np.array([
    [-0.5, -0.25, 1],
    [-0.5, 0.25, 1],
    [0, 0.5, 0.5],
    [0, -0.5, 0.5],
    [0.5, -0.25, 0],
    [0.5, 0.25, 0],
]).T, letter_scale)
plot_formation(formLetter['O'], 'O')


#%% A
formLetter['A'] = scale_formation(np.array([
    [0, -0.5, 0.5],
    [-0.5, 0, 1],
    [0, 0.5, 0.5],
    [0.5, -0.5, 0],
    [0, 0, 0.5],
    [0.5, 0.5, 0],
]).T, letter_scale)
plot_formation(formLetter['A'], 'A')

#%% L
formLetter['L'] = scale_formation(np.array([
    [-0.5, -0.5, 1],
    [0, -0.5, 0.5],
    [0, 0.5, 0.5],
    [0.5, -0.5, 0],
    [0.5, 0, 0],
    [0.5, 0.5, 0],
]).T, letter_scale)
plot_formation(formLetter['L'], 'L')

#%% 1
formLetter['1'] = scale_formation(np.array([
    [-0.5, -0.5, 1],
    [-0.5, 0, 1],
    [0, 0, 0.5],
    [0.5, -0.5, 0],
    [0.5, 0, 0],
    [0.5, 0.5, 0],
]).T, letter_scale)
plot_formation(formLetter['1'], '1')


#%% Create waypoints for flat P -> slanted P -> rotating slanted P -> flat P
waypoints = []
for letter in 'P U A P O L L O'.split(' '):
    form = formLetter[letter]
    waypoints.extend([formTakeoff, form, form])
    #for theta in np.linspace(0, 2*np.pi, 6)[1:]:
    #    waypoints.append(formation.rotate_points_z(form, theta))
waypoints.extend([formTakeoff, formTakeoff, formTakeoff])
waypoints = np.array(waypoints)

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


wait_time = 5
origin = np.array([1.5, 2, 2])

trajectories = []
T = dist_max*2
T = np.where(T == 0, wait_time, T)
for drone in range(waypoints.shape[2]):
    pos_wp = waypoints[:, :, drone] + origin
    yaw_wp = np.zeros((pos_wp.shape[0], 1))
    traj = tgen.min_deriv_4d(4, 
        np.hstack([pos_wp, yaw_wp]), T, stop=False)
    trajectories.append(traj)

tgen.plot_trajectories(trajectories)
plt.show()

tgen.trajectories_to_json(trajectories, 'scripts/data/p_form.json')

#%%
plt.figure()
tgen.plot_trajectories_time_history(trajectories)
plt.show()

#%%
plt.figure()
tgen.plot_trajectories_magnitudes(trajectories)
plt.show()

#%%
print('number of segments', len(traj.coef_array()))
#%%
plt.figure()
plt.title('durations')
plt.bar(range(len(T)), T)
plt.show()

#%%
tgen.animate_trajectories('p_formation.mp4', trajectories, 1)

#%%
