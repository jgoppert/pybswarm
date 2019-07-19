#%%
import sys
import os
sys.path.insert(0, os.getcwd())

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import bswarm.trajectory as tgen
import bswarm.formation
import bswarm
import json

class Formation:

    def __init__(self, points, order):
        self.points = points
        self.order = order

def plot_formation(F: Formation, name):
    plt.figure()
    ax = plt.axes(projection='3d')
    points = F.points
    for i in range(points.shape[1]):
        ax.text3D(points[0, i], points[1, i], points[2, i], str(i))
        ax.plot3D([points[0, i]], [points[1, i]], [points[2, i]], 'r.')
    plt.title(name)
    plt.show()

def scale_formation(form, scale):
    formNew = np.copy(form)
    for i in range(3):
        formNew[i, :] *= scale[i]
    return formNew

#%% takeoff
formations = {}
formations['takeoff'] = Formation(
    points = np.array([
        [-0.5, -1, 0],
        [-0.5, 0, 0],
        [-0.5, 1, 0],
        [0.5, -1, 0],
        [0.5, 0, 0],
        [0.5, 1, 0],
        ]).T,
    order=[0, 1, 2, 3, 4, 5])
plot_formation(formations['takeoff'], 'takeoff')



#%% P
letter_scale = np.array([1.5, 1.5, 1.5])
formations['P'] = Formation(
    points=scale_formation(np.array([
        [-0.5, -0.5, 1],
        [-0.5, 0, 1],
        [-0.25, 0.5, 0.75],
        [0, -0.5, 0.5],
        [0.5, -0.5, 0],
        [0, 0, 0.5],
    ]).T, letter_scale),
    order=[4, 3, 0, 1, 2, 5])

plot_formation(formations['P'], 'P')

#%% U
formations['U'] = Formation(
    points=scale_formation(np.array([
        [-0.5, -0.5, 1],
        [-0.5, 0.5, 1],
        [0, 0.5, 0.5],
        [0, -0.5, 0.5],
        [0.5, -0.25, 0],
        [0.5, 0.25, 0],
    ]).T, letter_scale),
    order=[0, 3, 4, 5, 2, 1])
plot_formation(formations['U'], 'U')

#%% 5
formations['5'] = Formation(
    points=scale_formation(np.array([
        [-0.5, -0.5, 1],
        [-0.5, 0.5, 1],
        [0, 0.25, 0.5],
        [0, -0.5, 0.5],
        [0.5, -0.5, 0],
        [0.5, 0.25, 0],
    ]).T, letter_scale),
    order=[1, 0, 3, 2, 5, 4])
plot_formation(formations['5'], '5')

#%% O
formations['O'] = Formation(
    points=scale_formation(np.array([
        [-0.5, -0.25, 1],
        [-0.5, 0.25, 1],
        [0, 0.5, 0.5],
        [0, -0.5, 0.5],
        [0.5, -0.25, 0],
        [0.5, 0.25, 0]
    ]).T, letter_scale),
    order=[0, 1, 2, 5, 4, 3])
plot_formation(formations['O'], 'O')

#%% A
formations['A'] = Formation(
    points=scale_formation(np.array([
        [0, -0.5, 0.5],
        [-0.5, 0, 1],
        [0, 0.5, 0.5],
        [0.5, -0.5, 0],
        [0, 0, 0.5],
        [0.5, 0.5, 0]
    ]).T, letter_scale),
    order=[3, 0, 1, 2, 5, 4])
plot_formation(formations['A'], 'A')

#%% L
formations['L'] = Formation(
    points=scale_formation(np.array([
        [-0.5, -0.5, 1],
        [0, -0.5, 0.5],
        [0, 0.5, 0.5],
        [0.5, -0.5, 0],
        [0.5, 0, 0],
        [0.5, 0.5, 0]
    ]).T, letter_scale),
    order=[0, 1, 3, 4, 5, 2])
plot_formation(formations['L'], 'L')

#%% 1
formations['1'] = Formation(
    points=scale_formation(np.array([
        [-0.5, -0.5, 1],
        [-0.5, 0, 1],
        [0, 0, 0.5],
        [0.5, -0.5, 0],
        [0.5, 0, 0],
        [0.5, 0.5, 0],
    ]).T, letter_scale),
    order=[0, 1, 2, 4, 3, 5])
plot_formation(formations['1'], '1')

#%%
class Letters:

    rgb = {
        'black': [0, 0, 0],
        'gold': [255, 100, 15],
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'white': [255, 255, 255]
    }

    def __init__(self):
        self.waypoints = [formations['takeoff'].points]
        self.T = []
        self.delays = []
        self.colors = []

    def add(self, formation_name: str, color: str, duration: float, led_delay: float, angle: float=0):
        formation = formations[formation_name]
        assert led_delay*len(formation.order) < duration
        self.T.append(duration)
        self.waypoints.append(bswarm.formation.rotate_points_z(formation.points, angle))
        self.delays.append((np.array(formation.order)*led_delay).tolist())
        self.colors.append(self.rgb[color])

    def plan_trajectory(self, origin):
        trajectories = []
        waypoints = np.array(self.waypoints)
        for drone in range(waypoints.shape[2]):
            pos_wp = waypoints[:, :, drone] + origin
            yaw_wp = np.zeros((pos_wp.shape[0], 1))
            traj = tgen.min_deriv_4d(4, 
                np.hstack([pos_wp, yaw_wp]), self.T, stop=False)
            trajectories.append(traj)
        return trajectories

def plan_letters(letter_string: str):
    letters = Letters()
    letters.add('takeoff', color='black', duration=2, led_delay=0)
    #%% Create waypoints for flat P -> slanted P -> rotating slanted P -> flat P
    for i, letter in enumerate(letter_string.split(' ')):
        letters.add('takeoff', color='blue', duration=5, led_delay=0)
        letters.add(letter, color='blue', duration=5, led_delay=0)
        letters.add(letter, color='gold', duration=5, led_delay=0.5)
        if i == 7:
            for theta in np.linspace(0, 2*np.pi, 6)[1:]:
                letters.add(letter, color='gold', duration=3, led_delay=0, angle=theta)
    letters.add('takeoff', color='black', duration=5, led_delay=0)

    trajectories = letters.plan_trajectory(origin=np.array([1.5, 2, 2]))
    traj_json = tgen.trajectories_to_json(trajectories)
    data = {}
    for key in traj_json.keys():
        data[key] = {
            'trajectory': traj_json[key],
            'T': letters.T,
            'color': letters.colors,
            'delay': [d[key] for d in letters.delays]
        }
    data['repeat'] = 1
    return trajectories, data

trajectories, data = plan_letters('P U A P O L L O')

with open('scripts/data/p_form.json', 'w') as f:
    json.dump(data, f)

tgen.plot_trajectories(trajectories)
plt.show()


#%%
plt.figure()
tgen.plot_trajectories_time_history(trajectories)
plt.show()

#%%
plt.figure()
tgen.plot_trajectories_magnitudes(trajectories)
plt.show()

#%%
print('number of segments', len(trajectories[0].coef_array()))
#%%
plt.figure()
plt.title('durations')
plt.bar(range(len(data[0]['T'])), data[0]['T'])
plt.show()

#%%

#tgen.animate_trajectories('p_formation.mp4', trajectories, fps=5)


#%%
