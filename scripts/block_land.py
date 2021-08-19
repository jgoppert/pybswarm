#%%
import sys
import os
sys.path.insert(0, os.getcwd())
os.chdir('../')

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import bswarm.trajectory as tgen
import bswarm.formation
import bswarm
import json


origin = np.array([0, 1, 2])
waypoints = []

# waypoint 1
waypoints.append(
    np.array([
        # drone 1
        [0, 0, 0]
    ]).T)

# waypoint 2
waypoints.append(
    np.array([
        # drone 1
        [1, 0, 0]
    ]).T)

# waypoint 3
waypoints.append(
    np.array([
        # drone 1
        [2, 0, 0]
    ]).T)

waypoints = np.array(waypoints)

# time to take on each leg, must be length of waypoints - 1
time_vector = [5, 5]

assert len(waypoints) < 33
trajectories = []
for drone in range(waypoints.shape[2]):
    print(waypoints[:, :, drone])
    pos_wp = waypoints[:, :, drone] + origin
    print(pos_wp)
    yaw_wp = np.zeros((pos_wp.shape[0], 1))
    traj = tgen.min_deriv_4d(4, 
        np.hstack([pos_wp, yaw_wp]), time_vector, stop=False)
    trajectories.append(traj)
assert len(trajectories) < 32
data = tgen.trajectories_to_json(trajectories)


with open('scripts/data/block_land.json', 'w') as f:
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
