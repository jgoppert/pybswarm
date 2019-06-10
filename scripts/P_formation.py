#%%
import bswarm
import bswarm.formation as form
import bswarm.trajectory_generation as tgen

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

%matplotlib inline
#%%

P = np.array([
    [-1, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [-1, 0, 1],
    [0, 0, 1],
    [1, 0, 1],
    [-1, -1, 1]
]).T
ax = plt.axes(projection='3d')
ax.plot3D(P[0,:], P[1,:], P[2,:], 'ro')

#%%

P2 = np.array([
    [-1, 1, 2],
    [0, 1, 2],
    [1, 1, 2],
    [-1, 0, 1.5],
    [0, 0, 1.5],
    [1, 0, 1.5],
    [-1, -1, 1]
]).T
ax = plt.axes(projection='3d')
ax.plot3D(P[0,:], P[1,:], P[2,:], 'ro')
ax.plot3D(P2[0,:], P2[1,:], P2[2,:], 'bo')

#%%

waypoints = [P]
for theta in np.linspace(0, 2*np.pi, 8):
    waypoints.append(form.rotate_points_z(P2, theta))
waypoints.append(P)
waypoints = np.array(waypoints)

ax = plt.axes(projection='3d')
for point in range(waypoints.shape[2]):
    ax.plot3D(waypoints[:,0,point], waypoints[:,1,point], waypoints[:,2,point], '-')
    ax.view_init(azim=0, elev=40)

#%%

dist = np.linalg.norm(waypoints[1:, :, :] - waypoints[:-1, :, :], axis=1)
dist_max = np.max(dist, axis=1)
dist_max

ax = plt.axes(projection='3d')
p_list = []
trajx_list = []
T = 10*np.ones(dist.shape[0])
print('T', T)

for drone in range(waypoints.shape[2]):
    print('drone', drone)
    planner = tgen.plan_min_snap
    trajx = planner(waypoints[:, 0, drone], T)
    trajy = planner(waypoints[:, 1, drone], T)
    trajz = planner(waypoints[:, 2, drone], T)
    p = np.array([trajx.compute_trajectory()['x'],
        trajy.compute_trajectory()['x'],
        trajz.compute_trajectory()['x']])
    p_list.append(p)
    trajx_list.append(trajx)

plt.figure()
for p in p_list:
    ax.plot3D(p[0,:], p[1,:], p[2,:])

plt.figure()
for trajx in trajx_list:
    trajx.plot()
T#%%


#%%


#%%


#%%
