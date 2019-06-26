#%%
# add dev path for non-interactive script call
import os
import sys
sys.path.insert(0, os.getcwd())

import bswarm
import bswarm.trajectory_generation as tgen
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#%%
n_drone = 7
pi = np.pi

#read wayponts
#init pos
P0 = np.array([
    [-1, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [-1, 0, 1],
    [0, 0, 1],
    [1, 0, 1],
    [-1, -1, 1]
]).T

ax = plt.axes(projection='3d')
ax.plot3D(P0[0,:], P0[1,:], P0[2,:], 'ro')
#%%
#form P:
P1 = np.array([
    [-1, 1, 2.5],
    [0, 1, 2.5],
    [1, 1, 2.5],
    [-1, 0, 1.75],
    [0, 0, 1.75],
    [1, 0, 1.75],
    [-1, -1, 1]
]).T
ax = plt.axes(projection='3d')
ax.plot3D(P0[0,:], P0[1,:], P0[2,:], 'ro')
ax.plot3D(P1[0,:], P1[1,:], P1[2,:], 'bo')
#%%
#form U:
P2 = np.array([
    [-1, 1, 2.5],
    [1, 0, 1.75],
    [1, 1, 2.5],
    [-1, 0, 1.75],
    [0, -1, 1],
    [1, -1, 1],
    [-1, -1, 1]
]).T
ax = plt.axes(projection='3d')
ax.plot3D(P1[0,:], P1[1,:], P1[2,:], 'ro')
ax.plot3D(P2[0,:], P2[1,:], P2[2,:], 'wx')

#%%
#initialize waypoints
waypoints = [P0]
waypoints.append(P1)
waypoints.append(P1)
waypoints.append(P2)
waypoints.append(P2)
#waypoints.append(P0)
waypoints = np.array(waypoints)
print(waypoints.shape)
#%%
T = [1]*(waypoints.shape[0]-1)
print(T)
p_list = []
trajxs = []
formation={}
for drone in range(n_drone):
    print('drone: ',drone)
    planner = tgen.plan_min_snap
    trajx = planner(waypoints[:,0,drone],T)
    trajy = planner(waypoints[:,1,drone],T)
    trajz = planner(waypoints[:,2,drone],T)
    cx = trajx.c
    cy = trajy.c
    cz = trajz.c
    cyaw = [0]*8
    traj=[]
    for leg in range(len(T)):
        cxi = np.flip(cx[8*leg:8*(leg+1)],0)
        cyi = np.flip(cy[8*leg:8*(leg+1)],0)
        czi = np.flip(cz[8*leg:8*(leg+1)],0)
        traj.append(np.hstack([T[leg],cxi,cyi,czi,cyaw]))
    formation[drone] = np.array(traj).tolist()
    p = np.array(
        [trajx.compute_trajectory()['x'],
        trajy.compute_trajectory()['x'],
        trajz.compute_trajectory()['x']
    ])
    p_list.append(p)
    trajxs.append(trajx)
plt.figure()
plt.grid()
ax = plt.axes(projection='3d')
for p in p_list:
    ax.plot3D(p[0,:],p[1,:],p[2,:])
ax.plot3D(P0[0,:],P0[1,:],P0[2,:],'ro')
ax.plot3D(P1[0,:],P1[1,:],P1[2,:],'bo')
ax.plot3D(P2[0,:],P2[1,:],P2[2,:],'wx')

plt.figure()
for trajx in trajxs:
    trajx.plot()
#%%
import json
path = '/home/zp/catkin_ws/src/turtlesim_cleaner/src/json/'
with open('scripts/json/p2u.json', 'w') as f:
    json.dump(formation, f)

#%%

plt.show()

#%%
