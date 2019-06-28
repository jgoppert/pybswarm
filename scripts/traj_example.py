#%%
import sys
import os
sys.path.insert(0, os.getcwd())
import bswarm.trajectory as tgen
import numpy as np
import matplotlib.pyplot as plt

waypoints = np.array([
    # x, y, z, yaw
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 1, 1, 0],
    [1, 0, 1, 0],
    [0, 0, 1, 0]
])
T = 2*np.ones(len(waypoints) - 1)
traj = tgen.min_snap_4d(waypoints, T)
res = traj.compute_inputs()
print(res.max_data())

#%%
plt.figure()
tgen.plot_trajectory_3d(traj)
plt.show()

#%%
plt.figure()
tgen.plot_trajectory_derivatives(traj)
plt.show()

#%%
check_with_other_library = True
if check_with_other_library:
    try:
        import bswarm.third_party.plot_trajectory as other
        traj.write_csv('/tmp/data.csv')
        other.plot_uav_trajectory('/tmp/data.csv')
        plt.show()
    except ImportError:
        print('requires plot_uav_trajectory module')

#%%
