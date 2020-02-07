import  bswarm
import bswarm.trajectory as traj
import bswarm.formation
import numpy as np
import matplotlib.pyplot as plt

def test_load():
    bswarm.load_file()
    assert True


def test_trajectory():
    res = traj.min_snap_1d([1, 2, 3], [1, 2], False)
    plt.figure(figsize=(6, 5))
    traj.plot_trajectories_time_history([res])
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def test_rotation():
    P = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [6, 3, 1]
    ]).T
    P_rot = bswarm.formation.rotate_points_z(P, 2*np.pi)
    assert np.allclose(P, P_rot)

def test_trajectory_2d():
    # continous
    T = [1, 2]
    waypoints = np.array([
        [0, 0],
        [1, 2],
        [3, 4]
    ])
    trajx = traj.min_accel_1d(waypoints[:, 0], T, False)
    trajy = traj.min_accel_1d(waypoints[:, 1], T, False)
    print('x coef', trajx.coef_array())
    print('y coef', trajy.coef_array())
