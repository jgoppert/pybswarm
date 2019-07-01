import  bswarm
import bswarm.trajectory as traj
import bswarm.formation
import numpy as np

def test_load():
    bswarm.load_file()
    assert True


def test_trajectory():
    res = traj.plan_min_snap([1, 2, 3], [1, 2])
    res.plot()


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
    trajx = traj.min_accel_1d(waypoints[:, 0], T)
    trajy = traj.min_accel_1d(waypoints[:, 1], T)
    print('x coef', trajx.coef_array())
    print('y coef', trajy.coef_array())