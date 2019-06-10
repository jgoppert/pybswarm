# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package
import  bswarm
import bswarm.trajectory_generation as traj
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