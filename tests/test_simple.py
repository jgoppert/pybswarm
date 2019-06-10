# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package
import  bswarm
import bswarm.trajectory_generation as traj
import numpy as np

def test_load():
    bswarm.load_file()
    assert True


def test_trajectory():
    res = traj.poly3_pos_vel(x_list=[1, 1, 1], v_list=[1, 1, 1], T=[1, 1])
    plt.plot(res['t'], res['x'])