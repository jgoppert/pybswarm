import numpy as np

def rotate_points_z_h(P, theta):
    R = np.array([
        [np.cos(theta), np.sin(theta), 0, 0],
        [-np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    P_h = np.vstack([P, np.zeros(P.shape[1])]) #homogenous coordinates
    return R.dot(P_h)[:3, :]

def rotate_points_z(P, theta):
    R = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    return R @ P