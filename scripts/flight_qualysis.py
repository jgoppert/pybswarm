import time
import signal
import sys
import json
import argparse
import numpy as np
from typing import Dict, List

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie

import asyncio
import math
import xml.etree.cElementTree as ET
from threading import Thread

import qtm

if sys.version_info[0] != 3:
    print("This script requires Python 3")
    exit()

# Change uris and sequences according to your setup
DRONE0 = 'radio://0/81/2M/E7E7E7E770'
DRONE1 = 'radio://0/81/2M/E7E7E7E771'

trajectory_assignment = {
    0: DRONE0,
    1: DRONE1,
}

n_drones = len(trajectory_assignment)

body_names = []
ids = [[1,3,4,2]]
for i in range(n_drones):
    body_names.append('cf'+str(i))
    if i != 0:
        ids.append([i*4 + j for j in ids[0]])

id_assignment = {trajectory_assignment[i]: ids[i] for i in range(n_drones)}
rigid_bodies = {trajectory_assignment[i]: body_names[i] for i in range(n_drones)}

# rigid_bodies = {
#     DRONE0: 'cf0',
#     DRONE1: 'cf1',
# }

figure8 = [
    [1.050000, 0.000000, -0.000000, 0.000000, -0.000000, 0.830443, -0.276140, -0.384219, 0.180493, -0.000000, 0.000000, -0.000000, 0.000000, -1.356107, 0.688430, 0.587426, -0.329106, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.710000, 0.396058, 0.918033, 0.128965, -0.773546, 0.339704, 0.034310, -0.026417, -0.030049, -0.445604, -0.684403, 0.888433, 1.493630, -1.361618, -0.139316, 0.158875, 0.095799, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.620000, 0.922409, 0.405715, -0.582968, -0.092188, -0.114670, 0.101046, 0.075834, -0.037926, -0.291165, 0.967514, 0.421451, -1.086348, 0.545211, 0.030109, -0.050046, -0.068177, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.700000, 0.923174, -0.431533, -0.682975, 0.177173, 0.319468, -0.043852, -0.111269, 0.023166, 0.289869, 0.724722, -0.512011, -0.209623, -0.218710, 0.108797, 0.128756, -0.055461, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.560000, 0.405364, -0.834716, 0.158939, 0.288175, -0.373738, -0.054995, 0.036090, 0.078627, 0.450742, -0.385534, -0.954089, 0.128288, 0.442620, 0.055630, -0.060142, -0.076163, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.560000, 0.001062, -0.646270, -0.012560, -0.324065, 0.125327, 0.119738, 0.034567, -0.063130, 0.001593, -1.031457, 0.015159, 0.820816, -0.152665, -0.130729, -0.045679, 0.080444, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.700000, -0.402804, -0.820508, -0.132914, 0.236278, 0.235164, -0.053551, -0.088687, 0.031253, -0.449354, -0.411507, 0.902946, 0.185335, -0.239125, -0.041696, 0.016857, 0.016709, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.620000, -0.921641, -0.464596, 0.661875, 0.286582, -0.228921, -0.051987, 0.004669, 0.038463, -0.292459, 0.777682, 0.565788, -0.432472, -0.060568, -0.082048, -0.009439, 0.041158, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [0.710000, -0.923935, 0.447832, 0.627381, -0.259808, -0.042325, -0.032258, 0.001420, 0.005294, 0.288570, 0.873350, -0.515586, -0.730207, -0.026023, 0.288755, 0.215678, -0.148061, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
    [1.053185, -0.398611, 0.850510, -0.144007, -0.485368, -0.079781, 0.176330, 0.234482, -0.153567, 0.447039, -0.532729, -0.855023, 0.878509, 0.775168, -0.391051, -0.713519, 0.391628, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
]

class QtmWrapper(Thread):
    def __init__(self, body_names):
        Thread.__init__(self)

        self.body_names = body_names
        # self.on_pose = None
        self.on_pose = {name: None for name in body_names}
        self.connection = None
        self.qtm_6DoF_labels = []
        self._stay_open = True

        self.start()

    def close(self):
        self._stay_open = False
        self.join()

    def run(self):
        asyncio.run(self._life_cycle())

    async def _life_cycle(self):
        await self._connect()
        while(self._stay_open):
            await asyncio.sleep(1)
        await self._close()

    async def _connect(self):
        # qtm_instance = await self._discover()
        # host = qtm_instance.host
        host = "192.168.1.16"
        print('Connecting to QTM on ' + host)
        self.connection = await qtm.connect(host=host, version="1.20")

        print(type(self.connection))
        params = await self.connection.get_parameters(parameters=['6d'])
        xml = ET.fromstring(params)
        self.qtm_6DoF_labels = [label.text for label in xml.iter('Name')]
        print(self.qtm_6DoF_labels)

        await self.connection.stream_frames(
            components=['6D'],
            on_packet=self._on_packet)

    async def _discover(self):
        async for qtm_instance in qtm.Discover('0.0.0.0'):
            return qtm_instance

    def _on_packet(self, packet):
        header, bodies = packet.get_6d()

        if bodies is None:
            return

        # if self.body_name not in self.qtm_6DoF_labels:
            # print('Body ' + self.body_name + ' not found.')
        if len(set(self.qtm_6DoF_labels).intersection(body_names)) < n_drones :
            print('Missing rigid bodies')
            
        else:
            for body_name in self.body_names:
                index = self.qtm_6DoF_labels.index(body_name)
                temp_cf_pos = bodies[index]
                x = temp_cf_pos[0][0] / 1000
                y = temp_cf_pos[0][1] / 1000
                z = temp_cf_pos[0][2] / 1000

                r = temp_cf_pos[1].matrix
                rot = [
                    [r[0], r[3], r[6]],
                    [r[1], r[4], r[7]],
                    [r[2], r[5], r[8]],
                ]

                if self.on_pose:
                    # Make sure we got a position
                    if math.isnan(x):
                        return

                    self.on_pose[body_name]([x, y, z, rot])


    async def _close(self):
        await self.connection.stream_frames_stop()
        self.connection.disconnect()

def _sqrt(a):
    """
    There might be rounding errors making 'a' slightly negative.
    Make sure we don't throw an exception.
    """
    if a < 0.0:
        return 0.0
    return math.sqrt(a)

def send_extpose_rot_matrix(scf: SyncCrazyflie, x, y, z, rot):
    """
    Send the current Crazyflie X, Y, Z position and attitude as a (3x3)
    rotaton matrix. This is going to be forwarded to the Crazyflie's
    position estimator.
    """
    cf = scf.cf

    qw = _sqrt(1 + rot[0][0] + rot[1][1] + rot[2][2]) / 2
    qx = _sqrt(1 + rot[0][0] - rot[1][1] - rot[2][2]) / 2
    qy = _sqrt(1 - rot[0][0] + rot[1][1] - rot[2][2]) / 2
    qz = _sqrt(1 - rot[0][0] - rot[1][1] + rot[2][2]) / 2

    # Normalize the quaternion
    ql = math.sqrt(qx ** 2 + qy ** 2 + qz ** 2 + qw ** 2)

    cf.extpos.send_extpose(x, y, z, qx / ql, qy / ql, qz / ql, qw / ql)

def send_extpos(scf: SyncCrazyflie, x, y, z):
    """
    Send only position data to the Crazyflie, same as cfclient and qualysis python resource
    """
    cf = scf.cf
    cf.extpos.send_extpos(x, y, z)

class Uploader:
    def __init__(self, trajectory_mem):
        self._is_done = False
        self.trajectory_mem = trajectory_mem

    def upload(self):
        print('upload started')
        self.trajectory_mem.write_data(self._upload_done)
        while not self._is_done:
            print('uploading...')
            time.sleep(1)

    def _upload_done(self, mem, addr):
        print('upload is done')
        self._is_done = True
        self.trajectory_mem.disconnect()


def check_battery(scf: SyncCrazyflie, min_voltage=3.7):
    print('Checking battery...')
    log_config = LogConfig(name='Battery', period_in_ms=500)
    log_config.add_variable('pm.vbat', 'float')

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            log_data = log_entry[1]
            vbat = log_data['pm.vbat']
            if log_data['pm.vbat'] < min_voltage:
                msg = "battery too low: {:10.4f} V, for {:s}".format(
                    vbat, scf.cf.link_uri)
                raise Exception(msg)
            else:
                return


def check_state(scf: SyncCrazyflie, min_voltage=4.0):
    print('Checking state.')
    log_config = LogConfig(name='State', period_in_ms=500)
    log_config.add_variable('stabilizer.roll', 'float')
    log_config.add_variable('stabilizer.pitch', 'float')
    log_config.add_variable('stabilizer.yaw', 'float')
    print('Log configured.')

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            log_data = log_entry[1]
            roll = log_data['stabilizer.roll']
            pitch = log_data['stabilizer.pitch']
            yaw = log_data['stabilizer.yaw']
            print('Checking roll/pitch/yaw.')

            for name, val in [('roll', roll), ('pitch', pitch), ('yaw', yaw)]:

                if np.abs(val) > 20:
                    print('exceeded')
                    msg = "too much {:s}, {:10.4f} deg, for {:s}".format(
                        name, val, scf.cf.link_uri)
                    print(msg)
                    raise Exception(msg)
            return


def wait_for_position_estimator(scf: SyncCrazyflie):
    print('Waiting for estimator to find position...',)

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf.cf, log_config) as logger:
        for log_entry in logger:
            log_data = log_entry[1]

            var_x_history.append(log_data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(log_data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(log_data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break
            else:
                print("{:s}\t{:10g}\t{:10g}\t{:10g}".
                      format(scf.cf.link_uri, max_x - min_x, max_y - min_y, max_z - min_z))


def wait_for_param_download(scf: SyncCrazyflie):
    while not scf.cf.param.is_updated:
        time.sleep(1.0)
    print('Parameters downloaded for', scf.cf.link_uri)


def reset_estimator(scf: SyncCrazyflie):
    print('Resetting estimator...')
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    wait_for_position_estimator(scf)

def activate_kalman_estimator(cf):
    cf.param.set_value('stabilizer.estimator', '2')

    # Set the std deviation for the quaternion data pushed into the
    # kalman filter. The default value seems to be a bit too low.
    cf.param.set_value('locSrv.extQuatStdDev', 0.06)

# def upload_trajectory(scf: SyncCrazyflie, data: Dict):
def upload_trajectory(scf: SyncCrazyflie):
    try:
        cf = scf.cf  # type: Crazyflie

        print('Starting upload')
        trajectory_mem = scf.cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]

        TRAJECTORY_MAX_LENGTH = 31
        # trajectory = data['trajectory']
        trajectory =  figure8

        if len(trajectory) > TRAJECTORY_MAX_LENGTH:
            raise ValueError("Trajectory too long for drone {:s}".format(cf.link_uri))

        for row in trajectory:
            duration = row[0]
            x = Poly4D.Poly(row[1:9])
            y = Poly4D.Poly(row[9:17])
            z = Poly4D.Poly(row[17:25])
            yaw = Poly4D.Poly(row[25:33])
            trajectory_mem.poly4Ds.append(Poly4D(duration, x, y, z, yaw))

        print('Calling upload method')
        uploader = Uploader(trajectory_mem)
        uploader.upload()
        
        print('Defining trajectory.')
        cf.high_level_commander.define_trajectory(
            trajectory_id=1, offset=0, n_pieces=len(trajectory_mem.poly4Ds))

    except Exception as e:
        print(e)
        land_sequence(scf)
        raise(e)

def preflight_sequence(scf: Crazyflie):
    """
    This is the preflight sequence. It calls all other subroutines before takeoff.
    """
    cf = scf.cf  # type: Crazyflie

    try:
        # switch to Kalman filter
        cf.param.set_value('stabilizer.estimator', '2')
        cf.param.set_value('locSrv.extQuatStdDev', 0.06)

        # enable high level commander
        cf.param.set_value('commander.enHighLevel', '1')

        # ensure params are downloaded
        wait_for_param_download(scf)

        # make sure not already flying
        # land_sequence(scf)

        # set pid gains, tune down Kp to deal with UWB noise
        # cf.param.set_value('posCtlPid.xKp', '1')
        # cf.param.set_value('posCtlPid.yKp', '1')
        # cf.param.set_value('posCtlPid.zKp', '1')

        # check battery level
        check_battery(scf, 3.7)

        # reset the estimator
        reset_estimator(scf)

        # check state
        check_state(scf)

    except Exception as e:
        print(e)
        land_sequence(scf)
        raise(e)


def preflight_sequence_waypoint(scf: Crazyflie):
    """
    This is the preflight sequence. It calls all other subroutines before takeoff.
    """
    cf = scf.cf  # type: Crazyflie

    try:
        # switch to Kalman filter
        cf.param.set_value('stabilizer.estimator', '2')
        cf.param.set_value('locSrv.extQuatStdDev', 0.06)

        # enable high level commander
        cf.param.set_value('commander.enHighLevel', '0')

        # ensure params are downloaded
        wait_for_param_download(scf)

        # make sure not already flying
        # land_sequence(scf)

        # set pid gains, tune down Kp to deal with UWB noise
        # cf.param.set_value('posCtlPid.xKp', '1')
        # cf.param.set_value('posCtlPid.yKp', '1')
        # cf.param.set_value('posCtlPid.zKp', '1')

        # check battery level
        check_battery(scf, 3.7)

        # reset the estimator
        reset_estimator(scf)

        # check state
        check_state(scf)

    except Exception as e:
        print(e)
        land_sequence(scf)
        raise(e)


def takeoff_sequence(scf: Crazyflie):
    """
    This is the takeoff sequence. It commands takeoff.
    """
    try:
        cf = scf.cf  # type: Crazyflie
        commander = cf.high_level_commander  # type: cflib.HighLevelCOmmander
        cf.param.set_value('commander.enHighLevel', '1')
        commander.takeoff(1.0, 3.0)
        time.sleep(10.0)
    except Exception as e:
        print(e)
        land_sequence(scf)

# def go_sequence(scf: Crazyflie, data: Dict):
def go_sequence(scf: Crazyflie):
    """
    This is the go sequence. It commands the trajectory to start.
    """
    try:
        cf = scf.cf  # type: Crazyflie
        commander = cf.high_level_commander  # type: cflib.HighLevelCOmmander
        commander.start_trajectory(
            trajectory_id=1, time_scale=1.0, relative=True)
        time.sleep(10.0)
    except Exception as e:
        print(e)
        land_sequence(scf)

def land_sequence(scf: Crazyflie):
    try:
        cf = scf.cf  # type: Crazyflie
        commander = cf.high_level_commander  # type: cflib.HighLevelCOmmander
        commander.land(0.0, 3.0)
        print('Landing...')
        time.sleep(3)
        commander.stop()
    except Exception as e:
        print(e)
        land_sequence(scf)

def send6DOF(scf: SyncCrazyflie, qtmWrapper: QtmWrapper, name):
    """
    This relays mocap data to each crazyflie
    """
    try:
        qtmWrapper.on_pose[name] = lambda pose: send_extpose_rot_matrix(scf, pose[0], pose[1], pose[2], pose[3])
    except Exception as e:
        print(e)
        print('Could not relay mocap data')
        land_sequence(scf)

def id_update(scf: SyncCrazyflie, id: List):
    cf = scf.cf
    cf.param.set_value('activeMarker.mode', '3') # qualisys mode
    time.sleep(1)

    # default id
    # [f, b, l, r] 
    # [1, 3, 4, 2]

    cf.param.set_value('activeMarker.front', str(id[0]))
    time.sleep(1)
    cf.param.set_value('activeMarker.back', str(id[1]))
    time.sleep(1)
    cf.param.set_value('activeMarker.left', str(id[2]))
    time.sleep(1)
    cf.param.set_value('activeMarker.right', str(id[3]))
    time.sleep(1)
    
    print('done!')

def swarm_id_update():
    cflib.crtp.init_drivers(enable_debug_driver=False)

    factory = CachedCfFactory(rw_cache='./cache')
    uris = {trajectory_assignment[key] for key in trajectory_assignment.keys()}
    id_args = {key: [id_assignment[key]] for key in id_assignment.keys()}
    with Swarm(uris, factory=factory) as swarm:
        print('Starting ID update')
        swarm.parallel_safe(id_update, id_args)

def go_waypoints(scf: SyncCrazyflie, waypoints: List):
    cf = scf.cf
    for position in waypoints:
        print('Setting position {}'.format(position))
        for i in range(20):
            cf.commander.send_position_setpoint(position[0],
                                                position[1],
                                                position[2],
                                                position[3])
            time.sleep(0.1)

    cf.commander.send_stop_setpoint()
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)


def run():
# def run(args):
    cflib.crtp.init_drivers(enable_debug_driver=False)

    factory = CachedCfFactory(rw_cache='./cache')
    uris = {trajectory_assignment[key] for key in trajectory_assignment.keys()}
    
    # with open(args.json, 'r') as f:
    #     data = json.load(f)
    # swarm_args = {trajectory_assignment[drone_pos]: [data[str(drone_pos)]]
    #     for drone_pos in trajectory_assignment.keys()}

    qtmWrapper = QtmWrapper(body_names)
    qtm_args = {key: [qtmWrapper, rigid_bodies[key]] for key in rigid_bodies.keys()}

    with Swarm(uris, factory=factory) as swarm:
        
        def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            swarm.parallel(land_sequence)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        print('Press Ctrl+C to land.')
    
        print('Starting mocap data relay...')
        swarm.parallel_safe(send6DOF, qtm_args)

        print('Preflight sequence...')
        swarm.parallel_safe(preflight_sequence)

        print('Takeoff sequence...')
        swarm.parallel_safe(takeoff_sequence)

        print('Upload sequence...')
        trajectory_count = 0

        # repeat = int(data['repeat'])

        for trajectory_count in range(1):
            if trajectory_count == 0:
                print('Uploading Trajectory')
                # swarm.parallel(upload_trajectory, args_dict=swarm_args)
                swarm.parallel(upload_trajectory)

            print('Go...')
            # swarm.parallel_safe(go_sequence, args_dict=swarm_args)
            swarm.parallel_safe(go_sequence)
        print('Land sequence...')
        swarm.parallel(land_sequence)

    print('Closing QTM connection...')


waypoints0 = [
    [0.5, 0.5, 1.0, 0],
    [1.5, 0.5, 1.0, 0],
    [1.5, 1.5, 1.0, 0],
    [0.5, 1.5, 1.0, 0],
    [0.5, 0.5, 1.0, 0],
    [0.5, 0.5, 0.4, 0],
]

waypoints1 = [
    [1.5, 0.5, 1.0, 0],
    [1.5, 1.5, 1.0, 0],
    [0.5, 1.5, 1.0, 0],
    [0.5, 0.5, 1.0, 0],
    [1.5, 0.5, 1.0, 0],
    [1.5, 0.5, 0.4, 0],
]

swarm_waypoints = {
    DRONE0: [waypoints0],
    DRONE1: [waypoints1],
}

def run_waypoints():
    cflib.crtp.init_drivers(enable_debug_driver=False)

    factory = CachedCfFactory(rw_cache='./cache')
    uris = {trajectory_assignment[key] for key in trajectory_assignment.keys()}
    # qtm_args = {key: [QtmWrapper(rigid_bodies[key])] for key in rigid_bodies.keys()} # args_dict for send6DOF
    qtmWrapper = QtmWrapper(body_names)
    qtm_args = {key: [qtmWrapper, rigid_bodies[key]] for key in rigid_bodies.keys()}

    with Swarm(uris, factory=factory) as swarm:
        def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            swarm.parallel(land_sequence)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        print('Press Ctrl+C to land.')
    
        print('Starting mocap data relay...')
        swarm.parallel_safe(send6DOF, qtm_args)

        print('Preflight sequence...')
        swarm.parallel_safe(preflight_sequence_waypoint)

        print('Takeoff sequence...')
        # swarm.parallel_safe(takeoff_sequence)

        swarm.parallel_safe(go_waypoints, swarm_waypoints)
        print('Land sequence...')
        swarm.parallel(land_sequence)

        print('Closing QTM connection...')

    qtmWrapper.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')

    # parser.add_argument('--json')
    args = parser.parse_args()
    # run(args)

    if args.mode == 'id':
        swarm_id_update()
    elif args.mode == 'traj':
        run()
    elif args.mode == 'waypoint':
        run_waypoints()
    else:
        print('Not a valid input!!!')
