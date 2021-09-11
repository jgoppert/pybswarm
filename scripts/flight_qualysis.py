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
DRONE0 = 'radio://0/80/250K/E7E7E7E7E0'
DRONE1 = 'radio://0/80/250K/E7E7E7E7E1'
DRONE2 = 'radio://0/80/250K/E7E7E7E7E2'

DRONE3 = 'radio://0/90/250K/E7E7E7E7E3'
DRONE4 = 'radio://0/90/250K/E7E7E7E7E4'
DRONE5 = 'radio://0/90/250K/E7E7E7E7E5'

# DRONE6 = 'radio://0/91/250K/E7E7E7E776'
# DRONE7 = 'radio://0/91/250K/E7E7E7E777'
# DRONE8 = 'radio://0/91/250K/E7E7E7E778'

SEND_FULL_POSE = True


def assignments(count):
    one_drone = [DRONE0]
    three_drone = [DRONE0, DRONE1, DRONE2]
    six_drone = [DRONE0, DRONE1, DRONE2, DRONE3, DRONE4, DRONE5]
    if count == 1:
        uris = one_drone
    elif count == 3:
        uris = three_drone
    elif count == 6:
        uris = six_drone
    else:
        raise ValueError('unknown number of drones')
    n_drones = len(uris)

    body_names = []
    ids = [[1,3,4,2]]
    for i in range(n_drones):
        body_names.append('cf'+str(i))
        if i != 0:
            ids.append([i*4 + j for j in ids[0]])

    print('Defined bodies are: ', body_names)
    print('Deck ids are:', ids)

    trajectory_assignment = {i: uris[i] for i in range(n_drones)}
    print('Trajectory assignment is: ', trajectory_assignment)
    id_assignment = {trajectory_assignment[i]: ids[i] for i in range(n_drones)}
    rigid_bodies = {trajectory_assignment[i]: body_names[i] for i in range(n_drones)}
    return {
        'id_assignment': id_assignment,
        'rigid_bodies': rigid_bodies,
        'trajectory_assignment': trajectory_assignment,
        'n_drones': n_drones,
        'body_names': body_names,
    }


class UnstableException(Exception):
    pass


class LowBatteryException(Exception):
    pass


class QtmWrapper(Thread):
    def __init__(self, body_names):
        Thread.__init__(self)

        self.body_names = body_names
        self.on_pose = {name: None for name in body_names}
        self.connection = None
        self.qtm_6DoF_labels = []
        self._stay_open = True

        self.last_send = time.time()
        self.dt_min = 0.2 # reducing send_extpose rate to 5HZ

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
        host = "192.168.1.2"
        print('Connecting to QTM on ' + host)
        self.connection = await qtm.connect(host=host, version="1.20") # version 1.21 has weird 6DOF labels, so using 1.20 here

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
        now = time.time()
        dt = now - self.last_send
        if dt < self.dt_min:
            return
        self.last_send = time.time()
        # print('Hz: ', 1.0/dt)
        
        header, bodies = packet.get_6d()

        if bodies is None:
            return

        intersect = set(self.qtm_6DoF_labels).intersection(self.body_names) 
        if len(intersect) < len(self.body_names) :
            print('Missing rigid bodies')
            print('In QTM: ', self.qtm_6DoF_labels)
            print('Intersection: ', intersect)
            return            
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

                if self.on_pose[body_name]:
                    # Make sure we got a position
                    if math.isnan(x):
                        print("======= LOST RB TRACKING : " + body_name)
                        continue

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
    By default the attitude is not sent.
    """
    cf = scf.cf

    if SEND_FULL_POSE:
        trace = rot[0][0] + rot[1][1] + rot[2][2]
        if trace > 0:
            a = _sqrt(1 + trace)
            qw = 0.5*a
            b = 0.5/a
            qx = (rot[2][1] - rot[1][2])*b
            qy = (rot[0][2] - rot[2][0])*b
            qz = (rot[1][0] - rot[0][1])*b
        elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
            a = _sqrt(1 + rot[0][0] - rot[1][1] - rot[2][2])
            qx = 0.5*a
            b = 0.5/a
            qw = (rot[2][1] - rot[1][2])*b
            qy = (rot[1][0] + rot[0][1])*b
            qz = (rot[0][2] + rot[2][0])*b
        elif rot[1][1] > rot[2][2]:
            a = _sqrt(1 - rot[0][0] + rot[1][1] - rot[2][2])
            qy = 0.5*a
            b = 0.5/a
            qw = (rot[0][2] - rot[2][0])*b
            qx = (rot[1][0] + rot[0][1])*b
            qz = (rot[2][1] + rot[1][2])*b
        else:
            a = _sqrt(1 - rot[0][0] - rot[1][1] + rot[2][2])
            qz = 0.5*a
            b = 0.5/a
            qw = (rot[1][0] - rot[0][1])*b
            qx = (rot[0][2] + rot[2][0])*b
            qy = (rot[2][1] + rot[1][2])*b
        cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)
    else:
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


def check_battery(scf: SyncCrazyflie, min_voltage=4):
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
                raise LowBatteryException(msg)
            else:
                return


def check_state(scf: SyncCrazyflie):
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

            #('yaw', yaw)
            if SEND_FULL_POSE:
                euler_checks = [('roll', roll, 5), ('pitch', pitch, 5)]
            else:
                euler_checks = [('roll', roll, 5), ('pitch', pitch, 5), ('yaw', yaw, 5)]

            for name, val, val_max in euler_checks:
                print(name, val, val_max)
                if np.abs(val) > val_max:
                    print('exceeded')
                    msg = "too much {:s}, {:10.4f} deg, for {:s}".format(
                        name, val, scf.cf.link_uri)
                    print(msg)
                    raise UnstableException(msg)
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

def sleep_while_checking_stable(scf: SyncCrazyflie, tf_sec, dt_sec=0.1):
    if tf_sec == 0:
        return
    log_config = LogConfig(name='Roll', period_in_ms=int(dt_sec*1000.0))
    log_config.add_variable('stabilizer.roll', 'float')
    log_config.add_variable('pm.vbat', 'float')
    t_sec = 0
    print('sleeping {:10.4f} seconds'.format(tf_sec))
    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            log_data = log_entry[1]
            roll = log_data['stabilizer.roll']
            batt = log_data['pm.vbat']
            if np.abs(roll) > 60:
                raise UnstableException("flip detected {:10.4f} deg, for {:s}".format(roll, scf.cf.link_uri))
            # print("battery {:10.4f} V, for {:s}".format(batt, scf.cf.link_uri))
            if batt < 3.0:
                raise LowBatteryException("low battery {:10.4f} V, for {:s}".format(batt, scf.cf.link_uri))
            t_sec += dt_sec
            if t_sec>tf_sec:
                return


def upload_trajectory(scf: SyncCrazyflie, data: Dict):
    cf = scf.cf  # type: Crazyflie

    print('Starting upload')
    trajectory_mem = scf.cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]

    TRAJECTORY_MAX_LENGTH = 31
    trajectory = data['trajectory']

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

def preflight_sequence(scf: SyncCrazyflie):
    """
    This is the preflight sequence. It calls all other subroutines before takeoff.
    """
    cf = scf.cf  # type: Crazyflie

    # switch to Kalman filter
    cf.param.set_value('stabilizer.estimator', '2')
    cf.param.set_value('locSrv.extQuatStdDev', 0.06)

    # enable high level commander
    cf.param.set_value('commander.enHighLevel', '1')

    # prepare for motor shut-off
    cf.param.set_value('motorPowerSet.enable', '0')
    cf.param.set_value('motorPowerSet.m1', '0')
    cf.param.set_value('motorPowerSet.m2', '0')
    cf.param.set_value('motorPowerSet.m3', '0')
    cf.param.set_value('motorPowerSet.m4', '0')

    # ensure params are downloaded
    wait_for_param_download(scf)

    # make sure not already flying
    # land_sequence(scf)

    cf.param.set_value('ring.effect', '0')

    # set pid gains, tune down Kp to smooth trajectories
    cf.param.set_value('posCtlPid.xKp', '1')
    cf.param.set_value('posCtlPid.yKp', '1')
    cf.param.set_value('posCtlPid.zKp', '1')

    # check battery level
    check_battery(scf, 3.7)

    # reset the estimator
    reset_estimator(scf)

    # check state
    check_state(scf)

def takeoff_sequence(scf: SyncCrazyflie):
    """
    This is the takeoff sequence. It commands takeoff.
    """
    try:
        cf = scf.cf  # type: Crazyflie
        commander = cf.high_level_commander  # type: cflib.HighLevelCOmmander
        cf.param.set_value('commander.enHighLevel', '1')
        cf.param.set_value('ring.effect', '7')
        cf.param.set_value('ring.solidRed', str(0))
        cf.param.set_value('ring.solidGreen', str(255))
        cf.param.set_value('ring.solidBlue', str(0))
        commander.takeoff(1.5, 3.0)
        sleep_while_checking_stable(scf, tf_sec=15)

    except UnstableException as e:
        print(e)
        kill_motor_sequence(scf)
        # raising here since we want to kill entire show if one fails early
        raise(e)
    
    except LowBatteryException as e:
        print(e)
        land_sequence(scf)

    except Exception as e:
        print(e)
        land_sequence(scf)
        raise(e)


def go_sequence(scf: SyncCrazyflie, data: Dict):
    """
    This is the go sequence. It commands the trajectory to start.
    """
    try:
        cf = scf.cf  # type: Crazyflie
        commander = cf.high_level_commander  # type: cflib.HighLevelCOmmander
        commander.start_trajectory(
            trajectory_id=1, time_scale=1.0, relative=False)

        intensity = 1  # 0-1

        # initial led color
        cf.param.set_value('ring.effect', '7')
        cf.param.set_value('ring.solidRed', str(0))
        cf.param.set_value('ring.solidGreen', str(255))
        cf.param.set_value('ring.solidBlue', str(0))
        time.sleep(0.1)

        for color, delay, T in zip(data['color'], data['delay'], data['T']):
            # change led color
            red = int(intensity * color[0])
            green = int(intensity * color[1])
            blue = int(intensity * color[2])
            #print('setting color', red, blue, green)
            sleep_while_checking_stable(scf, tf_sec=delay)
            cf.param.set_value('ring.solidRed', str(red))
            cf.param.set_value('ring.solidBlue', str(blue))
            cf.param.set_value('ring.solidGreen', str(green))
            # wait for leg to complete
            # print('sleeping leg duration', leg_duration)
            sleep_while_checking_stable(scf, tf_sec=T - delay)
        
    except UnstableException as e:
        print(e)
        kill_motor_sequence(scf)
        # raise if you want flight to stop if one drone crashes
        #raise(e)
    
    except LowBatteryException as e:
        print(e)
        # have this vehicle land
        land_sequence(scf)
    
    except Exception as e:
        print(e)
        land_sequence(scf)
        # raising here since we don't know what this exception is
        raise(e)


def kill_motor_sequence(scf: Crazyflie):
    cf = scf.cf
    cf.param.set_value('commander.enHighLevel', '0')
    cf.param.set_value('motorPowerSet.enable', '1')

def land_sequence(scf: Crazyflie):
    try:
        cf = scf.cf  # type: Crazyflie
        commander = cf.high_level_commander  # type: cflib.HighLevelCOmmander
        commander.land(0.0, 3.0)
        print('Landing...')
        sleep_while_checking_stable(scf, tf_sec=3)

        # disable led to save battery
        cf.param.set_value('ring.effect', '0')
        commander.stop()
    except Exception as e:
        print(e)
        kill_motor_sequence(scf)
        raise(e)

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
    
    # disable led to save battery
    cf.param.set_value('ring.effect', '0')
    
    cf.param.set_value('activeMarker.front', str(id[0]))
    time.sleep(1)
    cf.param.set_value('activeMarker.back', str(id[1]))
    time.sleep(1)
    cf.param.set_value('activeMarker.left', str(id[2]))
    time.sleep(1)
    cf.param.set_value('activeMarker.right', str(id[3]))
    time.sleep(1)
    
    print('ID update done! |'+ scf._link_uri)

def swarm_id_update(args):
    cflib.crtp.init_drivers(enable_debug_driver=False)

    data = assignments(int(args.count))
    trajectory_assignment = data['trajectory_assignment']
    body_names = data['body_names']
    rigid_bodies = data['rigid_bodies']
    id_assignment = data['id_assignment']

    factory = CachedCfFactory(rw_cache='./cache')
    uris = {trajectory_assignment[key] for key in trajectory_assignment.keys()}
    id_args = {key: [id_assignment[key]] for key in id_assignment.keys()}
    with Swarm(uris, factory=factory) as swarm:
        print('Starting ID update...')
        #swarm.sequential(id_update, id_args) # parallel update has some issue with more than 3 drones, using sequential update here.
        swarm.sequential(id_update, id_args) # parallel update has some issue with more than 3 drones, using sequential update here.

def hover(args):
    cflib.crtp.init_drivers(enable_debug_driver=False)

    data = assignments(int(args.count))
    trajectory_assignment = data['trajectory_assignment']
    body_names = data['body_names']
    rigid_bodies = data['rigid_bodies']

    factory = CachedCfFactory(rw_cache='./cache')
    uris = {trajectory_assignment[key] for key in trajectory_assignment.keys()}

    qtmWrapper = QtmWrapper(body_names)
    qtm_args = {key: [qtmWrapper, rigid_bodies[key]] for key in rigid_bodies.keys()}

    with Swarm(uris, factory=factory) as swarm:
        
        def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            swarm.parallel(land_sequence)
            time.sleep(1)
            print('Closing QTM link...')
            qtmWrapper.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        print('Press Ctrl+C to land.')
    
        print('Starting mocap data relay...')
        swarm.parallel_safe(send6DOF, qtm_args)

        print('Preflight sequence...')
        swarm.parallel_safe(preflight_sequence)

        print('Takeoff sequence...')
        swarm.parallel_safe(takeoff_sequence)

        print('Land sequence...')
        swarm.parallel(land_sequence)

    print('Closing QTM connection...')
    qtmWrapper.close()

def run(args):
    data = assignments(int(args.count))
    trajectory_assignment = data['trajectory_assignment']
    body_names = data['body_names']
    rigid_bodies = data['rigid_bodies']

    cflib.crtp.init_drivers(enable_debug_driver=False)

    factory = CachedCfFactory(rw_cache='./cache')
    uris = {trajectory_assignment[key] for key in trajectory_assignment.keys()}
    print(uris)
    
    with open(args.json, 'r') as f:
        data = json.load(f)
    swarm_args = {trajectory_assignment[drone_pos]: [data[str(drone_pos)]]
        for drone_pos in trajectory_assignment.keys()}

    qtmWrapper = QtmWrapper(body_names)
    qtm_args = {key: [qtmWrapper, rigid_bodies[key]] for key in rigid_bodies.keys()}

    with Swarm(uris, factory=factory) as swarm:
        
        def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            swarm.parallel(land_sequence)
            time.sleep(1)
            print('Closing QTM link')
            qtmWrapper.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        print('Press Ctrl+C to land.')
    
        try:
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
                    swarm.parallel(upload_trajectory, args_dict=swarm_args)

                print('Go...')
                swarm.parallel_safe(go_sequence, args_dict=swarm_args)
            print('Land sequence...')
            swarm.parallel(land_sequence)

        except Exception as e:
            print(e)
            print('Aborting go sequence, landing')
            swarm.parallel(land_sequence)


    print('Closing QTM connection...')
    qtmWrapper.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')

    parser.add_argument('--json')
    parser.add_argument('--count')
    args = parser.parse_args()

    if args.mode == 'id':
        swarm_id_update(args)
    elif args.mode == 'traj':
        run(args)
    elif args.mode == 'hover':
        hover(args)
    else:
        print('Not a valid input!!!')
