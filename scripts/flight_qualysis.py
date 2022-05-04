#!/usr/bin/env python3
import enum
import time
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
from enum import Enum
import logging

import qtm

# set logging levels for crazyflie
logger = logging.getLogger('flight_qualisys')
logging.getLogger('qtm').setLevel(logging.WARN)
logging.getLogger('cflib.crazyflie.log').setLevel(logging.ERROR)
logging.getLogger('cflib.crazyflie').setLevel(logging.INFO)

class FlightStage(Enum):
    INIT=0
    PREFLIGHT=1
    UPLOAD=2
    TAKEOFF=3
    GO=4
    LAND=5
    ABORT=6

if sys.version_info[0] != 3:
    print("This script requires Python 3")
    exit()

# Change uris and sequences according to your setup
DRONES = [
    'radio://2/80/250K/E7E7E7E7E8',

    'radio://0/0/250K/E7E7E7E7E0',
    'radio://1/40/250K/E7E7E7E7E1',
    'radio://2/80/250K/E7E7E7E7E2',

    'radio://0/0/250K/E7E7E7E7E3',
    'radio://1/40/250K/E7E7E7E7E4',
    'radio://2/80/250K/E7E7E7E7E5',

    'radio://0/0/250K/E7E7E7E7E6',
    'radio://1/40/250K/E7E7E7E7E7',
]

SEND_FULL_POSE = True



def assignments(count):
    if count > len(DRONES) or count <= 0:
        msg = 'unknown number of drones'
        logger.error(msg)
        raise ValueError(msg)

    uris = DRONES[:count]
    n_drones = len(uris)

    body_names = []
    led_id = [[1,3,4,2]]
    for i in range(n_drones):
        body_names.append('cf'+str(i))
        if i != 0:
            led_id.append([i*4 + j for j in led_id[0]])

    # print('Defined bodies are: ', body_names)
    # print('Deck ids are:', led_id)

    uri_to_traj_id = {uri: i for i, uri in enumerate(uris)}
    uri_to_led_id = {uri: led_id[i] for i, uri in enumerate(uris)}
    uri_to_rigid_body = {uri: body_names[i] for i, uri in enumerate(uris)}
    return {
        'uris': uris,
        'uri_to_traj_id': uri_to_traj_id,
        'uri_to_rigid_body': uri_to_rigid_body,
        'uri_to_led_id': uri_to_led_id,
        'n_drones': n_drones,
        'body_names': body_names
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
        logger.info('Connecting to QTM on %s', host)
        self.connection = await qtm.connect(host=host, version="1.20") # version 1.21 has weird 6DOF labels, so using 1.20 here

        params = await self.connection.get_parameters(parameters=['6d'])
        xml = ET.fromstring(params)
        self.qtm_6DoF_labels = [label.text for label in xml.iter('Name')]

        await self.connection.stream_frames(
            components=['6D'],
            on_packet=self._on_packet)
        logger.info('Connected to QTM')


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
            logger.error('Missing rigid bodies')
            logger.error('In QTM: %s', str(self.qtm_6DoF_labels))
            logger.error('Intersection: %s', str(intersect))
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
                        logger.error("======= LOST RB TRACKING : %s", body_name)
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
        logger.info('uploading')
        self.trajectory_mem.write_data(self._upload_done)
        while not self._is_done:
            time.sleep(1)

    def _upload_done(self, mem, addr):
        logger.info('upload is done')
        self._is_done = True
        self.trajectory_mem.disconnect()


def check_battery(scf: SyncCrazyflie, min_voltage=4):
    logger.info('Checking battery...')
    log_config = LogConfig(name='Battery', period_in_ms=500)
    log_config.add_variable('pm.vbat', 'float')

    with SyncLogger(scf, log_config) as sync_logger:
        for log_entry in sync_logger:
            log_data = log_entry[1]
            vbat = log_data['pm.vbat']
            if vbat < min_voltage:
                msg = "battery too low: {:10.4f} V, for {:s}".format(
                    vbat, scf.cf.link_uri)
                logger.error(msg)
                raise LowBatteryException(msg)
            else:
                return


def check_state(scf: SyncCrazyflie):
    logger.info('Checking state.')
    log_config = LogConfig(name='State', period_in_ms=500)
    log_config.add_variable('stabilizer.roll', 'float')
    log_config.add_variable('stabilizer.pitch', 'float')
    log_config.add_variable('stabilizer.yaw', 'float')
    logger.info('Log configured.')

    with SyncLogger(scf, log_config) as sync_logger:
        for log_entry in sync_logger:
            log_data = log_entry[1]
            roll = log_data['stabilizer.roll']
            pitch = log_data['stabilizer.pitch']
            yaw = log_data['stabilizer.yaw']
            logger.info('%s Checking roll/pitch/yaw', scf.cf.link_uri)

            #('yaw', yaw)
            if SEND_FULL_POSE:
                euler_checks = [('roll', roll, 5), ('pitch', pitch, 5)]
            else:
                euler_checks = [('roll', roll, 5), ('pitch', pitch, 5), ('yaw', yaw, 5)]

            for name, val, val_max in euler_checks:
                if np.abs(val) > val_max:
                    msg = "too much {:s}, {:10.4f} deg, for {:s}".format(
                        name, val, scf.cf.link_uri)
                    logger.error(msg)
                    raise UnstableException(msg)
            return


def wait_for_position_estimator(scf: SyncCrazyflie, duration: float = 10.0):
    logger.info('Waiting for estimator to find position...')

    dt = 0.5
    log_config = LogConfig(name='Kalman Variance', period_in_ms=dt*1000)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001
    t = 0

    with SyncLogger(scf.cf, log_config) as sync_logger:
        for log_entry in sync_logger:
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
                if t > duration:
                    msg = "estimator failed after {} seconds: {:s}\t{:10g}\t{:10g}\t{:10g}".format(
                            duration, scf.cf.link_uri, max_x - min_x, max_y - min_y, max_z - min_z)
                    logger.error(msg)
                    raise Exception(msg)
            t += dt


def wait_for_param_download(scf: SyncCrazyflie):
    while not scf.cf.param.is_updated:
        time.sleep(1.0)
    logger.info('Parameters downloaded for %s', scf.cf.link_uri)


def reset_estimator(scf: SyncCrazyflie):
    logger.info('Resetting estimator...')
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    wait_for_position_estimator(scf, 10)

def activate_kalman_estimator(cf):
    cf.param.set_value('stabilizer.estimator', '2')

    # Set the std deviation for the quaternion data pushed into the
    # kalman filter. The default value seems to be a bit too low.
    cf.param.set_value('locSrv.extQuatStdDev', 0.06)

def sleep_while_checking_stable(scf: SyncCrazyflie, tf_sec, dt_sec=0.1, check_low_battery=True):
    if tf_sec == 0:
        return
    log_config = LogConfig(name='Roll', period_in_ms=int(dt_sec*1000.0))
    log_config.add_variable('stabilizer.roll', 'float')
    log_config.add_variable('pm.vbat', 'float')
    t_sec = 0
    batt_lowpass = 3.7
    RC = 0.25 # time constant in seconds of first order low pass filter
    alpha = dt_sec/(RC + dt_sec)
    # print('sleeping {:10.4f} seconds'.format(tf_sec))
    with SyncLogger(scf, log_config) as sync_logger:
        for log_entry in sync_logger:
            log_data = log_entry[1]
            roll = log_data['stabilizer.roll']
            batt = log_data['pm.vbat']
            if np.abs(roll) > 60:
                msg = "flip detected {:10.4f} deg, for {:s}".format(roll, scf.cf.link_uri)
                logger.error(msg)
                raise UnstableException(msg)
            batt_lowpass = (1 - alpha)*batt_lowpass + alpha*batt
            # print("battery {:10.4f} V, for {:s}".format(batt_lowpass, scf.cf.link_uri))
            if batt_lowpass < 2.9 and check_low_battery:
                msg = "low battery {:10.4f} V, for {:s}".format(batt, scf.cf.link_uri)
                logger.error(msg)
                raise LowBatteryException(msg)
            t_sec += dt_sec
            if t_sec>tf_sec:
                return


def upload_trajectory(scf: SyncCrazyflie, data: Dict):
    logger.info('uploading trajectory at flight stage: %s', data['flight_stage'])

    cf = scf.cf  # type: Crazyflie

    trajectory_mem = scf.cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]

    TRAJECTORY_MAX_LENGTH = 31
    trajectory = data['traj']['trajectory']

    if len(trajectory) > TRAJECTORY_MAX_LENGTH:
        msg = "Trajectory too long for drone {:s}".format(cf.link_uri)
        logger.error(msg)
        raise ValueError(msg)

    for row in trajectory:
        duration = row[0]
        x = Poly4D.Poly(row[1:9])
        y = Poly4D.Poly(row[9:17])
        z = Poly4D.Poly(row[17:25])
        yaw = Poly4D.Poly(row[25:33])
        trajectory_mem.poly4Ds.append(Poly4D(duration, x, y, z, yaw))

    logger.info('%s Calling upload method', scf.cf.link_uri)
    uploader = Uploader(trajectory_mem)
    uploader.upload()
    
    logger.info('%s Defining trajectory', scf.cf.link_uri)
    cf.high_level_commander.define_trajectory(
        trajectory_id=1, offset=0, n_pieces=len(trajectory_mem.poly4Ds))

    data['flight_stage'] = FlightStage.UPLOAD


def preflight_sequence(scf: SyncCrazyflie, data: Dict):
    """
    This is the preflight sequence. It calls all other subroutines before takeoff.
    """
    logger.info('%s preflight at flight stage: %s', scf.cf.link_uri, data['flight_stage'])

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

    data['flight_stage'] = FlightStage.PREFLIGHT


def takeoff_sequence(scf: SyncCrazyflie, data: Dict):
    """
    This is the takeoff sequence. It commands takeoff.
    """
    logger.info('%s takeoff at flight stage: %s', scf.cf.link_uri, data['flight_stage'])

    try:
        cf = scf.cf  # type: Crazyflie
        commander = cf.high_level_commander  # type: cflib.HighLevelCOmmander
        cf.param.set_value('commander.enHighLevel', '1')
        cf.param.set_value('ring.effect', '7')
        cf.param.set_value('ring.solidRed', str(0))
        cf.param.set_value('ring.solidGreen', str(255))
        cf.param.set_value('ring.solidBlue', str(0))
        commander.takeoff(1.5, 3.0)
        sleep_while_checking_stable(scf, tf_sec=10)

    except UnstableException as e:
        logger.error('%s unstable exception: %s', scf.cf.link_uri, e)
        kill_motor_sequence(scf, data)
        # raising here since we want to kill entire show if one fails early
        raise(e)
    
    except LowBatteryException as e:
        logger.error('low battery exception: %s', e)
        kill_motor_sequence(scf, data)

    except Exception as e:
        logger.error('%s general exception: %s', scf.cf.link_uri, e)
        kill_motor_sequence(scf, data)
        raise(e)

    data['flight_stage'] = FlightStage.TAKEOFF


def go_sequence(scf: SyncCrazyflie, data: Dict):
    """
    This is the go sequence. It commands the trajectory to start.
    """
    logger.info('%s go at flight stage: %s', scf.cf.link_uri, data['flight_stage'])
    data['flight_stage'] = FlightStage.TAKEOFF

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

        traj = data['traj']
        for color, delay, T in zip(traj['color'], traj['delay'], traj['T']):
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
        logger.error('go general exceptoin: %s', e)
        kill_motor_sequence(scf, data)
        # raise if you want flight to stop if one drone crashes
        #raise(e)
    
    except LowBatteryException as e:
        logger.error('go low battery exception: %s', e)
        # have this vehicle land
        land_sequence(scf, data)
    
    except Exception as e:
        logger.error('%s go general exception: %s', scf.cf.link_uri, e)
        land_sequence(scf, data)
        # raising here since we don't know what this exception is
        raise(e)


def kill_motor_sequence(scf: Crazyflie, data: Dict):
    logger.info('%s killing motors at flight stage: %s', scf.cf.link_uri, data['flight_stage'])
    cf = scf.cf
    cf.param.set_value('commander.enHighLevel', '0')
    cf.param.set_value('motorPowerSet.enable', '1')
    data['flight_stage'] = FlightStage.ABORT


def land_sequence(scf: Crazyflie, data: Dict):
    logger.info('%slanding at flight stage: %s', scf.cf.link_uri, data['flight_stage'])
    flight_stage = data['flight_stage']
    if flight_stage == FlightStage.ABORT or \
        flight_stage == FlightStage.LAND or \
            flight_stage.value < FlightStage.TAKEOFF.value:
        kill_motor_sequence(scf, data)
        logger.error('land called, but not flying, killing motors')
        return
    
    try:
        cf = scf.cf  # type: Crazyflie
        commander = cf.high_level_commander  # type: cflib.HighLevelCOmmander
        commander.land(0.2, 5.0)
        logger.info('%s Landing...', scf.cf.link_uri)
        sleep_while_checking_stable(scf, tf_sec=5, check_low_battery=False)

        # disable led to save battery
        cf.param.set_value('ring.effect', '0')
        commander.stop()
        kill_motor_sequence(scf, data)

    except UnstableException as e:
        logger.error('%s land unstable exception: %s', scf.cf.link_uri, e)
        kill_motor_sequence(scf, data)
    
    except Exception as e:
        logger.error('%s land general exception: %s', scf.cf.link_uri, e)
        kill_motor_sequence(scf, data)

    data['flight_stage'] = FlightStage.LAND


def send6DOF(scf: SyncCrazyflie, data: Dict):
    """
    This relays mocap data to each crazyflie
    """
    try:
        data['qtmWrapper'].on_pose[data['rigid_body']] = lambda pose: send_extpose_rot_matrix(scf, pose[0], pose[1], pose[2], pose[3])
    except Exception as e:
        logger.error('send6DOF general exception: %s', e)
        logger.error('Could not relay mocap data')
        land_sequence(scf, data)

def id_update(scf: SyncCrazyflie, id: List):
    cf = scf.cf
    cf.param.set_value('activeMarker.mode', '3') # qualisys mode
    time.sleep(1)

    # default id
    # [f, b, l, r] 
    # [1, 3, 4, 2]
    
    # disable led to save battery
    cf.param.set_value('ring.effect', '0')
    t_sleep = 1 # seconds
    cf.param.set_value('activeMarker.front', str(id[0]))
    time.sleep(t_sleep)
    cf.param.set_value('activeMarker.back', str(id[1]))
    time.sleep(t_sleep)
    cf.param.set_value('activeMarker.left', str(id[2]))
    time.sleep(t_sleep)
    cf.param.set_value('activeMarker.right', str(id[3]))
    time.sleep(t_sleep)
    
    logger.info('ID update done! | %s', scf._link_uri)

def swarm_id_update(args):
    cflib.crtp.init_drivers(enable_debug_driver=False)
    assign = assignments(int(args.count))
    factory = CachedCfFactory(rw_cache='./cache')
    swarm_args = {
        uri: [assign['uri_to_led_id'][uri]]
        for uri in assign['uris']
    }

    with Swarm(assign['uris'], factory=factory) as swarm:
        logger.info('Starting ID update...')
        swarm.sequential(id_update, swarm_args) # parallel update has some issue with more than 3 drones, using sequential update here.

def hover(args):
    cflib.crtp.init_drivers(enable_debug_driver=False)
    assign = assignments(int(args.count))
    factory = CachedCfFactory(rw_cache='./cache')
    qtmWrapper = QtmWrapper(assign['body_names'])
    swarm_args = {}
    for uri in assign['uris']:
        swarm_args[uri] = [{
            'qtmWrapper': qtmWrapper,
            'rigid_body': assign['uri_to_rigid_body'][uri],
            'flight_stage': FlightStage.INIT
        }]

    with Swarm(assign['uris'], factory=factory) as swarm:

        logger.info('swarm links created')
        
        try:
            logger.info('Starting mocap data relay...')
            swarm.parallel_safe(send6DOF, swarm_args)

            logger.info('Preflight sequence...')
            swarm.parallel_safe(preflight_sequence, swarm_args)

            logger.info('Takeoff sequence...')
            swarm.parallel(takeoff_sequence, swarm_args)
            
        except KeyboardInterrupt:
            pass

        except Exception as e:
            print(e)

        logger.info('Land sequence...')
        swarm.parallel(land_sequence, swarm_args)
        
    logger.info('Closing QTM connection...')
    qtmWrapper.close()

def run(args):
    assign = assignments(int(args.count))
    cflib.crtp.init_drivers(enable_debug_driver=False)
    factory = CachedCfFactory(rw_cache='./cache')
    with open(args.json, 'r') as f:
        traj_data = json.load(f)
    qtmWrapper = QtmWrapper(assign['body_names'])
    swarm_args = {}
    for uri in assign['uris']:
        swarm_args[uri] = [{
            'qtmWrapper': qtmWrapper,
            'rigid_body': assign['uri_to_rigid_body'][uri],
            'flight_stage': FlightStage.INIT,
            'traj': traj_data[str(assign['uri_to_traj_id'][uri])]
        }]

    with Swarm(assign['uris'], factory=factory) as swarm:
        
        try:
            logger.info('Starting mocap data relay...')
            swarm.parallel_safe(send6DOF, swarm_args)

            logger.info('Preflight sequence...')
            swarm.parallel_safe(preflight_sequence, swarm_args)

            logger.info('Upload sequence...')
            swarm.parallel_safe(upload_trajectory, swarm_args)
                
            logger.info('Takeoff sequence...')
            swarm.parallel(takeoff_sequence, swarm_args)

            # repeat = int(data['repeat'])
            repeat = 1
            for trajectory_count in range(repeat):
                logger.info('Go sequence...')
                swarm.parallel(go_sequence, swarm_args)

        except KeyboardInterrupt:
            pass

        except Exception as e:
            logger.error('swarm general exception: %s', e)

        logger.info('Land sequence...')
        swarm.parallel(land_sequence, swarm_args)

    logger.info('Closing QTM connection...')
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
        logger.error('Not a valid input!!!')
