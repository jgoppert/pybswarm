import time
import signal
import sys
import json
import argparse
import numpy as np
from typing import Dict

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from typing import List, Dict
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie

if sys.version_info[0] != 3:
    print("This script requires Python 3")
    exit()

# Change uris and sequences according to your setup
DRONE0 = 'radio://0/70/2M/E7E7E7E701'
DRONE1 = 'radio://0/70/2M/E7E7E7E702'
DRONE2 = 'radio://0/70/2M/E7E7E7E703'
DRONE3 = 'radio://0/70/2M/E7E7E7E704'
DRONE4 = 'radio://0/70/2M/E7E7E7E705'
DRONE5 = 'radio://0/70/2M/E7E7E7E706'
DRONE6 = 'radio://0/80/2M/E7E7E7E707'
DRONE7 = 'radio://0/80/2M/E7E7E7E708'
DRONE8 = 'radio://0/80/2M/E7E7E7E709'
DRONE9 = 'radio://0/80/2M/E7E7E7E710'
DRONE10 = 'radio://0/80/2M/E7E7E7E711'
DRONE11 = 'radio://0/90/2M/E7E7E7E712'
DRONE12 = 'radio://0/90/2M/E7E7E7E713'
DRONE13 = 'radio://0/90/2M/E7E7E7E714'
DRONE14 = 'radio://0/90/2M/E7E7E7E715'
DRONE15 = 'radio://0/100/2M/E7E7E7E716'
DRONE16 = 'radio://0/100/2M/E7E7E7E717'
DRONE17 = 'radio://0/100/2M/E7E7E7E718'
DRONE18 = 'radio://0/100/2M/E7E7E7E719'
DRONE19 = 'radio://0/100/2M/E7E7E7E720'
DRONE20 = 'radio://0/100/2M/E7E7E7E721'

# List of URIs, comment the one you do not want to fly
# DRONE4 ## Faulty Drone // Does not work
trajectory_assignment = {
    0: DRONE15,
    1: DRONE19,
    2: DRONE2,
    3: DRONE9,
    4: DRONE16,
    5: DRONE12,
    # 6: DRONE7,
    # 7: DRONE17,
    # 8: DRONE18,
    # 9: DRONE10,
    # 10: DRONE11,
    # 11: DRONE12,
    # 12: DRONE13,
}

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


def check_battery(scf: SyncCrazyflie, min_voltage=4.0):
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


def upload_trajectory(scf: SyncCrazyflie, data: Dict):
    try:
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
        # enable high level commander
        cf.param.set_value('commander.enHighLevel', '1')

        # ensure params are downloaded
        wait_for_param_download(scf)

        # make sure not already flying
        land_sequence(scf)

        # disable LED to save battery
        cf.param.set_value('ring.effect', '0')

        # set pid gains, tune down Kp to deal with UWB noise
        cf.param.set_value('posCtlPid.xKp', '1')
        cf.param.set_value('posCtlPid.yKp', '1')
        cf.param.set_value('posCtlPid.zKp', '1')

        # check battery level
        check_battery(scf, 4.0)

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
        cf.param.set_value('ring.effect', '7')
        cf.param.set_value('ring.solidRed', str(0))
        cf.param.set_value('ring.solidGreen', str(0))
        cf.param.set_value('ring.solidBlue', str(0))
        commander.takeoff(2, 3.0)
        time.sleep(3.0)
    except Exception as e:
        print(e)
        land_sequence(scf)


def go_sequence(scf: Crazyflie, data: Dict):
    """
    This is the go sequence. It commands the trajectory to start and runs the lighting.
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
            time.sleep(delay)
            cf.param.set_value('ring.solidRed', str(red))
            cf.param.set_value('ring.solidBlue', str(blue))
            cf.param.set_value('ring.solidGreen', str(green))
            # wait for leg to complete
            # print('sleeping leg duration', leg_duration)
            time.sleep(T - delay)

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
        # disable led to save battery
        cf.param.set_value('ring.effect', '0')
        commander.stop()
    except Exception as e:
        print(e)
        land_sequence(scf)

def run(args):

    # logging.basicConfig(level=logging.DEBUG)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    factory = CachedCfFactory(rw_cache='./cache')
    uris = {trajectory_assignment[key] for key in trajectory_assignment.keys()}
    with open(args.json, 'r') as f:
        data = json.load(f)
    swarm_args = {trajectory_assignment[drone_pos]: [data[str(drone_pos)]]
        for drone_pos in trajectory_assignment.keys()}

    with Swarm(uris, factory=factory) as swarm:

        def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            swarm.parallel(land_sequence)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        print('Press Ctrl+C to land.')

        print('Preflight sequence...')
        swarm.parallel_safe(preflight_sequence)

        #print('Takeoff sequence...')
        swarm.parallel_safe(takeoff_sequence)

        print('Upload sequence...')
        trajectory_count = 0

        repeat = int(data['repeat'])

        for trajectory_count in range(repeat):
            if trajectory_count == 0:
                print('Uploading Trajectory')
                swarm.parallel(upload_trajectory, args_dict=swarm_args)

            print('Go...')
            swarm.parallel_safe(go_sequence, args_dict=swarm_args)
        
        print('Land sequence...')
        swarm.parallel(land_sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json')
    args = parser.parse_args()
    run(args)
