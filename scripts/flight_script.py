import time

import cflib.crtp
import csv
import json
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from typing import List, Dict
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie
import signal
import sys

# Change uris and sequences according to your setup
DRONE0 = 'radio://0/70/2M/E7E7E7E701'
DRONE1 = 'radio://0/70/2M/E7E7E7E702'
DRONE2 = 'radio://0/70/2M/E7E7E7E703'
DRONE3 = 'radio://0/70/2M/E7E7E7E704'
DRONE4 = 'radio://0/70/2M/E7E7E7E705'
DRONE5 = 'radio://0/70/2M/E7E7E7E706'
DRONE6 = 'radio://1/80/2M/E7E7E7E707'
DRONE7 = 'radio://1/80/2M/E7E7E7E708'
DRONE8 = 'radio://1/80/2M/E7E7E7E709'
DRONE9 = 'radio://1/80/2M/E7E7E7E710'
DRONE10 = 'radio://1/80/2M/E7E7E7E711'
DRONE11 = 'radio://0/80/2M/E7E7E7E712'
DRONE12 = 'radio://0/80/2M/E7E7E7E713'
DRONE13 = 'radio://0/80/2M/E7E7E7E714'
DRONE14 = 'radio://0/80/2M/E7E7E7E715'
DRONE15 = 'radio://0/80/2M/E7E7E7E716'
DRONE16 = 'radio://0/80/2M/E7E7E7E717'
DRONE17 = 'radio://0/80/2M/E7E7E7E718'
DRONE18 = 'radio://0/80/2M/E7E7E7E719'


# List of URIs, comment the one you do not want to fly
#DRONE4 ## Faulty Drone // Does not work
trajectory_assigment = {
    0: DRONE1,
    1: DRONE0,
    2: DRONE3,
    3: DRONE2,
    4: DRONE5,
    5: DRONE6,
    6: DRONE7,
    #7: DRONE17,
    #8: DRONE18,
    #9: DRONE10,
    #10: DRONE11,
    #11: DRONE12,
    #12: DRONE13,
}


class Uploader:
    def __init__(self):
        self._is_done = False

    def upload(self, trajectory_mem):
        print('Uploading data')
        trajectory_mem.write_data(self._upload_done)

        while not self._is_done:
            time.sleep(0.2)

    def _upload_done(self, mem, addr):
        print('Data uploaded')
        self._is_done = True

def position_callback(timestamp, data, logconf):
    x = data['kalman.stateX']
    y = data['kalman.stateY']
    z = data['kalman.stateZ']
    print('pos: ({}, {}, {})'.format(x, y, z))

def start_position_printing(scf):
    log_conf = LogConfig(name='Position', period_in_ms=500)
    log_conf.add_variable('kalman.stateX', 'float')
    log_conf.add_variable('kalman.stateY', 'float')
    log_conf.add_variable('kalman.stateZ', 'float')

    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(position_callback)
    log_conf.start()

def wait_for_position_estimator(scf: SyncCrazyflie):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break


def wait_for_param_download(scf: SyncCrazyflie):
    print("param")
    while not scf.cf.param.is_updated:
        time.sleep(1.0)
    print('Parameters downloaded for', scf.cf.link_uri)

def reset_estimator(scf: SyncCrazyflie):
    print("reset")
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    wait_for_position_estimator(cf)

def upload_trajectory(cf: Crazyflie, trajectory_id: int, trajectory: List):
    trajectory_mem = cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]

    total_duration = 0
    for row in trajectory:
        duration = row[0]
        x = Poly4D.Poly(row[1:9])
        y = Poly4D.Poly(row[9:17])
        z = Poly4D.Poly(row[17:25])
        yaw = Poly4D.Poly(row[25:33])
        trajectory_mem.poly4Ds.append(Poly4D(duration, x, y, z, yaw))
        total_duration += duration

    Uploader().upload(trajectory_mem)
    cf.high_level_commander.define_trajectory(trajectory_id, 0,
                                              len(trajectory_mem.poly4Ds))
    return total_duration

def upload_sequence(scf: Crazyflie,trajectory: List, duration: float):
    try:
        print("upload")
        cf = scf.cf # type: Crazyflie
        trajectory_id = 1
        cf.param.set_value('commander.enHighLevel','1')
        upload_trajectory(cf, trajectory_id, trajectory)
        reset_estimator(scf)
    except Exception as e:
        print(e)

def go_sequence(scf: Crazyflie,trajectory: List, duration: float):
    try:
        cf = scf.cf # type: Crazyflie
        trajectory_id = 1       
        commander = cf.high_level_commander # type: cflib.HighLevelCOmmander
        print("go")
        commander.takeoff(1.0, 3.0)
        time.sleep(6.0)
        relative = False
        commander.start_trajectory(trajectory_id, 1.0, relative)
        time.sleep(duration)
    except Exception as e:
        print(e)

def land_sequence(scf: Crazyflie,trajectory: List, duration: float):
    try:
        cf = scf.cf # type: Crazyflie
        commander = cf.high_level_commander # type: cflib.HighLevelCOmmander        
        print("land")
        commander.land(0.0, 5.0)
        time.sleep(5)
        commander.stop()
    except Exception as e:
        print(e)



if __name__ == '__main__':

    # logging.basicConfig(level=logging.DEBUG)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    factory = CachedCfFactory(rw_cache='./cache')
    uris = {trajectory_assigment[key] for key in trajectory_assigment.keys()}
    print("uris:", uris)
    with open('scripts/data/p_form.json', 'r') as f:
        traj_list = json.load(f)
    
    #building arguments list in swarm
    swarm_args = {}
    for key in trajectory_assigment.keys():
        print("key", key)
        trajectory = traj_list[str(key)]
        print("trajectory assigned")
        duration = 0
        print("duration assigned")
        for leg in trajectory:
            duration += leg[0]
        swarm_args[trajectory_assigment[key]] = [trajectory, duration]
        print("trajectory URI:", trajectory_assigment[key])
        print("duration:", duration)
                


    with Swarm(uris, factory=factory) as swarm:
        def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            swarm.parallel(land_sequence, args_dict=swarm_args)
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        print('Press Ctrl+C')
    
        # If the copters are started in their correct positions this is
        # probably not needed. The Kalman filter will have time to converge
        # any way since it takes a while to start them all up and connect. We
        # keep the code here to illustrate how to do it.
        print('Resetting estimators...')
        #swarm.parallel(reset_estimator)

        # The current values of all parameters are downloaded as a part of the
        # connections sequence. Since we have 10 copters this is clogging up
        # communication and we have to wait for it to finish before we start
        # flying.
        print('Waiting for parameters to be downloaded...')
        swarm.parallel(wait_for_param_download)

        #print('Setting up trajectories...')
        #swarm.parallel(setup_trajectory, args_dict = swarm_args)

        print('Running trajectory...')
        swarm.parallel(upload_sequence, args_dict=swarm_args)
        swarm.parallel(start_position_printing, args_dict = swarm_args)
        swarm.parallel(go_sequence, args_dict=swarm_args)
        swarm.parallel(land_sequence, args_dict=swarm_args)
