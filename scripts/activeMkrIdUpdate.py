import logging
import time

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = 'radio://0/81/2M/E7E7E7E771'
# URI = 'usb://0'

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

def updateId(ids = [11,31,41,21]):
    with SyncCrazyflie(URI) as scf:
        print('starting ID update')

        cf = scf.cf
        cf.param.set_value('activeMarker.mode', '3') # qualisys mode
        time.sleep(1)

        # default id
        # [f, b, l, r] 
        # [1, 3, 4, 2]

        cf.param.set_value('activeMarker.front', str(ids[0]))
        time.sleep(1)
        cf.param.set_value('activeMarker.back', str(ids[1]))
        time.sleep(1)
        cf.param.set_value('activeMarker.left', str(ids[2]))
        time.sleep(1)
        cf.param.set_value('activeMarker.right', str(ids[3]))
        time.sleep(1)
        
        print('done!')


if __name__ == '__main__':
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)
    updateId([5,6,7,8])

   