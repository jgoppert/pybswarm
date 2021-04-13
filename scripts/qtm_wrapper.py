import asyncio
import math
import xml.etree.cElementTree as ET
from threading import Thread

import qtm
import time

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
        qtm_instance = await self._discover()
        host = qtm_instance.host
        # host = "192.168.1.2"
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
        intersect = set(self.qtm_6DoF_labels).intersection(body_names) 
        if len(intersect) < n_drones :
            print('Missing rigid bodies')
            print('In qualisys: ', self.qtm_6DoF_labels)
            print('Intersection: ', intersect)
            time.sleep(0.5)
            
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

if __name__ == "__main__":
    wrapper = QtmWrapper(['cf0'])
    wrapper.close()