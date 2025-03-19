from hardware.opentrons.protocol import Protocol
from hardware.opentrons.http_communications import OpentronsAPI
from hardware.cameras import PendantDropCamera
from hardware.sensor.sensor_api import SensorAPI

opentron_api = OpentronsAPI()
sensor_api = SensorAPI()
pd_cam = PendantDropCamera()
protocol = Protocol(opentrons_api=opentron_api, sensor_api=sensor_api, pendant_drop_camera=pd_cam)
protocol.characterize_surfactant()
