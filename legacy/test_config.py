from hardware.opentrons.http_communications import Opentrons_http_api
from hardware.opentrons.configuration import Configuration

# initialize
api = Opentrons_http_api()
api.initialise()
config = Configuration(opentrons_api=api)
labware = config.load_labware()
pipettes = config.load_pipettes()
containers = config.load_containers()
right_pipette = pipettes["right"]
left_pipette = pipettes["left"]

print(containers)
# print(containers["3A1"])
# print(containers["6A1"])
# #! excutable robot commands
# api.home()
# left_pipette.pick_up_tip()
# left_pipette.measure_pendant_drop(
#     source=containers["water"],
#     destination=containers["drop_stage"],
#     drop_volume=10,
#     delay=10,
#     dispense_rate=100,
# )
# left_pipette.drop_tip(return_tip=True)