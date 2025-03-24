from hardware.opentrons.http_communications import Opentrons_http_api
from hardware.opentrons.configuration import Configuration

api = Opentrons_http_api()
api.initialise()

config = Configuration(opentrons_api=api)

labware = config.load_labware()
pipettes = config.load_pipettes()
containers = config.load_containers()

right_pipette = pipettes["right"]

## excutable robot commands
api.home()
right_pipette.move_to_tip(well="C3")
# right_pipette.pick_up_tip(well="C3")
# right_pipette.drop_tip(return_tip=True)
