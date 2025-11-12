"""
Example: Using PendantProp custom containers with Opentrons robot
This example demonstrates how to use LightHolder and DropStage in a protocol.
"""

from opentrons_api.load_save_functions import load_settings, save_settings
from pendantprop.hardware.opentrons.config import Config  # Use PendantProp Config!
from pendantprop.hardware.opentrons.washing import Washer

# Import settings and set experiment tag
settings = load_settings(file_path="config/settings.json")

# Initialize platform with PendantProp Config
config = Config(settings=settings)
left_pipette, right_pipette, containers = config.load_all()

config.home()
left_pipette.pick_up_tip()
washer = Washer(
    settings=settings,
    left_pipette=left_pipette,
    right_pipette=right_pipette,
    containers=containers,
    labware=config.labware
    )

# Example washing procedure using the Washer class
washer.wash()




