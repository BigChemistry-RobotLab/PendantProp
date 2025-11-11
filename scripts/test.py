"""
Example: Using PendantProp custom containers with Opentrons robot
This example demonstrates how to use LightHolder and DropStage in a protocol.
"""

from opentrons_api.load_save_functions import load_settings, save_settings
from pendantprop.hardware.opentrons.config import Config  # Use PendantProp Config!

# Import settings and set experiment tag
settings = load_settings(file_path="config/settings.json")
settings['EXP_TAG'] = "example_pendant_drop"

# Initialize platform with PendantProp Config
# This Config recognizes LightHolder and DropStage containers
config = Config(settings=settings)
left_pipette, right_pipette, containers = config.load_all()

config.home()

right_pipette.pick_up_tip()
left_pipette.pick_up_tip()

right_pipette.drop_tip()
left_pipette.drop_tip()
