from opentrons_api.load_save_functions import load_settings, save_settings
from opentrons_api.config import Config

# import settings and set exp tag 
settings = load_settings(file_path="config/settings.json")
settings['EXP_TAG'] = "example1"

# initialise platform
config = Config(settings=settings)
left_pipette, right_pipette, containers = config.load_all()

# home robot
config.home()

# protocol
left_pipette.pick_up_tip()
left_pipette.drop_tip()
right_pipette.pick_up_tip()
right_pipette.drop_tip()

# save meta data
config.save_layout_final()
save_settings(settings)

# Log the protocol summary at the end
config.log_protocol_summary()
