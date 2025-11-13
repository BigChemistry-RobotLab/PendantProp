from opentrons_api.load_save_functions import load_settings, save_settings
from pendantprop.hardware.opentrons.config import Config
from pendantprop.hardware.droplet_management import DropletManager

# import settings and set exp tag 
settings = load_settings(file_path="config/settings.json")
settings['file_settings']['exp_tag'] = "test_droplet_manager"

# initialise platform
config = Config(settings=settings)
left_pipette, right_pipette, containers = config.load_all()


source = containers["3A1"]
source.sample_id = "TestSample001"

# home robot
config.home()

left_pipette.aspirate(17, source=source)


# config.home()
# # save meta data
# config.save_layout_final()
# save_settings(settings)

# # Log the protocol summary at the end
# config.log_protocol_summary()
