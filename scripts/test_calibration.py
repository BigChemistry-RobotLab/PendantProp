from opentrons_api.load_save_functions import load_settings, save_settings
from pendantprop.hardware.opentrons.config import Config
from pendantprop.hardware.droplet_management import DropletManager

# import settings and set exp tag 
settings = load_settings(file_path="config/settings.json")
settings['file_settings']['exp_tag'] = "test_calibration"

# initialise platform
config = Config(settings=settings)
left_pipette, right_pipette, containers = config.load_all()
droplet_manager = DropletManager(
    settings=settings,
    left_pipette=left_pipette,
    containers=containers,
)

source = containers["8A1"]
source.sample_id = "TestSample001"

# home robot
config.home()

average_scale = droplet_manager.calibrate(source=source, vol_droplet=19.0, calibration_time=60.0)
config.logger.info(f"Average scale from calibration: {average_scale}")

config.home()
settings['image_analysis_settings']['scale'] = average_scale

# save meta data
config.save_layout_final()
save_settings(settings, file_path = "config/settings_calibration.json")

# Log the protocol summary at the end
config.log_protocol_summary()