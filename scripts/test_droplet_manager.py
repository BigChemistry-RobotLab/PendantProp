from opentrons_api.load_save_functions import load_settings, save_settings
from pendantprop.hardware.opentrons.config import Config
from pendantprop.hardware.droplet_management import DropletManager
from pendantprop.utils.utils import calculate_equillibrium_value

# import settings and set exp tag 
settings = load_settings(file_path="config/settings.json")
settings['file_settings']['exp_tag'] = "test_droplet_manager"

# initialise platform
config = Config(settings=settings)
left_pipette, right_pipette, containers = config.load_all()
droplet_manager = DropletManager(
    settings=settings,
    left_pipette=left_pipette,
    containers=containers,
)

source = containers["7A1"]
source.sample_id = "TestSample001"

# home robot
config.home()

dynamic_surface_tension, drop_volume, max_measure_time, drop_count = droplet_manager.measure_pendant_drop(source=source)
equilibrium_surface_tension = calculate_equillibrium_value(dynamic_surface_tension, n_eq_points=100, column_index=1)
config.home()
# # save meta data
# config.save_layout_final()
# save_settings(settings)

# # Log the protocol summary at the end
# config.log_protocol_summary()
