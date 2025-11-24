from opentrons_api.load_save_functions import load_settings
from pendantprop.protocol import Protocol

settings = load_settings(file_path="config/settings.json")
settings['file_settings']['exp_tag'] = "test_protocol_characterisation"
protocol = Protocol(settings=settings)
protocol.characterise_solutions()